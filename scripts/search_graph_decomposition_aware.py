import argparse
import json
import re
from collections import defaultdict, deque

import numpy as np
import torch

try:
    import spacy
except ModuleNotFoundError:
    spacy = None
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from search_graph_hopaware import (
    build_hetero_graph,
    build_semantic_sim_map,
    entity_entry_n,
    extract_keywords_simple,
    extract_numbers,
    get_sim,
    make_personalization,
    norm_text,
    ppr,
    relation_entry_r,
)


MONTH_MARKERS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
}
TIME_MARKERS = {
    "before", "after", "during", "since", "until", "when", "while",
    "earlier", "later", "former", "current", "previous", "next",
    "first", "last", "century", "year", "years", "month", "months",
    "day", "days", "week", "weeks",
} | MONTH_MARKERS
QUANTITY_MARKERS = {
    "more", "less", "least", "most", "over", "under", "below", "above",
    "greater", "smaller", "higher", "lower", "equal", "exactly",
    "approximately", "around", "nearly", "roughly", "percent", "percentage",
    "million", "billion", "thousand", "dozen", "twice", "half",
}
NEGATION_MARKERS = {
    "no", "not", "never", "without", "none", "neither", "nor", "non",
    "n't", "cannot", "can't", "didn't", "doesn't", "isn't", "wasn't",
    "weren't", "won't", "hasn't", "haven't", "hadn't",
}
CONTEXT_DEPENDENT_STARTS = {
    "it", "he", "she", "they", "this", "that", "these", "those",
    "such", "former", "latter", "his", "her", "their", "its",
    "also", "however", "meanwhile", "therefore", "then",
}
GENERIC_BACKGROUND_PATTERNS = (
    "there is", "there was", "there are", "there were",
    "it is", "it was", "they are", "they were",
)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_spacy_model(model_name: str):
    if spacy is None:
        class _SimpleDoc:
            def __init__(self):
                self.ents = []

        class _SimpleNLP:
            def __call__(self, _text):
                return _SimpleDoc()

        print("Warning: spaCy is unavailable, falling back to empty entity matches.")
        return _SimpleNLP()

    try:
        return spacy.load(model_name, disable=["tagger", "parser", "lemmatizer"])
    except OSError:
        print(f"Warning: spaCy model '{model_name}' is unavailable, falling back to blank English.")
        return spacy.blank("en")


def extract_time_tokens(text: str):
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", norm_text(text).lower())
    return {tok for tok in toks if tok in TIME_MARKERS}


def extract_quantity_tokens(text: str):
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", norm_text(text).lower())
    return {tok for tok in toks if tok in QUANTITY_MARKERS}


def has_negation(text: str) -> bool:
    lower = norm_text(text).lower()
    if "n't" in lower:
        return True
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", lower)
    return any(tok in NEGATION_MARKERS for tok in toks)


def flatten_constraint_value(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_constraint_value(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_constraint_value(item))
        return out
    return [str(value)]


def render_constraint_text(constraint) -> str:
    if not isinstance(constraint, dict):
        return ""
    parts = []
    for key in ("negation", "time", "quantity"):
        values = flatten_constraint_value(constraint.get(key))
        if values:
            parts.append(f"{key}: " + " ; ".join(values))
    return " | ".join(parts)


def infer_negation_mode(fact) -> str:
    constraint = fact.get("constraint") or {}
    neg = constraint.get("negation")
    if isinstance(neg, bool):
        return "require" if neg else "forbid"
    if isinstance(neg, str):
        lower = neg.strip().lower()
        if not lower:
            return "neutral"
        if any(token in lower for token in ("affirm", "positive", "no negation", "without negation")):
            return "forbid"
        if any(token in lower for token in ("neg", "not", "without", "never", "false")):
            return "require"
    if has_negation(fact.get("text", "")):
        return "require"
    return "neutral"


def clamp_score(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, float(value)))


def lookup_sentence_score(score_map, sid):
    if not score_map:
        return 0.0
    if sid in score_map:
        return float(score_map.get(sid, 0.0))
    return float(score_map.get(f"S::{sid}", 0.0))


def topk_normalize(score_map, topk):
    if not score_map:
        return {}
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:topk]
    max_score = float(ranked[0][1]) if ranked else 0.0
    if max_score <= 0:
        return {}
    return {key: float(score) / max_score for key, score in ranked if float(score) > 0}


def merge_score_maps(*weighted_maps):
    merged = defaultdict(float)
    for weight, score_map in weighted_maps:
        if not score_map or weight == 0:
            continue
        for key, value in score_map.items():
            merged[key] += float(weight) * float(value)
    return dict(merged)


def normalize_sentence_node_scores(score_map, topk):
    sent_scores = {}
    for key, value in (score_map or {}).items():
        if isinstance(key, str) and key.startswith("S::"):
            sent_scores[key.split("::", 1)[1]] = float(value)
    return topk_normalize(sent_scores, topk)


def build_example_context(node, edge):
    sid2meta = {}
    sid_list = []
    sent_texts = []
    sid2keywords = {}
    sid2numbers = {}
    sid2time_tokens = {}
    sid2quantity_tokens = {}
    for sent in node["sent_nodes"]:
        sid = str(sent["sid"])
        text = sent.get("sentences") or sent.get("text") or ""
        if not text:
            continue
        sid_list.append(sid)
        sent_texts.append(text)
        sid2meta[sid] = {
            "text": text,
            "docid": sent.get("docid"),
            "doc_rank": int(sent.get("doc_rank", 10**9)),
            "sent_idx": int(sent.get("sent_idx", -1)),
        }
        sid2keywords[sid] = extract_keywords_simple(text)
        sid2numbers[sid] = extract_numbers(text)
        sid2time_tokens[sid] = extract_time_tokens(text)
        sid2quantity_tokens[sid] = extract_quantity_tokens(text)

    eid2name = {}
    eid2norm = {}
    for ent in node["entity_nodes"]:
        eid = str(ent["eid"])
        eid2name[eid] = ent.get("name", "")
        eid2norm[eid] = ent.get("norm", "")

    rid2name = {}
    rid2norm = {}
    for rel in node["relation_nodes"]:
        rid = str(rel["rid"])
        rid2name[rid] = rel.get("name", "")
        rid2norm[rid] = rel.get("norm", "")

    sid2eids = defaultdict(set)
    sid2rids = defaultdict(set)
    for item in edge["sn_edges"]:
        sid = str(item["sid"])
        eid = str(item["eid"])
        if sid in sid2meta:
            sid2eids[sid].add(eid)
    for item in edge["sr_edges"]:
        sid = str(item["sid"])
        rid = str(item["rid"])
        if sid in sid2meta:
            sid2rids[sid].add(rid)

    return {
        "sent_nodes": node["sent_nodes"],
        "entity_nodes": node["entity_nodes"],
        "relation_nodes": node["relation_nodes"],
        "sn_edges": edge["sn_edges"],
        "sr_edges": edge["sr_edges"],
        "nrn_edges": edge["nrn_edges"],
        "sid_list": sid_list,
        "sent_texts": sent_texts,
        "sid2meta": sid2meta,
        "sid2eids": sid2eids,
        "sid2rids": sid2rids,
        "sid2keywords": sid2keywords,
        "sid2numbers": sid2numbers,
        "sid2time_tokens": sid2time_tokens,
        "sid2quantity_tokens": sid2quantity_tokens,
        "eid2name": eid2name,
        "eid2norm": eid2norm,
        "rid2name": rid2name,
        "rid2norm": rid2norm,
    }

def encode_sentence_bank(biencoder, context):
    if not context["sid_list"]:
        return {"sid_list": [], "texts": [], "embeddings": np.zeros((0, 1), dtype=np.float32)}
    embeddings = biencoder.encode(
        context["sent_texts"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return {
        "sid_list": context["sid_list"],
        "texts": context["sent_texts"],
        "embeddings": embeddings,
    }


def semantic_entry_from_bank(biencoder, query_text, sentence_bank, topk):
    if not sentence_bank["sid_list"]:
        return {}
    query_vec = biencoder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]
    scores = np.dot(sentence_bank["embeddings"], query_vec)
    top_idx = np.argsort(scores)[::-1][:topk]
    return {sentence_bank["sid_list"][i]: float(scores[i]) for i in top_idx}


def select_sentence_candidates(context, sentence_scores, topk, component_maps=None):
    component_maps = component_maps or {}
    ranked = sorted(
        context["sid_list"],
        key=lambda sid: (
            lookup_sentence_score(sentence_scores, sid),
            lookup_sentence_score(component_maps.get("lexical_score"), sid),
            lookup_sentence_score(component_maps.get("dense_score"), sid),
        ),
        reverse=True,
    )[:topk]
    out = []
    for sid in ranked:
        meta = context["sid2meta"][sid]
        ppr_score = lookup_sentence_score(component_maps.get("ppr_score"), sid)
        out.append({
            "sid": sid,
            "recall_score": float(lookup_sentence_score(sentence_scores, sid)),
            "dense_score": float(lookup_sentence_score(component_maps.get("dense_score"), sid)),
            "lexical_score": float(lookup_sentence_score(component_maps.get("lexical_score"), sid)),
            "constraint_score": float(lookup_sentence_score(component_maps.get("constraint_score"), sid)),
            "dependency_score": float(lookup_sentence_score(component_maps.get("dependency_score"), sid)),
            "ppr_score": float(ppr_score),
            "graph_score": float(ppr_score),
            "text": meta["text"],
            "doc_rank": meta["doc_rank"],
            "docid": meta["docid"],
        })
    return out


def rerank_cross_encoder(crossencoder, query_text, candidates):
    if not candidates:
        return []
    pairs = [(query_text, cand["text"]) for cand in candidates]
    ce_scores = crossencoder.predict(pairs, show_progress_bar=False)
    out = []
    for cand, ce_score in zip(candidates, ce_scores):
        item = dict(cand)
        item["ce_score"] = float(ce_score)
        out.append(item)
    return out

def topological_sort_facts(atomic_facts):
    if not atomic_facts:
        return []
    id2fact = {fact["id"]: fact for fact in atomic_facts}
    children = defaultdict(list)
    indegree = {fact["id"]: 0 for fact in atomic_facts}
    position = {fact["id"]: idx for idx, fact in enumerate(atomic_facts)}

    for fact in atomic_facts:
        fid = fact["id"]
        for parent_id in fact.get("rely_on", []):
            if parent_id in id2fact:
                indegree[fid] += 1
                children[parent_id].append(fid)

    queue = deque(sorted([fid for fid, deg in indegree.items() if deg == 0], key=lambda x: position[x]))
    ordered = []
    seen = set()

    while queue:
        fid = queue.popleft()
        if fid in seen:
            continue
        seen.add(fid)
        ordered.append(id2fact[fid])
        for child in sorted(children[fid], key=lambda x: position[x]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(ordered) != len(atomic_facts):
        ordered.extend([fact for fact in atomic_facts if fact["id"] not in seen])
    return ordered


def top_score_items(score_map, topk):
    if not score_map:
        return []
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:topk]


def pick_salient_keywords(keywords, topk):
    if not keywords:
        return []
    return sorted(keywords, key=lambda x: (-len(x), x))[:topk]


def build_fact_graph_stats(fact_sequence):
    id2fact = {fact["id"]: fact for fact in fact_sequence}
    children = defaultdict(list)
    for fact in fact_sequence:
        for parent_id in fact.get("rely_on", []):
            if parent_id in id2fact:
                children[parent_id].append(fact["id"])

    depth_cache = {}

    def get_depth(fid, trail=None):
        if fid in depth_cache:
            return depth_cache[fid]
        parents = [pid for pid in id2fact[fid].get("rely_on", []) if pid in id2fact]
        if not parents:
            depth_cache[fid] = 1
            return 1

        trail = set() if trail is None else set(trail)
        trail.add(fid)
        best = 1
        for pid in parents:
            if pid in trail:
                continue
            best = max(best, 1 + get_depth(pid, trail))
        depth_cache[fid] = best
        return best

    depth_map = {fid: get_depth(fid) for fid in id2fact}
    return {
        "id2fact": id2fact,
        "children": children,
        "depth_map": depth_map,
        "max_depth": max(depth_map.values(), default=1),
        "critical_count": sum(1 for fact in fact_sequence if fact.get("critical")),
        "fact_count": len(fact_sequence),
    }


def infer_fact_role(fact, profile, fact_stats):
    if profile["numbers"] or profile["time_tokens"] or profile["quantity_tokens"]:
        return "anchor"
    fid = fact.get("id")
    child_count = len((fact_stats.get("children") or {}).get(fid, []))
    parent_count = sum(1 for pid in fact.get("rely_on", []) if pid in (fact_stats.get("id2fact") or {}))
    if fact.get("critical") or child_count == 0:
        return "verify"
    if parent_count > 0 or child_count > 0:
        return "bridge"
    return "verify"


def requires_direct_support(fact_role, fact):
    return fact_role in {"verify", "anchor"} or bool(fact.get("critical"))


def get_role_candidate_budget(fact_role, critical, args):
    if fact_role == "bridge":
        base = args.bridge_candidate_k
    elif fact_role == "anchor":
        base = args.anchor_candidate_k
    else:
        base = args.verify_candidate_k
    if critical:
        base = max(base, args.fact_candidate_k)
    return base


def get_direct_support_threshold(fact_role, args):
    if fact_role == "bridge":
        return args.bridge_direct_support_threshold
    if fact_role == "anchor":
        return args.anchor_direct_support_threshold
    if fact_role == "verify":
        return args.verify_direct_support_threshold
    return args.direct_support_threshold

def build_parent_support_summary(parent_results):
    summary = {
        "sids": set(),
        "eids": set(),
        "rids": set(),
        "docids": set(),
        "fact_ids": [],
    }
    for parent in parent_results:
        fact_id = parent.get("fact_id")
        if fact_id and fact_id not in summary["fact_ids"]:
            summary["fact_ids"].append(fact_id)
        support = parent.get("support_profile") or {}
        summary["sids"].update(support.get("sids", set()))
        summary["eids"].update(support.get("eids", set()))
        summary["rids"].update(support.get("rids", set()))
        summary["docids"].update(support.get("docids", set()))
    return summary


def build_fact_profile(fact, nlp, entity_nodes, relation_nodes):
    fact_text = fact.get("text", "")
    constraint = fact.get("constraint") or {}
    constraint_text = render_constraint_text(constraint)
    combined_text = fact_text if not constraint_text else f"{fact_text} {constraint_text}"
    entry_n = entity_entry_n(nlp, fact_text, entity_nodes)
    entry_r = relation_entry_r(fact_text, relation_nodes, topk=10)

    entity_lookup = {
        str(item.get("eid")): item.get("norm") or item.get("name", "")
        for item in entity_nodes
    }
    relation_lookup = {
        str(item.get("rid")): item.get("norm") or item.get("name", "")
        for item in relation_nodes
    }

    salient_keywords = set(pick_salient_keywords(extract_keywords_simple(fact_text), 8))
    entity_surface_keywords = set()
    for eid, _ in top_score_items(entry_n, 4):
        entity_surface_keywords.update(extract_keywords_simple(entity_lookup.get(str(eid), "")))

    relation_keywords = set()
    for rid, _ in top_score_items(entry_r, 4):
        relation_keywords.update(extract_keywords_simple(relation_lookup.get(str(rid), "")))
    if not relation_keywords:
        relation_keywords = set(pick_salient_keywords(salient_keywords - entity_surface_keywords, 6))

    return {
        "keywords": extract_keywords_simple(fact_text),
        "salient_keywords": salient_keywords,
        "entity_surface_keywords": entity_surface_keywords,
        "relation_keywords": relation_keywords,
        "numbers": extract_numbers(combined_text),
        "time_tokens": extract_time_tokens(combined_text),
        "quantity_tokens": extract_quantity_tokens(combined_text),
        "constraint_text": constraint_text,
        "negation_mode": infer_negation_mode(fact),
        "entry_n": entry_n,
        "entry_r": entry_r,
    }

def build_constraint_entry(profile, biencoder, sentence_bank, topk):
    if not (profile["numbers"] or profile["time_tokens"] or profile["quantity_tokens"]):
        return {}

    lexical_scores = defaultdict(float)
    for sid, text in zip(sentence_bank["sid_list"], sentence_bank["texts"]):
        lexical_scores[sid] += 1.4 * len(profile["numbers"] & extract_numbers(text))
        lexical_scores[sid] += 0.8 * len(profile["time_tokens"] & extract_time_tokens(text))
        lexical_scores[sid] += 0.8 * len(profile["quantity_tokens"] & extract_quantity_tokens(text))

    lexical_scores = topk_normalize(lexical_scores, topk)
    if not profile["constraint_text"]:
        return lexical_scores

    semantic_scores = semantic_entry_from_bank(
        biencoder,
        profile["constraint_text"],
        sentence_bank,
        topk=topk,
    )
    return merge_score_maps((1.0, lexical_scores), (0.6, semantic_scores))


def build_dependency_seed_maps(parent_results):
    dep_entry_n = defaultdict(float)
    dep_entry_r = defaultdict(float)
    for parent in parent_results:
        support = parent.get("support_profile") or {}
        for eid in support.get("eids", []):
            dep_entry_n[eid] += 1.0
        for rid in support.get("rids", []):
            dep_entry_r[rid] += 1.0
    return topk_normalize(dep_entry_n, 32), topk_normalize(dep_entry_r, 24)


def build_support_profile(candidates, context, max_candidates):
    profile = {"sids": set(), "eids": set(), "rids": set(), "docids": set()}
    ordered = sorted(candidates or [], key=candidate_rank_key, reverse=True)
    for cand in ordered[:max_candidates]:
        sid = cand["sid"]
        profile["sids"].add(sid)
        profile["eids"].update(context["sid2eids"].get(sid, set()))
        profile["rids"].update(context["sid2rids"].get(sid, set()))
        docid = context["sid2meta"].get(sid, {}).get("docid")
        if docid:
            profile["docids"].add(docid)
    return profile

def collect_support_from_sids(sids, context):
    summary = {"sids": set(), "eids": set(), "rids": set(), "docids": set()}
    for sid in sids:
        if sid not in context["sid2meta"]:
            continue
        summary["sids"].add(sid)
        summary["eids"].update(context["sid2eids"].get(sid, set()))
        summary["rids"].update(context["sid2rids"].get(sid, set()))
        docid = context["sid2meta"][sid].get("docid")
        if docid:
            summary["docids"].add(docid)
    return summary


def derive_binding_requirements(fact, profile, parent_results):
    parent_summary = build_parent_support_summary(parent_results)
    parent_keywords = set()
    parent_entry_eids = set()
    parent_entry_rids = set()
    for parent in parent_results:
        parent_profile = parent.get("fact_profile") or {}
        parent_keywords.update(parent_profile.get("keywords", set()))
        parent_entry_eids.update((parent_profile.get("entry_n") or {}).keys())
        parent_entry_rids.update((parent_profile.get("entry_r") or {}).keys())

    binding_eids = set(profile["entry_n"].keys()) & (parent_summary["eids"] | parent_entry_eids)
    binding_rids = set(profile["entry_r"].keys()) & (parent_summary["rids"] | parent_entry_rids)
    binding_keywords = set(profile["keywords"]) & parent_keywords
    binding_numbers = set(profile["numbers"])
    binding_time = set(profile["time_tokens"])
    binding_quantity = set(profile["quantity_tokens"])

    if not (binding_eids or binding_rids or binding_keywords or binding_numbers or binding_time or binding_quantity):
        binding_eids = {key for key, _ in top_score_items(profile["entry_n"], 2)}
        binding_rids = {key for key, _ in top_score_items(profile["entry_r"], 1)}
        binding_keywords = set(pick_salient_keywords(profile["keywords"], 2))

    return {
        "active": bool(fact.get("rely_on")),
        "eids": binding_eids,
        "rids": binding_rids,
        "keywords": binding_keywords,
        "numbers": binding_numbers,
        "time_tokens": binding_time,
        "quantity_tokens": binding_quantity,
        "min_keyword_hits": 1 if binding_keywords else 0,
    }


def score_keyword_overlap(profile, sid, context):
    target_keywords = profile.get("salient_keywords") or profile.get("keywords") or set()
    if not target_keywords:
        return 0.0
    sent_keywords = context["sid2keywords"].get(sid, set())
    return len(sent_keywords & target_keywords) / max(1, len(target_keywords))


def score_entity_pair_presence(profile, sid, context):
    target_eids = set(profile["entry_n"].keys())
    entity_terms = profile.get("entity_surface_keywords") or set()
    sent_eids = context["sid2eids"].get(sid, set())
    sent_keywords = context["sid2keywords"].get(sid, set())
    matched = len(sent_eids & target_eids)
    entity_hit = matched / max(1, len(target_eids)) if target_eids else 0.0
    term_hit = len(sent_keywords & entity_terms) / max(1, len(entity_terms)) if entity_terms else 0.0
    if not target_eids and not entity_terms:
        return 0.0
    if len(target_eids) >= 2:
        pair_bonus = 1.0 if matched >= 2 else (0.6 if matched == 1 and term_hit > 0 else 0.0)
        return clamp_score(0.60 * entity_hit + 0.40 * max(pair_bonus, term_hit))
    if target_eids:
        return clamp_score(max(entity_hit, 0.75 * term_hit))
    return clamp_score(term_hit)


def score_relation_expression(profile, sid, context):
    target_rids = set(profile["entry_r"].keys())
    relation_keywords = profile.get("relation_keywords") or set()
    sent_rids = context["sid2rids"].get(sid, set())
    sent_keywords = context["sid2keywords"].get(sid, set())

    score = 0.0
    weight = 0.0
    if target_rids:
        score += 0.60 * (len(sent_rids & target_rids) / max(1, len(target_rids)))
        weight += 0.60
    if relation_keywords:
        score += 0.40 * (len(sent_keywords & relation_keywords) / max(1, len(relation_keywords)))
        weight += 0.40
    if weight <= 0:
        fallback_keywords = set(profile.get("salient_keywords") or set()) - set(profile.get("entity_surface_keywords") or set())
        if not fallback_keywords:
            return 0.0
        return len(sent_keywords & fallback_keywords) / max(1, len(fallback_keywords))
    return clamp_score(score / weight)


def score_context_independence(profile, sid, context):
    text = context["sid2meta"].get(sid, {}).get("text", "")
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", norm_text(text).lower())
    score = 1.0
    entity_pair = score_entity_pair_presence(profile, sid, context)
    keyword_overlap = score_keyword_overlap(profile, sid, context)
    if toks and toks[0] in CONTEXT_DEPENDENT_STARTS:
        score -= 0.45 if entity_pair < 0.35 else 0.15
    if entity_pair < 0.35 and keyword_overlap < 0.25:
        score -= 0.20
    if len(toks) < 6:
        score -= 0.10
    return clamp_score(score)


def score_background_penalty(profile, sid, context, entity_pair_score, relation_score, keyword_score, constraint_consistency):
    text = norm_text(context["sid2meta"].get(sid, {}).get("text", "")).lower()
    generic_hit = 1.0 if any(text.startswith(pattern) for pattern in GENERIC_BACKGROUND_PATTERNS) else 0.0
    low_specificity = 1.0 if entity_pair_score < 0.35 and relation_score < 0.30 and keyword_score < 0.30 else 0.0
    anchor_mismatch = 1.0 if (profile["numbers"] or profile["time_tokens"] or profile["quantity_tokens"]) and constraint_consistency < 0 else 0.0
    return clamp_score(0.45 * generic_hit + 0.40 * low_specificity + 0.15 * anchor_mismatch)


def score_target_match(profile, candidate, context):
    sid = candidate["sid"]
    sent_keywords = context["sid2keywords"].get(sid, set())
    target_eids = set(profile["entry_n"].keys())
    target_rids = set(profile["entry_r"].keys())
    target_keywords = profile["keywords"]
    sent_eids = context["sid2eids"].get(sid, set())
    sent_rids = context["sid2rids"].get(sid, set())

    score = 0.0
    weight = 0.0
    if target_eids:
        score += 0.45 * (len(sent_eids & target_eids) / max(1, len(target_eids)))
        weight += 0.45
    if target_rids:
        score += 0.20 * (len(sent_rids & target_rids) / max(1, len(target_rids)))
        weight += 0.20
    if target_keywords:
        score += 0.35 * (len(sent_keywords & target_keywords) / max(1, len(target_keywords)))
        weight += 0.35
    if weight <= 0:
        return 0.0
    return score / weight

def score_time_quantity_consistency(profile, candidate, context):
    numbers = profile["numbers"]
    time_tokens = profile["time_tokens"]
    quantity_tokens = profile["quantity_tokens"]
    if not (numbers or time_tokens or quantity_tokens):
        return 0.0

    sid = candidate["sid"]
    sent_numbers = context["sid2numbers"].get(sid, set())
    sent_time = context["sid2time_tokens"].get(sid, set())
    sent_quantity = context["sid2quantity_tokens"].get(sid, set())

    score = 0.0
    weight = 0.0
    if numbers:
        overlap = len(numbers & sent_numbers) / max(1, len(numbers))
        mismatch = 1.0 if sent_numbers and not (numbers & sent_numbers) else 0.0
        score += 0.55 * overlap - 0.25 * mismatch
        weight += 0.55
    if time_tokens:
        score += 0.25 * (len(time_tokens & sent_time) / max(1, len(time_tokens)))
        weight += 0.25
    if quantity_tokens:
        score += 0.20 * (len(quantity_tokens & sent_quantity) / max(1, len(quantity_tokens)))
        weight += 0.20
    if weight <= 0:
        return 0.0
    return max(-1.0, min(1.0, score / weight))


def score_negation_compatibility(profile, candidate_text):
    mode = profile["negation_mode"]
    sent_neg = has_negation(candidate_text)
    if mode == "require":
        return 1.0 if sent_neg else -0.5
    if mode == "forbid":
        return 0.6 if not sent_neg else -0.6
    return 0.0 if not sent_neg else -0.2

def score_bridge_features(sid, target_support, context, semantic_sim_map, args):
    if not target_support["sids"] and not target_support["docids"] and not target_support["eids"] and not target_support["rids"]:
        return {
            "score": 0.0,
            "entity_overlap": 0.0,
            "relation_overlap": 0.0,
            "same_doc": 0.0,
            "semantic": 0.0,
            "cross_doc": 0.0,
            "satisfied": False,
        }

    sent_eids = context["sid2eids"].get(sid, set())
    sent_rids = context["sid2rids"].get(sid, set())
    docid = context["sid2meta"].get(sid, {}).get("docid")

    entity_overlap = 0.0 if not target_support["eids"] else len(sent_eids & target_support["eids"]) / max(1, len(target_support["eids"]))
    relation_overlap = 0.0 if not target_support["rids"] else len(sent_rids & target_support["rids"]) / max(1, len(target_support["rids"]))
    same_doc = 1.0 if docid and docid in target_support["docids"] else 0.0

    semantic = 0.0
    for target_sid in target_support["sids"]:
        semantic = max(semantic, get_sim(semantic_sim_map, sid, target_sid))

    has_entity_bridge = bool(sent_eids & target_support["eids"])
    has_relation_bridge = bool(sent_rids & target_support["rids"])
    cross_doc = 1.0 if docid and target_support["docids"] and docid not in target_support["docids"] and (has_entity_bridge or has_relation_bridge or semantic >= args.bridge_semantic_threshold) else 0.0
    score = 0.40 * entity_overlap + 0.15 * relation_overlap + 0.20 * same_doc + 0.15 * semantic + 0.10 * cross_doc
    satisfied = bool(same_doc or has_entity_bridge or has_relation_bridge or semantic >= args.bridge_semantic_threshold or cross_doc)
    return {
        "score": float(score),
        "entity_overlap": float(entity_overlap),
        "relation_overlap": float(relation_overlap),
        "same_doc": float(same_doc),
        "semantic": float(semantic),
        "cross_doc": float(cross_doc),
        "satisfied": satisfied,
    }


def score_upstream_bridge(candidate, parent_results, context, semantic_sim_map, args):
    parent_summary = build_parent_support_summary(parent_results)
    return score_bridge_features(candidate["sid"], parent_summary, context, semantic_sim_map, args)


def score_binding_coverage(binding_requirements, candidate, context):
    sid = candidate["sid"]
    sent_eids = context["sid2eids"].get(sid, set())
    sent_rids = context["sid2rids"].get(sid, set())
    sent_keywords = context["sid2keywords"].get(sid, set())
    sent_numbers = context["sid2numbers"].get(sid, set())
    sent_time = context["sid2time_tokens"].get(sid, set())
    sent_quantity = context["sid2quantity_tokens"].get(sid, set())

    score = 0.0
    weight = 0.0
    if binding_requirements["eids"]:
        score += 0.35 * (len(sent_eids & binding_requirements["eids"]) / max(1, len(binding_requirements["eids"])))
        weight += 0.35
    if binding_requirements["rids"]:
        score += 0.15 * (len(sent_rids & binding_requirements["rids"]) / max(1, len(binding_requirements["rids"])))
        weight += 0.15
    if binding_requirements["keywords"]:
        score += 0.20 * (len(sent_keywords & binding_requirements["keywords"]) / max(1, len(binding_requirements["keywords"])))
        weight += 0.20
    if binding_requirements["numbers"]:
        score += 0.15 * (len(sent_numbers & binding_requirements["numbers"]) / max(1, len(binding_requirements["numbers"])))
        weight += 0.15
    if binding_requirements["time_tokens"]:
        score += 0.10 * (len(sent_time & binding_requirements["time_tokens"]) / max(1, len(binding_requirements["time_tokens"])))
        weight += 0.10
    if binding_requirements["quantity_tokens"]:
        score += 0.05 * (len(sent_quantity & binding_requirements["quantity_tokens"]) / max(1, len(binding_requirements["quantity_tokens"])))
        weight += 0.05

    if weight <= 0:
        return {
            "score": 1.0 if not binding_requirements["active"] else 0.0,
            "direct_hit": not binding_requirements["active"],
            "keyword_hits": 0,
        }

    keyword_hits = len(sent_keywords & binding_requirements["keywords"])
    direct_hit = bool(sent_eids & binding_requirements["eids"])
    direct_hit = direct_hit or bool(sent_rids & binding_requirements["rids"])
    direct_hit = direct_hit or bool(sent_numbers & binding_requirements["numbers"])
    direct_hit = direct_hit or bool(sent_time & binding_requirements["time_tokens"])
    direct_hit = direct_hit or bool(sent_quantity & binding_requirements["quantity_tokens"])
    direct_hit = direct_hit or keyword_hits >= binding_requirements["min_keyword_hits"]
    return {
        "score": float(score / weight),
        "direct_hit": direct_hit,
        "keyword_hits": int(keyword_hits),
    }


def build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="base", topk=None):
    scores = defaultdict(float)
    binding_requirements = derive_binding_requirements(fact, profile, parent_results)
    parent_summary = build_parent_support_summary(parent_results)
    budget = topk or max(args.fact_candidate_k, get_role_candidate_budget(fact_role, bool(fact.get("critical")), args) * 2)

    for sid in context["sid_list"]:
        candidate = {"sid": sid}
        target_match = score_target_match(profile, candidate, context)
        entity_pair = score_entity_pair_presence(profile, sid, context)
        relation_match = score_relation_expression(profile, sid, context)
        keyword_overlap = score_keyword_overlap(profile, sid, context)
        constraint_consistency = max(0.0, score_time_quantity_consistency(profile, candidate, context))
        negation_compatibility = max(0.0, score_negation_compatibility(profile, context["sid2meta"][sid]["text"]))
        bridge = score_bridge_features(sid, parent_summary, context, context["semantic_sim_map"], args)
        binding = score_binding_coverage(binding_requirements, candidate, context)
        same_parent_doc = 1.0 if context["sid2meta"].get(sid, {}).get("docid") in parent_summary["docids"] else 0.0

        if mode == "direct":
            score = 0.34 * entity_pair + 0.24 * relation_match + 0.18 * keyword_overlap + 0.14 * constraint_consistency + 0.10 * target_match
            if fact_role == "verify":
                score += 0.08 * entity_pair + 0.04 * keyword_overlap
            elif fact_role == "anchor":
                score += 0.12 * constraint_consistency
        elif mode == "bridge":
            score = 0.36 * bridge["score"] + 0.28 * binding["score"] + 0.16 * same_parent_doc + 0.12 * relation_match + 0.08 * keyword_overlap
            if fact_role == "bridge":
                score += 0.10 * bridge["entity_overlap"] + 0.06 * bridge["relation_overlap"]
        else:
            score = 0.24 * entity_pair + 0.18 * relation_match + 0.18 * keyword_overlap + 0.16 * target_match + 0.10 * constraint_consistency + 0.04 * negation_compatibility + 0.10 * binding["score"]
            if fact_role == "verify":
                score += 0.08 * entity_pair + 0.06 * keyword_overlap
            elif fact_role == "bridge":
                score += 0.10 * bridge["score"] + 0.06 * same_parent_doc
            elif fact_role == "anchor":
                score += 0.14 * constraint_consistency

        if score > 0:
            scores[sid] = float(score)

    return topk_normalize(scores, budget)


def build_direct_repair_query_text(fact, profile, context):
    parts = []
    entity_names = []
    for eid, _ in top_score_items(profile["entry_n"], 2):
        name = context["eid2name"].get(str(eid)) or context["eid2norm"].get(str(eid))
        if name:
            entity_names.append(name)
    if entity_names:
        parts.append(" ".join(entity_names))
    relation_terms = sorted(profile.get("relation_keywords") or set())
    if relation_terms:
        parts.append(" ".join(relation_terms[:6]))
    parts.append(fact.get("text", ""))
    if profile.get("constraint_text"):
        parts.append(profile["constraint_text"])

    ordered = []
    seen = set()
    for part in parts:
        norm = norm_text(part)
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return " ".join(ordered)


def build_bridge_repair_query_text(fact, profile, parent_results, context):
    parent_summary = build_parent_support_summary(parent_results)
    parts = [fact.get("text", "")]

    entity_names = []
    for eid in sorted(parent_summary["eids"], key=lambda x: str(x)):
        name = context["eid2name"].get(str(eid)) or context["eid2norm"].get(str(eid))
        if name:
            entity_names.append(name)
        if len(entity_names) >= 3:
            break
    if entity_names:
        parts.append(" ".join(entity_names))

    relation_names = []
    for rid in sorted(parent_summary["rids"], key=lambda x: str(x)):
        name = context["rid2name"].get(str(rid)) or context["rid2norm"].get(str(rid))
        if name:
            relation_names.append(name)
        if len(relation_names) >= 2:
            break
    if relation_names:
        parts.append(" ".join(relation_names))

    if profile.get("constraint_text"):
        parts.append(profile["constraint_text"])

    ordered = []
    seen = set()
    for part in parts:
        norm = norm_text(part)
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return " ".join(ordered)


def filter_anchor_candidates(candidates, profile, context, args):
    if not candidates or not (profile["numbers"] or profile["time_tokens"] or profile["quantity_tokens"]):
        return candidates

    annotated = []
    for cand in candidates:
        item = dict(cand)
        item["_constraint_consistency"] = score_time_quantity_consistency(profile, item, context)
        annotated.append(item)

    filtered = [item for item in annotated if item["_constraint_consistency"] >= args.anchor_prefilter_threshold]
    if not filtered:
        keep = min(len(annotated), max(6, len(annotated) // 2))
        filtered = sorted(annotated, key=lambda x: (x["_constraint_consistency"], x["recall_score"]), reverse=True)[:keep]
    else:
        filtered = sorted(filtered, key=lambda x: (x["_constraint_consistency"], x["recall_score"]), reverse=True)

    for item in filtered:
        item.pop("_constraint_consistency", None)
    return filtered

def get_role_ranking_weights(fact_role):
    if fact_role == "bridge":
        return {
            "ce": 0.26,
            "dense": 0.08,
            "lexical": 0.12,
            "entity_pair": 0.12,
            "relation": 0.14,
            "target": 0.08,
            "constraint": 0.04,
            "negation": 0.02,
            "context": 0.08,
            "background_penalty": 0.10,
            "bridge_bonus": 0.28,
        }
    if fact_role == "anchor":
        return {
            "ce": 0.24,
            "dense": 0.08,
            "lexical": 0.14,
            "entity_pair": 0.10,
            "relation": 0.10,
            "target": 0.08,
            "constraint": 0.18,
            "negation": 0.02,
            "context": 0.08,
            "background_penalty": 0.12,
            "bridge_bonus": 0.08,
        }
    return {
        "ce": 0.30,
        "dense": 0.10,
        "lexical": 0.16,
        "entity_pair": 0.14,
        "relation": 0.12,
        "target": 0.10,
        "constraint": 0.06,
        "negation": 0.03,
        "context": 0.09,
        "background_penalty": 0.12,
        "bridge_bonus": 0.10,
    }


def candidate_rank_key(candidate):
    return (
        1 if candidate.get("support_type") == "direct_support" else 0,
        float(candidate.get("direct_support_score", 0.0)),
        float(candidate.get("aggregate_score", 0.0)),
        float(candidate.get("fact_score", 0.0)),
        float(candidate.get("bridge_support_score", 0.0)),
        float(candidate.get("coverage_score", 0.0)),
    )


def enrich_fact_candidates(fact, profile, fact_role, reranked, parent_results, context, args, critical_bonus):
    if not reranked:
        return []

    binding_requirements = derive_binding_requirements(fact, profile, parent_results)
    parents_covered = all((parent.get("coverage_summary") or {}).get("covered", False) for parent in parent_results)

    ce_vals = np.array([cand.get("ce_score", 0.0) for cand in reranked], dtype=np.float32)
    dense_vals = np.array([cand.get("dense_score", 0.0) for cand in reranked], dtype=np.float32)
    lexical_vals = np.array([cand.get("lexical_score", 0.0) for cand in reranked], dtype=np.float32)
    dependency_vals = np.array([cand.get("dependency_score", 0.0) for cand in reranked], dtype=np.float32)
    ppr_vals = np.array([cand.get("ppr_score", 0.0) for cand in reranked], dtype=np.float32)

    def _norm(value, values):
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax - vmin < 1e-12:
            return 0.0
        return (float(value) - vmin) / (vmax - vmin)

    weights = get_role_ranking_weights(fact_role)
    direct_threshold = get_direct_support_threshold(fact_role, args)
    scored = []

    for cand in reranked:
        ce_norm = _norm(cand.get("ce_score", 0.0), ce_vals)
        dense_norm = _norm(cand.get("dense_score", 0.0), dense_vals)
        lexical_norm = _norm(cand.get("lexical_score", 0.0), lexical_vals)
        dependency_norm = _norm(cand.get("dependency_score", 0.0), dependency_vals)
        ppr_norm = _norm(cand.get("ppr_score", 0.0), ppr_vals)

        semantic_relevance = 0.72 * ce_norm + 0.18 * dense_norm + 0.10 * lexical_norm
        target_match = score_target_match(profile, cand, context)
        entity_pair = score_entity_pair_presence(profile, cand["sid"], context)
        relation_match = score_relation_expression(profile, cand["sid"], context)
        keyword_overlap = score_keyword_overlap(profile, cand["sid"], context)
        constraint_consistency = score_time_quantity_consistency(profile, cand, context)
        negation_compatibility = score_negation_compatibility(profile, cand["text"])
        context_independence = score_context_independence(profile, cand["sid"], context)
        background_penalty = score_background_penalty(profile, cand["sid"], context, entity_pair, relation_match, keyword_overlap, constraint_consistency)
        bridge = score_upstream_bridge(cand, parent_results, context, context["semantic_sim_map"], args)
        binding = score_binding_coverage(binding_requirements, cand, context)
        doc_rank_bonus = 1.0 / (1.0 + max(0, cand.get("doc_rank", 10**9)))

        direct_support_score = clamp_score(
            weights["ce"] * ce_norm
            + weights["dense"] * dense_norm
            + weights["lexical"] * lexical_norm
            + weights["entity_pair"] * entity_pair
            + weights["relation"] * relation_match
            + weights["target"] * target_match
            + weights["constraint"] * max(0.0, constraint_consistency)
            + weights["negation"] * max(0.0, negation_compatibility)
            + weights["context"] * context_independence
            - weights["background_penalty"] * background_penalty
        )
        bridge_support_score = clamp_score(
            0.48 * max(0.0, bridge["score"])
            + 0.27 * binding["score"]
            + 0.15 * dependency_norm
            + 0.10 * ppr_norm
        )

        direct_support_pass = direct_support_score >= direct_threshold and (
            relation_match >= args.min_direct_relation_score or entity_pair >= 0.60 or target_match >= 0.55
        )
        if fact_role == "anchor" and (profile["numbers"] or profile["time_tokens"] or profile["quantity_tokens"]):
            direct_support_pass = direct_support_pass and constraint_consistency >= args.anchor_prefilter_threshold

        bridge_support_pass = bridge_support_score >= args.bridge_threshold or bridge["satisfied"] or binding["score"] >= args.binding_threshold
        dependency_closure_ready = (
            not fact.get("rely_on")
            or binding["direct_hit"]
            or binding["score"] >= args.binding_threshold
            or bridge["satisfied"]
            or bridge["score"] >= args.bridge_threshold
        )
        support_type = "direct_support" if direct_support_score >= max(direct_threshold - 0.05, bridge_support_score) else "bridge_support"

        fact_score = clamp_score(
            0.58 * direct_support_score
            + 0.18 * max(0.0, constraint_consistency)
            + 0.12 * context_independence
            + 0.12 * max(0.0, negation_compatibility)
        )
        aggregate_score = (
            1.45 * direct_support_score
            + 0.60 * ce_norm
            + 0.25 * lexical_norm
            + 0.18 * dense_norm
            + (0.30 if fact_role == "anchor" else 0.18) * max(0.0, constraint_consistency)
            + 0.12 * max(0.0, negation_compatibility)
            + 0.08 * context_independence
            + weights["bridge_bonus"] * bridge_support_score
            + 0.05 * ppr_norm
            + args.doc_rank_weight * doc_rank_bonus
            + critical_bonus
            - 0.20 * background_penalty
        )
        coverage_score = 0.68 * direct_support_score + 0.20 * bridge_support_score + 0.12 * max(0.0, constraint_consistency)

        if requires_direct_support(fact_role, fact):
            coverage_gate_pass = direct_support_pass and parents_covered and dependency_closure_ready
        else:
            coverage_gate_pass = parents_covered and ((direct_support_pass and dependency_closure_ready) or bridge_support_pass)

        item = dict(cand)
        item.update({
            "fact_id": fact["id"],
            "fact_role": fact_role,
            "semantic_relevance": float(semantic_relevance),
            "entity_target_match": float(target_match),
            "entity_pair_score": float(entity_pair),
            "relation_match_score": float(relation_match),
            "keyword_overlap": float(keyword_overlap),
            "time_quantity_consistency": float(constraint_consistency),
            "negation_compatibility": float(negation_compatibility),
            "context_independence": float(context_independence),
            "background_penalty": float(background_penalty),
            "dependency_compatibility": float(bridge["score"]),
            "critical_coverage_bonus": float(critical_bonus),
            "doc_rank_bonus": float(doc_rank_bonus),
            "fact_score": float(fact_score),
            "binding_score": float(binding["score"]),
            "binding_satisfied": bool(binding["direct_hit"] or binding["score"] >= args.binding_threshold),
            "bridge_score": float(bridge["score"]),
            "bridge_satisfied": bool(bridge["satisfied"] or bridge["score"] >= args.bridge_threshold),
            "bridge_support_score": float(bridge_support_score),
            "bridge_support_pass": bool(bridge_support_pass),
            "direct_support_score": float(direct_support_score),
            "direct_support_pass": bool(direct_support_pass),
            "dependency_closure_ready": bool(dependency_closure_ready),
            "support_type": support_type,
            "cross_doc_bridge_score": float(bridge["cross_doc"]),
            "coverage_score": float(coverage_score),
            "aggregate_score": float(aggregate_score),
            "coverage_gate_pass": bool(coverage_gate_pass),
        })
        scored.append(item)

    scored.sort(key=candidate_rank_key, reverse=True)
    return scored


def build_fact_coverage_summary(fact, fact_role, parent_results, scored_candidates, args):
    parent_covered = all((parent.get("coverage_summary") or {}).get("covered", False) for parent in parent_results)
    direct_candidates = sorted([cand for cand in scored_candidates if cand.get("direct_support_pass")], key=candidate_rank_key, reverse=True)
    bridge_candidates = sorted([cand for cand in scored_candidates if cand.get("bridge_support_pass")], key=candidate_rank_key, reverse=True)
    requires_direct = requires_direct_support(fact_role, fact)
    dependency_closure = (not fact.get("rely_on")) or any(cand.get("dependency_closure_ready") for cand in direct_candidates) or any(cand.get("bridge_support_pass") for cand in bridge_candidates)
    coverage_candidates = direct_candidates if requires_direct else (direct_candidates or bridge_candidates)
    need = args.critical_min_per_fact if fact.get("critical") else args.default_min_per_fact
    support_seed_candidates = list(direct_candidates)
    seen_sids = {cand["sid"] for cand in support_seed_candidates}
    for cand in bridge_candidates:
        if cand["sid"] not in seen_sids:
            support_seed_candidates.append(cand)
            seen_sids.add(cand["sid"])

    best_candidate = scored_candidates[0] if scored_candidates else None
    best_direct = direct_candidates[0] if direct_candidates else None
    best_bridge = bridge_candidates[0] if bridge_candidates else None
    covered = bool(parent_covered and coverage_candidates and dependency_closure and len(coverage_candidates) >= need)
    return {
        "covered": bool(covered),
        "parent_covered": bool(parent_covered),
        "requires_direct_support": bool(requires_direct),
        "has_direct_support": bool(direct_candidates),
        "dependency_closure": bool(dependency_closure),
        "needs_direct_repair": not bool(direct_candidates),
        "needs_bridge_repair": bool(fact.get("rely_on")) and bool(parent_covered) and not bool(dependency_closure),
        "top_fact_score": float(best_candidate["fact_score"]) if best_candidate else 0.0,
        "top_direct_support_score": float(best_direct["direct_support_score"]) if best_direct else 0.0,
        "top_bridge_support_score": float(best_bridge["bridge_support_score"]) if best_bridge else 0.0,
        "num_coverage_candidates": len(coverage_candidates),
        "num_direct_candidates": len(direct_candidates),
        "num_bridge_candidates": len(bridge_candidates),
        "best_direct_sid": best_direct["sid"] if best_direct else None,
        "best_bridge_sid": best_bridge["sid"] if best_bridge else None,
        "direct_candidates": direct_candidates,
        "bridge_candidates": bridge_candidates,
        "support_seed_candidates": support_seed_candidates,
    }


def coverage_insufficient(fact, fact_role, parent_results, scored_candidates, args):
    summary = build_fact_coverage_summary(fact, fact_role, parent_results, scored_candidates, args)
    return not summary["covered"] or summary["needs_direct_repair"] or summary["needs_bridge_repair"]

def build_chain_bridge_sentence_map(binding_requirements, parent_results, context, args):
    scores = defaultdict(float)
    parent_summary = build_parent_support_summary(parent_results)
    for sid in context["sid_list"]:
        binding = score_binding_coverage(binding_requirements, {"sid": sid}, context)
        bridge = score_bridge_features(sid, parent_summary, context, context["semantic_sim_map"], args)
        score = 0.45 * binding["score"] + 0.45 * bridge["score"] + 0.10 * bridge["cross_doc"]
        if score > 0:
            scores[sid] = score
    return topk_normalize(scores, args.chain_seed_k)


def build_chain_completion_entries(
    fact,
    profile,
    base_entry_s,
    base_entry_n,
    base_entry_r,
    scored_candidates,
    claim_entry_s,
    claim_entry_n,
    parent_results,
    context,
    args,
):
    expanded_s = defaultdict(float, base_entry_s)
    expanded_n = defaultdict(float, base_entry_n)
    expanded_r = defaultdict(float, base_entry_r)
    binding_requirements = derive_binding_requirements(fact, profile, parent_results)
    parent_summary = build_parent_support_summary(parent_results)

    for sid in parent_summary["sids"]:
        expanded_s[sid] += args.chain_parent_sentence_weight
    for eid in parent_summary["eids"]:
        expanded_n[eid] += args.chain_parent_neighbor_weight
    for rid in parent_summary["rids"]:
        expanded_r[rid] += args.chain_parent_neighbor_weight

    for sid, score in build_chain_bridge_sentence_map(binding_requirements, parent_results, context, args).items():
        expanded_s[sid] += args.chain_binding_sentence_weight * float(score)

    for eid in binding_requirements["eids"]:
        expanded_n[eid] += args.chain_binding_anchor_weight
    for rid in binding_requirements["rids"]:
        expanded_r[rid] += args.chain_binding_anchor_weight

    for cand in scored_candidates[:args.chain_seed_k]:
        if cand["bridge_score"] >= args.bridge_threshold or cand["binding_score"] >= args.binding_threshold or cand["fact_score"] >= args.fact_score_threshold:
            seed_weight = 0.5 * max(cand["fact_score"], cand["bridge_score"], cand["binding_score"])
            expanded_s[cand["sid"]] += seed_weight
            for eid in context["sid2eids"].get(cand["sid"], set()):
                expanded_n[eid] += 0.25 * seed_weight
            for rid in context["sid2rids"].get(cand["sid"], set()):
                expanded_r[rid] += 0.25 * seed_weight

    if fact.get("critical"):
        for cand in scored_candidates[:args.chain_seed_k]:
            expanded_s[cand["sid"]] += args.chain_critical_seed_weight * cand["fact_score"]
        for eid, score in top_score_items(profile["entry_n"], 4):
            expanded_n[eid] += args.chain_critical_seed_weight * float(score)
        for rid, score in top_score_items(profile["entry_r"], 4):
            expanded_r[rid] += args.chain_critical_seed_weight * float(score)
        if not parent_results:
            for sid, score in sorted(claim_entry_s.items(), key=lambda x: x[1], reverse=True)[:args.chain_seed_k]:
                expanded_s[sid] += args.chain_claim_weight * float(score)
            for eid, score in top_score_items(claim_entry_n, 4):
                expanded_n[eid] += args.chain_claim_weight * float(score)

    return dict(expanded_s), dict(expanded_n), dict(expanded_r)


def build_targeted_direct_repair_candidates(
    fact,
    profile,
    fact_role,
    context,
    sentence_bank,
    biencoder,
    parent_results,
    args,
):
    budget = args.critical_direct_repair_candidate_k if fact.get("critical") else args.direct_repair_candidate_k
    topk = max(budget * 2, args.fact_k)
    direct_lexical = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="direct", topk=topk)
    query_text = build_direct_repair_query_text(fact, profile, context)
    direct_dense = topk_normalize(semantic_entry_from_bank(biencoder, query_text, sentence_bank, topk=topk), topk)
    constraint_entry = build_constraint_entry(profile, biencoder, sentence_bank, topk=topk)
    combined = merge_score_maps(
        (args.direct_repair_lexical_weight, direct_lexical),
        (args.direct_repair_dense_weight, direct_dense),
        (0.40, constraint_entry),
    )
    candidates = select_sentence_candidates(
        context,
        combined,
        topk=budget,
        component_maps={
            "dense_score": direct_dense,
            "lexical_score": direct_lexical,
            "constraint_score": constraint_entry,
            "dependency_score": {},
            "ppr_score": {},
        },
    )
    if fact_role == "anchor":
        candidates = filter_anchor_candidates(candidates, profile, context, args)
    return candidates


def build_targeted_bridge_repair_candidates(
    fact,
    profile,
    fact_role,
    claim_entry_s,
    claim_entry_n,
    context,
    sentence_bank,
    graph,
    scored_candidates,
    parent_results,
    biencoder,
    args,
):
    if not parent_results:
        return []
    budget = args.bridge_repair_candidate_k
    topk = max(budget * 2, args.chain_seed_k)
    bridge_lexical = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="bridge", topk=topk)
    query_text = build_bridge_repair_query_text(fact, profile, parent_results, context)
    bridge_dense = topk_normalize(semantic_entry_from_bank(biencoder, query_text, sentence_bank, topk=topk), topk)

    dep_entry_n, dep_entry_r = build_dependency_seed_maps(parent_results)
    exp_s, exp_n, exp_r = build_chain_completion_entries(
        fact,
        profile,
        bridge_lexical,
        merge_score_maps((1.0, profile["entry_n"]), (args.dependency_seed_weight, dep_entry_n)),
        merge_score_maps((1.0, profile["entry_r"]), (args.dependency_seed_weight, dep_entry_r)),
        scored_candidates,
        claim_entry_s,
        claim_entry_n,
        parent_results,
        context,
        args,
    )
    exp_personalization = make_personalization(exp_s, exp_n, exp_r, w_s=args.w_entry_s, w_n=args.w_entry_n, w_r=args.w_entry_r)
    ppr_scores = normalize_sentence_node_scores(ppr(graph, exp_personalization, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter), topk)
    combined = merge_score_maps(
        (args.bridge_repair_lexical_weight, bridge_lexical),
        (args.bridge_repair_dense_weight, bridge_dense),
        (args.ppr_expand_weight, ppr_scores),
    )
    candidates = select_sentence_candidates(
        context,
        combined,
        topk=budget,
        component_maps={
            "dense_score": bridge_dense,
            "lexical_score": bridge_lexical,
            "constraint_score": {},
            "dependency_score": bridge_lexical,
            "ppr_score": ppr_scores,
        },
    )
    if fact_role == "anchor":
        candidates = filter_anchor_candidates(candidates, profile, context, args)
    return candidates


def merge_recall_candidates(*candidate_lists):
    merged = {}
    for candidates in candidate_lists:
        for cand in candidates:
            sid = cand["sid"]
            prev = merged.get(sid)
            if prev is None:
                merged[sid] = dict(cand)
                continue
            better = cand if (
                cand.get("recall_score", 0.0),
                cand.get("lexical_score", 0.0),
                cand.get("dense_score", 0.0),
            ) > (
                prev.get("recall_score", 0.0),
                prev.get("lexical_score", 0.0),
                prev.get("dense_score", 0.0),
            ) else prev
            item = dict(better)
            for field in ("recall_score", "dense_score", "lexical_score", "constraint_score", "dependency_score", "ppr_score", "graph_score"):
                item[field] = max(float(prev.get(field, 0.0)), float(cand.get(field, 0.0)))
            merged[sid] = item
    return sorted(merged.values(), key=lambda x: (x.get("recall_score", 0.0), x.get("lexical_score", 0.0), x.get("dense_score", 0.0)), reverse=True)


def merge_candidate_lists(*candidate_lists):
    merged = {}
    for candidates in candidate_lists:
        for cand in candidates:
            sid = cand["sid"]
            prev = merged.get(sid)
            if prev is None or candidate_rank_key(cand) > candidate_rank_key(prev):
                merged[sid] = dict(cand)
            elif prev is not None:
                for field in ("aggregate_score", "fact_score", "coverage_score", "direct_support_score", "bridge_support_score", "semantic_relevance"):
                    prev[field] = max(float(prev.get(field, 0.0)), float(cand.get(field, 0.0)))
                merged[sid] = prev
    return sorted(merged.values(), key=candidate_rank_key, reverse=True)

def retrieve_one_fact(
    fact,
    claim_entry_s,
    claim_entry_n,
    context,
    sentence_bank,
    graph,
    biencoder,
    crossencoder,
    nlp,
    parent_results,
    fact_stats,
    args,
):
    critical = bool(fact.get("critical"))
    critical_bonus = args.critical_bonus if critical else 0.0
    profile = build_fact_profile(fact, nlp, context["entity_nodes"], context["relation_nodes"])
    fact_role = infer_fact_role(fact, profile, fact_stats)
    fact["role"] = fact_role
    profile["fact_role"] = fact_role

    entry_k_s = args.critical_fact_k if critical else args.fact_k
    candidate_k = get_role_candidate_budget(fact_role, critical, args)
    base_entry_s = topk_normalize(semantic_entry_from_bank(biencoder, fact["text"], sentence_bank, topk=entry_k_s), entry_k_s)
    lexical_entry_s = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="base", topk=max(entry_k_s, candidate_k * 2))
    constraint_entry_s = build_constraint_entry(profile, biencoder, sentence_bank, topk=max(args.constraint_k, candidate_k))
    dependency_entry_s = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="bridge", topk=max(args.constraint_k, candidate_k)) if parent_results or fact_role == "bridge" else {}

    dep_entry_n, dep_entry_r = build_dependency_seed_maps(parent_results)
    entry_s = merge_score_maps(
        (args.initial_dense_weight, base_entry_s),
        (args.initial_lexical_weight, lexical_entry_s),
        (args.constraint_entry_weight, constraint_entry_s),
        (args.initial_dependency_sentence_weight, dependency_entry_s),
    )
    entry_n = merge_score_maps((1.0, profile["entry_n"]), (args.dependency_seed_weight, dep_entry_n))
    entry_r = merge_score_maps((1.0, profile["entry_r"]), (args.dependency_seed_weight, dep_entry_r))

    if not entry_s:
        entry_s = dict(sorted(claim_entry_s.items(), key=lambda x: x[1], reverse=True)[:entry_k_s])
    if not entry_n:
        entry_n = claim_entry_n

    local_candidates = select_sentence_candidates(
        context,
        entry_s,
        topk=candidate_k,
        component_maps={
            "dense_score": base_entry_s,
            "lexical_score": lexical_entry_s,
            "constraint_score": constraint_entry_s,
            "dependency_score": dependency_entry_s,
            "ppr_score": {},
        },
    )
    if fact_role == "anchor":
        local_candidates = filter_anchor_candidates(local_candidates, profile, context, args)

    reranked = rerank_cross_encoder(crossencoder, fact["text"], local_candidates)
    scored = enrich_fact_candidates(fact, profile, fact_role, reranked, parent_results, context, args, critical_bonus)
    coverage_summary = build_fact_coverage_summary(fact, fact_role, parent_results, scored, args)

    expanded = False
    repair_scored = []
    if coverage_summary["needs_direct_repair"]:
        expanded = True
        direct_repair_candidates = build_targeted_direct_repair_candidates(
            fact,
            profile,
            fact_role,
            context,
            sentence_bank,
            biencoder,
            parent_results,
            args,
        )
        if direct_repair_candidates:
            direct_reranked = rerank_cross_encoder(crossencoder, fact["text"], direct_repair_candidates)
            repair_scored.extend(enrich_fact_candidates(fact, profile, fact_role, direct_reranked, parent_results, context, args, critical_bonus))

    if coverage_summary["needs_bridge_repair"]:
        expanded = True
        bridge_repair_candidates = build_targeted_bridge_repair_candidates(
            fact,
            profile,
            fact_role,
            claim_entry_s,
            claim_entry_n,
            context,
            sentence_bank,
            graph,
            scored,
            parent_results,
            biencoder,
            args,
        )
        if bridge_repair_candidates:
            bridge_reranked = rerank_cross_encoder(crossencoder, fact["text"], bridge_repair_candidates)
            repair_scored.extend(enrich_fact_candidates(fact, profile, fact_role, bridge_reranked, parent_results, context, args, critical_bonus))

    if repair_scored:
        scored = merge_candidate_lists(scored, repair_scored)
        coverage_summary = build_fact_coverage_summary(fact, fact_role, parent_results, scored, args)

    support_seed_candidates = coverage_summary["support_seed_candidates"] if coverage_summary["support_seed_candidates"] else scored
    return {
        "fact_id": fact["id"],
        "text": fact["text"],
        "role": fact_role,
        "rely_on": fact.get("rely_on", []),
        "critical": critical,
        "constraint": fact.get("constraint", {}),
        "expanded": expanded,
        "entry_s": entry_s,
        "entry_n": entry_n,
        "entry_r": entry_r,
        "fact_profile": profile,
        "coverage_summary": coverage_summary,
        "support_profile": build_support_profile(support_seed_candidates, context, max_candidates=args.parent_support_k),
        "candidates": scored[:args.per_fact_output_k],
    }

def aggregate_entry_ids(fact_results, field_name, topk):
    merged = defaultdict(float)
    for fact_result in fact_results.values():
        for key, score in fact_result.get(field_name, {}).items():
            merged[key] += float(score)
    return [key for key, _ in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:topk]]


def is_redundant(candidate, selected, semantic_sim_map, threshold):
    for item in selected:
        if item["sid"] == candidate["sid"]:
            return True
        if item["text"] == candidate["text"]:
            return True
        if get_sim(semantic_sim_map, item["sid"], candidate["sid"]) >= threshold:
            return True
    return False


def build_global_candidate_view(fact_results, topk):
    sid2best = {}
    for fact_id, fact_result in fact_results.items():
        for cand in fact_result.get("candidates", []):
            sid = cand["sid"]
            prev = sid2best.get(sid)
            if prev is None:
                item = dict(cand)
                item["source_fact_ids"] = [fact_id]
                sid2best[sid] = item
                continue
            if fact_id not in prev["source_fact_ids"]:
                prev["source_fact_ids"].append(fact_id)
            if candidate_rank_key(cand) > candidate_rank_key(prev):
                updated = dict(cand)
                updated["source_fact_ids"] = prev["source_fact_ids"]
                sid2best[sid] = updated
    ranked = sorted(sid2best.values(), key=candidate_rank_key, reverse=True)
    return ranked[:topk]

def build_sentence_support_pool(fact_results, topk_per_fact):
    sentence_pool = {}
    for fact_id, fact_result in fact_results.items():
        for cand in fact_result.get("candidates", [])[:topk_per_fact]:
            sid = cand["sid"]
            item = sentence_pool.get(sid)
            if item is None:
                item = {
                    "sid": sid,
                    "text": cand["text"],
                    "docid": cand.get("docid"),
                    "doc_rank": int(cand.get("doc_rank", 10**9)),
                    "score": float(cand["aggregate_score"]),
                    "best_fact_score": float(cand["fact_score"]),
                    "source_fact_ids": [],
                    "fact_support": {},
                }
                sentence_pool[sid] = item
            if fact_id not in item["source_fact_ids"]:
                item["source_fact_ids"].append(fact_id)
            if cand["aggregate_score"] > item["score"]:
                item["text"] = cand["text"]
                item["docid"] = cand.get("docid")
                item["doc_rank"] = int(cand.get("doc_rank", 10**9))
                item["score"] = float(cand["aggregate_score"])
            item["best_fact_score"] = max(item["best_fact_score"], float(cand["fact_score"]))
            item["fact_support"][fact_id] = {
                "aggregate_score": float(cand["aggregate_score"]),
                "fact_score": float(cand["fact_score"]),
                "coverage_score": float(cand["coverage_score"]),
                "semantic_relevance": float(cand["semantic_relevance"]),
                "entity_target_match": float(cand["entity_target_match"]),
                "entity_pair_score": float(cand["entity_pair_score"]),
                "relation_match_score": float(cand["relation_match_score"]),
                "keyword_overlap": float(cand["keyword_overlap"]),
                "time_quantity_consistency": float(cand["time_quantity_consistency"]),
                "negation_compatibility": float(cand["negation_compatibility"]),
                "context_independence": float(cand["context_independence"]),
                "background_penalty": float(cand["background_penalty"]),
                "dependency_compatibility": float(cand["dependency_compatibility"]),
                "binding_score": float(cand["binding_score"]),
                "binding_satisfied": bool(cand["binding_satisfied"]),
                "bridge_score": float(cand["bridge_score"]),
                "bridge_satisfied": bool(cand["bridge_satisfied"]),
                "bridge_support_score": float(cand["bridge_support_score"]),
                "bridge_support_pass": bool(cand["bridge_support_pass"]),
                "direct_support_score": float(cand["direct_support_score"]),
                "direct_support_pass": bool(cand["direct_support_pass"]),
                "dependency_closure_ready": bool(cand["dependency_closure_ready"]),
                "support_type": cand["support_type"],
                "fact_role": cand["fact_role"],
                "cross_doc_bridge_score": float(cand["cross_doc_bridge_score"]),
                "critical_coverage_bonus": float(cand["critical_coverage_bonus"]),
                "doc_rank_bonus": float(cand["doc_rank_bonus"]),
            }
    return sentence_pool

def compute_dynamic_doc_budget(fact_sequence, fact_stats, sentence_pool, args):
    candidate_doc_count = len({item["docid"] for item in sentence_pool.values() if item.get("docid")})
    budget = args.base_max_docs_per_claim
    if fact_stats["fact_count"] >= 5:
        budget += 1
    if fact_stats["max_depth"] >= 4:
        budget += 1
    if fact_stats["critical_count"] >= 3:
        budget += 1
    if candidate_doc_count >= args.doc_budget_candidate_docs_threshold:
        budget += 1
    budget = max(1, min(args.max_docs_per_claim_cap, budget))
    return budget, {
        "base_max_docs_per_claim": int(args.base_max_docs_per_claim),
        "fact_count": int(fact_stats["fact_count"]),
        "dag_depth": int(fact_stats["max_depth"]),
        "critical_fact_count": int(fact_stats["critical_count"]),
        "candidate_doc_count": int(candidate_doc_count),
        "dynamic_max_docs_per_claim": int(budget),
    }


def score_bridge_against_selected_sids(sid, selected_parent_sids, context, semantic_sim_map, args):
    parent_support = collect_support_from_sids(selected_parent_sids, context)
    return score_bridge_features(sid, parent_support, context, semantic_sim_map, args)


def evaluate_selected_set(selected_sids, sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, args):
    selected_sids = list(selected_sids)
    selected_docs = {sentence_pool[sid]["docid"] for sid in selected_sids if sentence_pool[sid].get("docid")}
    fact_direct_candidates = defaultdict(list)
    fact_bridge_candidates = defaultdict(list)
    for sid in selected_sids:
        for fact_id, support in sentence_pool[sid]["fact_support"].items():
            if support.get("support_type") == "direct_support":
                fact_direct_candidates[fact_id].append((sid, support))
            else:
                fact_bridge_candidates[fact_id].append((sid, support))

    ordered_facts = sorted(
        fact_sequence,
        key=lambda fact: (fact_stats["depth_map"].get(fact["id"], 1), 0 if fact.get("critical") else 1),
    )
    covered_facts = set()
    fact_witnesses = {}
    facts_by_sid = defaultdict(list)
    coverage_value = 0.0
    dependency_covered = 0
    cross_doc_bridge_count = 0

    def zero_bridge_eval():
        return {
            "score": 0.0,
            "same_doc": 0.0,
            "entity_overlap": 0.0,
            "relation_overlap": 0.0,
            "semantic": 0.0,
            "cross_doc": 0.0,
            "satisfied": False,
        }

    def sort_direct(items):
        return sorted(items, key=lambda x: (float(x[1].get("direct_support_score", 0.0)), float(x[1].get("fact_score", 0.0)), float(x[1].get("aggregate_score", 0.0))), reverse=True)

    def sort_bridge(items):
        return sorted(items, key=lambda x: (float(x[1].get("bridge_support_score", 0.0)), float(x[1].get("aggregate_score", 0.0)), float(x[1].get("fact_score", 0.0))), reverse=True)

    for fact in ordered_facts:
        fid = fact["id"]
        fact_role = fact.get("role", "verify")
        parents = [pid for pid in fact.get("rely_on", []) if pid in fact_stats["id2fact"]]
        if parents and not all(pid in covered_facts for pid in parents):
            continue

        parent_witness_sids = [fact_witnesses[pid]["sid"] for pid in parents if pid in fact_witnesses]
        direct_candidates = sort_direct([item for item in fact_direct_candidates.get(fid, []) if item[1].get("direct_support_pass")])
        bridge_candidates = sort_bridge([item for item in fact_bridge_candidates.get(fid, []) if item[1].get("bridge_support_pass")])
        requires_direct = requires_direct_support(fact_role, fact)

        witness_sid = None
        witness_support = None
        witness_bridge_eval = zero_bridge_eval()
        for sid, support in direct_candidates:
            bridge_eval = zero_bridge_eval() if not parents else score_bridge_against_selected_sids(sid, parent_witness_sids, context, semantic_sim_map, args)
            if not parents or support.get("dependency_closure_ready") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                witness_sid = sid
                witness_support = support
                witness_bridge_eval = bridge_eval
                break

        if witness_sid is None and direct_candidates:
            witness_sid, witness_support = direct_candidates[0]
            witness_bridge_eval = zero_bridge_eval() if not parents else score_bridge_against_selected_sids(witness_sid, parent_witness_sids, context, semantic_sim_map, args)

        if witness_sid is None and not requires_direct and bridge_candidates:
            witness_sid, witness_support = bridge_candidates[0]
            witness_bridge_eval = zero_bridge_eval() if not parents else score_bridge_against_selected_sids(witness_sid, parent_witness_sids, context, semantic_sim_map, args)

        if witness_sid is None:
            continue

        helper_sid = None
        helper_support = None
        helper_bridge_eval = zero_bridge_eval()
        dependency_ready = not parents
        if parents:
            dependency_ready = bool(witness_support.get("dependency_closure_ready") or witness_bridge_eval["satisfied"] or witness_bridge_eval["score"] >= args.bridge_threshold)
            if not dependency_ready:
                helper_candidates = bridge_candidates + [item for item in direct_candidates if item[0] != witness_sid]
                seen_helpers = set()
                for sid, support in helper_candidates:
                    if sid in seen_helpers or sid == witness_sid:
                        continue
                    seen_helpers.add(sid)
                    bridge_eval = score_bridge_against_selected_sids(sid, parent_witness_sids, context, semantic_sim_map, args)
                    if support.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                        helper_sid = sid
                        helper_support = support
                        helper_bridge_eval = bridge_eval
                        dependency_ready = True
                        break

        if not dependency_ready:
            continue

        covered_facts.add(fid)
        depth = fact_stats["depth_map"].get(fid, 1)
        fact_value = 1.0 + args.assembly_depth_gain * max(0, depth - 1) + args.assembly_child_gain * len(fact_stats["children"].get(fid, []))
        if fact.get("critical"):
            fact_value += 1.0
        coverage_value += fact_value
        coverage_value += args.assembly_fact_score_weight * float(witness_support.get("fact_score", 0.0))
        coverage_value += args.assembly_direct_support_weight * float(witness_support.get("direct_support_score", 0.0))
        if helper_support is not None:
            coverage_value += args.assembly_bridge_helper_gain * float(helper_support.get("bridge_support_score", 0.0))

        if parents:
            dependency_covered += 1
            bridge_eval = helper_bridge_eval if helper_support is not None else witness_bridge_eval
            if bridge_eval.get("cross_doc", 0.0) > 0:
                cross_doc_bridge_count += 1

        fact_witnesses[fid] = {
            "sid": witness_sid,
            "helper_sid": helper_sid,
            "fact_score": float(witness_support.get("fact_score", 0.0)),
            "direct_support_score": float(witness_support.get("direct_support_score", 0.0)),
            "bridge_score": float(max(witness_support.get("bridge_support_score", 0.0), 0.0 if helper_support is None else helper_support.get("bridge_support_score", 0.0))),
            "cross_doc_bridge": float((helper_bridge_eval if helper_support is not None else witness_bridge_eval).get("cross_doc", 0.0)),
        }
        facts_by_sid[witness_sid].append(fid)
        if helper_sid is not None and helper_sid != witness_sid:
            facts_by_sid[helper_sid].append(fid)

    redundancy = 0.0
    for i, sid_i in enumerate(selected_sids):
        for sid_j in selected_sids[:i]:
            sim = get_sim(semantic_sim_map, sid_i, sid_j)
            if sim >= args.redundancy_threshold:
                redundancy += sim - args.redundancy_threshold
            doc_i = sentence_pool[sid_i].get("docid")
            doc_j = sentence_pool[sid_j].get("docid")
            if doc_i and doc_j and doc_i == doc_j:
                redundancy += args.assembly_same_doc_penalty

    critical_covered = sum(1 for fact in fact_sequence if fact.get("critical") and fact["id"] in covered_facts)
    utility = coverage_value
    utility += args.assembly_dependency_gain * dependency_covered
    utility += args.assembly_cross_doc_gain * cross_doc_bridge_count
    utility -= args.assembly_redundancy_weight * redundancy
    if len(selected_docs) > doc_budget:
        utility -= 10.0 * (len(selected_docs) - doc_budget)

    return {
        "utility": float(utility),
        "covered_facts": covered_facts,
        "fact_witnesses": fact_witnesses,
        "facts_by_sid": {sid: fact_ids for sid, fact_ids in facts_by_sid.items()},
        "critical_covered": int(critical_covered),
        "dependency_covered": int(dependency_covered),
        "cross_doc_bridge_count": int(cross_doc_bridge_count),
        "docids": selected_docs,
        "redundancy": float(redundancy),
    }

def aggregate_top_evidences(fact_sequence, fact_results, fact_stats, context, semantic_sim_map, args):
    sentence_pool = build_sentence_support_pool(fact_results, topk_per_fact=args.assembly_candidates_per_fact)
    if not sentence_pool:
        empty_summary = {
            "base_max_docs_per_claim": int(args.base_max_docs_per_claim),
            "dynamic_max_docs_per_claim": int(args.base_max_docs_per_claim),
            "selected_docids": [],
            "selected_sids": [],
            "covered_facts": [],
            "critical_covered": 0,
            "dependency_covered": 0,
            "cross_doc_bridge_count": 0,
        }
        return [], {}, empty_summary

    doc_budget, budget_summary = compute_dynamic_doc_budget(fact_sequence, fact_stats, sentence_pool, args)

    def sentence_rank(item):
        direct_max = max((support.get("direct_support_score", 0.0) for support in item[1]["fact_support"].values()), default=0.0)
        return (direct_max, item[1]["best_fact_score"], item[1]["score"])

    ranked_sids = [sid for sid, _ in sorted(sentence_pool.items(), key=sentence_rank, reverse=True)]

    selected_sids = []
    state = evaluate_selected_set(selected_sids, sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, args)
    while len(selected_sids) < args.max_evidence:
        best_sid = None
        best_state = None
        best_gain = args.assembly_stop_gain
        current_docids = set(state["docids"])

        for sid in ranked_sids:
            if sid in selected_sids:
                continue
            docid = sentence_pool[sid].get("docid")
            if docid and docid not in current_docids and len(current_docids) >= doc_budget:
                continue

            candidate_state = evaluate_selected_set(selected_sids + [sid], sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, args)
            gain = candidate_state["utility"] - state["utility"]
            if gain > best_gain:
                best_sid = sid
                best_state = candidate_state
                best_gain = gain

        if best_sid is None:
            break
        selected_sids.append(best_sid)
        state = best_state

    if not selected_sids and ranked_sids:
        selected_sids = [ranked_sids[0]]
        state = evaluate_selected_set(selected_sids, sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, args)

    fact_coverage = defaultdict(list)
    for sid in selected_sids:
        for fact_id in state["facts_by_sid"].get(sid, []):
            fact_coverage[fact_id].append(sid)

    selected = []
    for sid in selected_sids:
        item = sentence_pool[sid]
        supporting_facts = state["facts_by_sid"].get(sid) or item["source_fact_ids"]
        supporting_facts = sorted(supporting_facts, key=lambda fid: fact_stats["depth_map"].get(fid, 1))
        supports = [item["fact_support"][fid] for fid in supporting_facts if fid in item["fact_support"]]
        support_details = []
        for fid in supporting_facts:
            support = item["fact_support"].get(fid)
            if support is None:
                continue
            witness = state["fact_witnesses"].get(fid)
            support_details.append({
                "fact_id": fid,
                "aggregate_score": float(support["aggregate_score"]),
                "fact_score": float(support["fact_score"]),
                "support_type": support.get("support_type"),
                "direct_support_score": float(support.get("direct_support_score", 0.0)),
                "bridge_support_score": float(support.get("bridge_support_score", 0.0)),
                "covered": bool(witness and witness["sid"] == sid),
                "dependency_helper": bool(witness and witness.get("helper_sid") == sid),
            })

        if supports:
            score = max(s["aggregate_score"] for s in supports)
            fact_score = max(s["fact_score"] for s in supports)
            semantic_relevance = max(s["semantic_relevance"] for s in supports)
            entity_target_match = max(s["entity_target_match"] for s in supports)
            time_quantity_consistency = max(s["time_quantity_consistency"] for s in supports)
            negation_compatibility = max(s["negation_compatibility"] for s in supports)
            dependency_compatibility = max(s["dependency_compatibility"] for s in supports)
            binding_score = max(s["binding_score"] for s in supports)
            bridge_score = max(s["bridge_score"] for s in supports)
            direct_support_score = max(s.get("direct_support_score", 0.0) for s in supports)
            bridge_support_score = max(s.get("bridge_support_score", 0.0) for s in supports)
            cross_doc_bridge_score = max(s["cross_doc_bridge_score"] for s in supports)
            coverage_score = max(s["coverage_score"] for s in supports)
            critical_bonus = max(s["critical_coverage_bonus"] for s in supports)
            support_type = "direct_support" if any(s.get("support_type") == "direct_support" for s in supports) else "bridge_support"
        else:
            score = item["score"]
            fact_score = item["best_fact_score"]
            semantic_relevance = 0.0
            entity_target_match = 0.0
            time_quantity_consistency = 0.0
            negation_compatibility = 0.0
            dependency_compatibility = 0.0
            binding_score = 0.0
            bridge_score = 0.0
            direct_support_score = 0.0
            bridge_support_score = 0.0
            cross_doc_bridge_score = 0.0
            coverage_score = 0.0
            critical_bonus = 0.0
            support_type = "bridge_support"

        selected.append({
            "sid": sid,
            "text": item["text"],
            "docid": item.get("docid"),
            "doc_rank": int(item.get("doc_rank", 10**9)),
            "score": float(score),
            "fact_score": float(fact_score),
            "support_type": support_type,
            "direct_support_score": float(direct_support_score),
            "bridge_support_score": float(bridge_support_score),
            "semantic_relevance": float(semantic_relevance),
            "entity_target_match": float(entity_target_match),
            "time_quantity_consistency": float(time_quantity_consistency),
            "negation_compatibility": float(negation_compatibility),
            "dependency_compatibility": float(dependency_compatibility),
            "binding_score": float(binding_score),
            "bridge_score": float(bridge_score),
            "cross_doc_bridge_score": float(cross_doc_bridge_score),
            "critical_coverage_bonus": float(critical_bonus),
            "coverage_score": float(coverage_score),
            "supporting_facts": supporting_facts,
            "support_details": support_details,
            "selection_stage": "set_gain",
        })

    selected.sort(key=lambda x: (1 if x.get("support_type") == "direct_support" else 0, float(x.get("direct_support_score", 0.0)), float(x.get("fact_score", 0.0)), float(x.get("score", 0.0))), reverse=True)

    assembly_summary = dict(budget_summary)
    assembly_summary.update({
        "selected_docids": sorted(state["docids"]),
        "selected_sids": list(selected_sids),
        "covered_facts": sorted(state["covered_facts"], key=lambda fid: fact_stats["depth_map"].get(fid, 1)),
        "critical_covered": int(state["critical_covered"]),
        "dependency_covered": int(state["dependency_covered"]),
        "cross_doc_bridge_count": int(state["cross_doc_bridge_count"]),
        "redundancy": float(state["redundancy"]),
    })
    return selected, dict(fact_coverage), assembly_summary

def main(args):
    with open(args.decomposition_path, "r", encoding="utf-8") as f:
        decomposed_data = json.load(f)
    with open(args.nodes_path.replace("[SPLIT]", args.split), "r", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(args.edges_path.replace("[SPLIT]", args.split), "r", encoding="utf-8") as f:
        edges = json.load(f)
    with open(args.semantic_edges_path.replace("[SPLIT]", args.split), "r", encoding="utf-8") as f:
        semantic_edges = json.load(f)

    if args.limit > 0:
        decomposed_data = decomposed_data[:args.limit]

    id2node = {item["id"]: item for item in nodes}
    id2edge = {item["id"]: item for item in edges}
    id2semantic = {item["id"]: item.get("semantic_edges", []) for item in semantic_edges}

    device = resolve_device(args.device)
    biencoder = SentenceTransformer(args.embedding_model, device=device)
    crossencoder = CrossEncoder(args.cross_encoder_model, device=device)
    nlp = load_spacy_model(args.spacy_model)
    print(f"Loaded {len(decomposed_data)} decomposed claims.")
    print(f"Using device: {device}")

    results = []
    missing_ids = 0

    for sample in tqdm(decomposed_data):
        ex_id = sample["id"]
        node = id2node.get(ex_id)
        edge = id2edge.get(ex_id)
        if node is None or edge is None:
            missing_ids += 1
            continue

        context = build_example_context(node, edge)
        context["semantic_sim_map"] = build_semantic_sim_map(id2semantic.get(ex_id, []), min_sen_sim=args.min_sen_sim)
        sentence_bank = encode_sentence_bank(biencoder, context)

        graph = build_hetero_graph(
            context["sent_nodes"],
            context["entity_nodes"],
            context["relation_nodes"],
            context["sn_edges"],
            context["sr_edges"],
            context["nrn_edges"],
            semantic_edges=id2semantic.get(ex_id, []),
            w_sn=args.w_sn,
            w_sr=args.w_sr,
            w_nrn=args.w_nrn,
            w_ss=args.w_ss,
            min_sen_sim=args.min_sen_sim,
        )

        claim = sample["claim"]
        claim_entry_s = semantic_entry_from_bank(biencoder, claim, sentence_bank, topk=args.claim_entry_k)
        claim_entry_n = entity_entry_n(nlp, claim, context["entity_nodes"])

        fact_sequence = topological_sort_facts((sample.get("decomposition") or {}).get("atomic_facts", []))
        fact_stats = build_fact_graph_stats(fact_sequence)
        fact_results = {}
        for fact in fact_sequence:
            parent_results = [fact_results[parent_id] for parent_id in fact.get("rely_on", []) if parent_id in fact_results]
            fact_results[fact["id"]] = retrieve_one_fact(
                fact=fact,
                claim_entry_s=claim_entry_s,
                claim_entry_n=claim_entry_n,
                context=context,
                sentence_bank=sentence_bank,
                graph=graph,
                biencoder=biencoder,
                crossencoder=crossencoder,
                nlp=nlp,
                parent_results=parent_results,
                fact_stats=fact_stats,
                args=args,
            )

        top_evidences, fact_coverage, assembly_summary = aggregate_top_evidences(
            fact_sequence,
            fact_results,
            fact_stats,
            context,
            context["semantic_sim_map"],
            args,
        )

        fact_traces = []
        for fact in fact_sequence:
            fact_result = fact_results[fact["id"]]
            fact_traces.append({
                "fact_id": fact["id"],
                "text": fact["text"],
                "role": fact.get("role", fact_result.get("role")),
                "critical": bool(fact.get("critical")),
                "rely_on": fact.get("rely_on", []),
                "constraint": fact.get("constraint", {}),
                "expanded": bool(fact_result["expanded"]),
                "covered": bool((fact_result.get("coverage_summary") or {}).get("covered", False)),
                "has_direct_support": bool((fact_result.get("coverage_summary") or {}).get("has_direct_support", False)),
                "dependency_closure": bool((fact_result.get("coverage_summary") or {}).get("dependency_closure", False)),
                "top_fact_score": float((fact_result.get("coverage_summary") or {}).get("top_fact_score", 0.0)),
                "top_direct_support_score": float((fact_result.get("coverage_summary") or {}).get("top_direct_support_score", 0.0)),
                "selected_sids": fact_coverage.get(fact["id"], []),
                "top_candidates": fact_result["candidates"][:args.fact_trace_k],
            })

        results.append({
            "id": ex_id,
            "claim": claim,
            "label": sample.get("label"),
            "num_hops": sample.get("num_hops"),
            "entry_sids": aggregate_entry_ids(fact_results, "entry_s", topk=args.max_export_entry_s),
            "entry_nids": aggregate_entry_ids(fact_results, "entry_n", topk=args.max_export_entry_n),
            "entry_rids": aggregate_entry_ids(fact_results, "entry_r", topk=args.max_export_entry_r),
            "top_evidences": top_evidences,
            "reranked_candidates": build_global_candidate_view(fact_results, topk=args.max_export_candidates),
            "fact_traces": fact_traces,
            "assembly_summary": assembly_summary,
        })

    out_path = args.out_path.replace("[PLAN]", args.plan).replace("[SPLIT]", args.split)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {out_path}")
    if missing_ids:
        print(f"Skipped {missing_ids} examples missing graph data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition_path", type=str, default="./data/plan4.2/dev_2_decomposed_0_4000.json")
    parser.add_argument("--nodes_path", type=str, default="./data/plan1/bm25_nodes_[SPLIT].json")
    parser.add_argument("--edges_path", type=str, default="./data/plan1/bm25_edges_[SPLIT].json")
    parser.add_argument("--semantic_edges_path", type=str, default="./data/plan1/bm25_semantic_edges_[SPLIT].json")
    parser.add_argument("--out_path", type=str, default="./data/[PLAN]/nodefc_decomposition_aware_dev_0_4000.json")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--plan", type=str, default="plan4.3")

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross_encoder_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--claim_entry_k", type=int, default=20)
    parser.add_argument("--fact_k", type=int, default=18)
    parser.add_argument("--critical_fact_k", type=int, default=30)
    parser.add_argument("--constraint_k", type=int, default=16)
    parser.add_argument("--fact_candidate_k", type=int, default=28)
    parser.add_argument("--expanded_candidate_k", type=int, default=42)
    parser.add_argument("--verify_candidate_k", type=int, default=20)
    parser.add_argument("--bridge_candidate_k", type=int, default=32)
    parser.add_argument("--anchor_candidate_k", type=int, default=18)
    parser.add_argument("--direct_repair_candidate_k", type=int, default=24)
    parser.add_argument("--bridge_repair_candidate_k", type=int, default=18)
    parser.add_argument("--critical_direct_repair_candidate_k", type=int, default=36)
    parser.add_argument("--per_fact_output_k", type=int, default=12)
    parser.add_argument("--fact_trace_k", type=int, default=5)
    parser.add_argument("--max_evidence", type=int, default=8)

    parser.add_argument("--w_entry_s", type=float, default=0.55)
    parser.add_argument("--w_entry_n", type=float, default=0.25)
    parser.add_argument("--w_entry_r", type=float, default=0.20)
    parser.add_argument("--constraint_entry_weight", type=float, default=0.70)
    parser.add_argument("--dependency_seed_weight", type=float, default=0.85)
    parser.add_argument("--initial_dense_weight", type=float, default=0.95)
    parser.add_argument("--initial_lexical_weight", type=float, default=1.20)
    parser.add_argument("--initial_dependency_sentence_weight", type=float, default=0.75)
    parser.add_argument("--direct_repair_lexical_weight", type=float, default=1.20)
    parser.add_argument("--direct_repair_dense_weight", type=float, default=0.85)
    parser.add_argument("--bridge_repair_lexical_weight", type=float, default=1.00)
    parser.add_argument("--bridge_repair_dense_weight", type=float, default=0.45)
    parser.add_argument("--ppr_expand_weight", type=float, default=0.22)

    parser.add_argument("--w_sn", type=float, default=1.0)
    parser.add_argument("--w_sr", type=float, default=0.6)
    parser.add_argument("--w_nrn", type=float, default=1.0)
    parser.add_argument("--w_ss", type=float, default=0.6)
    parser.add_argument("--min_sen_sim", type=float, default=0.25)
    parser.add_argument("--ppr_alpha", type=float, default=0.68)
    parser.add_argument("--ppr_max_iter", type=int, default=25)

    parser.add_argument("--expand_seed_k", type=int, default=6)
    parser.add_argument("--expand_claim_k", type=int, default=8)
    parser.add_argument("--expand_sentence_weight", type=float, default=0.45)
    parser.add_argument("--expand_neighbor_weight", type=float, default=0.25)
    parser.add_argument("--expand_claim_weight", type=float, default=0.18)
    parser.add_argument("--expand_dependency_weight", type=float, default=0.20)

    parser.add_argument("--semantic_weight", type=float, default=1.20)
    parser.add_argument("--target_weight", type=float, default=1.15)
    parser.add_argument("--constraint_weight", type=float, default=0.75)
    parser.add_argument("--negation_weight", type=float, default=0.40)
    parser.add_argument("--dependency_weight", type=float, default=0.80)
    parser.add_argument("--binding_weight", type=float, default=0.70)
    parser.add_argument("--cross_doc_bridge_weight", type=float, default=0.45)
    parser.add_argument("--doc_rank_weight", type=float, default=0.10)
    parser.add_argument("--critical_bonus", type=float, default=0.20)

    parser.add_argument("--coverage_threshold", type=float, default=0.35)
    parser.add_argument("--fact_score_threshold", type=float, default=0.50)
    parser.add_argument("--binding_threshold", type=float, default=0.35)
    parser.add_argument("--bridge_threshold", type=float, default=0.28)
    parser.add_argument("--root_target_threshold", type=float, default=0.18)
    parser.add_argument("--bridge_semantic_threshold", type=float, default=0.42)
    parser.add_argument("--direct_support_threshold", type=float, default=0.58)
    parser.add_argument("--verify_direct_support_threshold", type=float, default=0.62)
    parser.add_argument("--anchor_direct_support_threshold", type=float, default=0.60)
    parser.add_argument("--bridge_direct_support_threshold", type=float, default=0.52)
    parser.add_argument("--min_direct_relation_score", type=float, default=0.16)
    parser.add_argument("--anchor_prefilter_threshold", type=float, default=0.00)
    parser.add_argument("--default_min_per_fact", type=int, default=1)
    parser.add_argument("--critical_min_per_fact", type=int, default=2)
    parser.add_argument("--parent_support_k", type=int, default=2)
    parser.add_argument("--redundancy_threshold", type=float, default=0.88)

    parser.add_argument("--chain_seed_k", type=int, default=8)
    parser.add_argument("--chain_parent_sentence_weight", type=float, default=0.55)
    parser.add_argument("--chain_parent_neighbor_weight", type=float, default=0.35)
    parser.add_argument("--chain_binding_sentence_weight", type=float, default=0.50)
    parser.add_argument("--chain_binding_anchor_weight", type=float, default=0.65)
    parser.add_argument("--chain_critical_seed_weight", type=float, default=0.45)
    parser.add_argument("--chain_claim_weight", type=float, default=0.18)

    parser.add_argument("--assembly_candidates_per_fact", type=int, default=6)
    parser.add_argument("--base_max_docs_per_claim", type=int, default=2)
    parser.add_argument("--max_docs_per_claim_cap", type=int, default=5)
    parser.add_argument("--doc_budget_candidate_docs_threshold", type=int, default=6)
    parser.add_argument("--assembly_depth_gain", type=float, default=0.30)
    parser.add_argument("--assembly_child_gain", type=float, default=0.12)
    parser.add_argument("--assembly_fact_score_weight", type=float, default=0.65)
    parser.add_argument("--assembly_direct_support_weight", type=float, default=0.85)
    parser.add_argument("--assembly_bridge_helper_gain", type=float, default=0.35)
    parser.add_argument("--assembly_dependency_gain", type=float, default=0.80)
    parser.add_argument("--assembly_cross_doc_gain", type=float, default=0.60)
    parser.add_argument("--assembly_redundancy_weight", type=float, default=1.00)
    parser.add_argument("--assembly_same_doc_penalty", type=float, default=0.10)
    parser.add_argument("--assembly_stop_gain", type=float, default=0.02)

    parser.add_argument("--max_export_entry_s", type=int, default=40)
    parser.add_argument("--max_export_entry_n", type=int, default=32)
    parser.add_argument("--max_export_entry_r", type=int, default=24)
    parser.add_argument("--max_export_candidates", type=int, default=64)
    args = parser.parse_args()
    main(args)
