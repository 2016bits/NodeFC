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


def select_sentence_candidates(context, sentence_scores, topk):
    ranked = sorted(
        context["sid_list"],
        key=lambda sid: sentence_scores.get(f"S::{sid}", 0.0),
        reverse=True,
    )[:topk]
    out = []
    for sid in ranked:
        meta = context["sid2meta"][sid]
        out.append((
            sid,
            float(sentence_scores.get(f"S::{sid}", 0.0)),
            meta["text"],
            meta["doc_rank"],
            meta["docid"],
        ))
    return out


def rerank_cross_encoder(crossencoder, query_text, candidates):
    if not candidates:
        return []
    pairs = [(query_text, cand[2]) for cand in candidates]
    ce_scores = crossencoder.predict(pairs, show_progress_bar=False)
    out = []
    for (sid, graph_score, text, doc_rank, docid), ce_score in zip(candidates, ce_scores):
        out.append({
            "sid": sid,
            "graph_score": float(graph_score),
            "ce_score": float(ce_score),
            "text": text,
            "doc_rank": int(doc_rank),
            "docid": docid,
        })
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
    return {
        "keywords": extract_keywords_simple(fact_text),
        "numbers": extract_numbers(combined_text),
        "time_tokens": extract_time_tokens(combined_text),
        "quantity_tokens": extract_quantity_tokens(combined_text),
        "constraint_text": constraint_text,
        "negation_mode": infer_negation_mode(fact),
        "entry_n": entity_entry_n(nlp, fact_text, entity_nodes),
        "entry_r": relation_entry_r(fact_text, relation_nodes, topk=10),
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
    for cand in candidates[:max_candidates]:
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


def enrich_fact_candidates(fact, profile, reranked, parent_results, context, args, critical_bonus):
    if not reranked:
        return []

    binding_requirements = derive_binding_requirements(fact, profile, parent_results)
    parents_covered = all((parent.get("coverage_summary") or {}).get("covered", False) for parent in parent_results)

    ce_vals = np.array([cand["ce_score"] for cand in reranked], dtype=np.float32)
    graph_vals = np.array([cand["graph_score"] for cand in reranked], dtype=np.float32)
    ce_min, ce_max = float(ce_vals.min()), float(ce_vals.max())
    graph_min, graph_max = float(graph_vals.min()), float(graph_vals.max())

    scored = []
    for cand in reranked:
        ce_norm = 0.0 if ce_max - ce_min < 1e-12 else (cand["ce_score"] - ce_min) / (ce_max - ce_min)
        graph_norm = 0.0 if graph_max - graph_min < 1e-12 else (cand["graph_score"] - graph_min) / (graph_max - graph_min)
        semantic_relevance = 0.7 * ce_norm + 0.3 * graph_norm
        target_match = score_target_match(profile, cand, context)
        constraint_consistency = score_time_quantity_consistency(profile, cand, context)
        negation_compatibility = score_negation_compatibility(profile, cand["text"])
        bridge = score_upstream_bridge(cand, parent_results, context, context["semantic_sim_map"], args)
        binding = score_binding_coverage(binding_requirements, cand, context)
        doc_rank_bonus = 1.0 / (1.0 + max(0, cand["doc_rank"]))

        fact_score = max(0.0, min(1.0, 0.34 * semantic_relevance + 0.22 * target_match + 0.10 * max(0.0, constraint_consistency) + 0.08 * max(0.0, negation_compatibility) + 0.14 * binding["score"] + 0.12 * max(0.0, bridge["score"])))
        aggregate_score = (
            args.semantic_weight * semantic_relevance
            + args.target_weight * target_match
            + args.constraint_weight * constraint_consistency
            + args.negation_weight * negation_compatibility
            + args.dependency_weight * max(0.0, bridge["score"])
            + args.binding_weight * binding["score"]
            + args.cross_doc_bridge_weight * bridge["cross_doc"]
            + args.doc_rank_weight * doc_rank_bonus
            + critical_bonus
        )
        coverage_score = 0.60 * fact_score + 0.20 * binding["score"] + 0.20 * max(0.0, bridge["score"])
        binding_satisfied = binding["direct_hit"] or binding["score"] >= args.binding_threshold
        bridge_satisfied = bridge["satisfied"] or bridge["score"] >= args.bridge_threshold
        if fact.get("rely_on"):
            coverage_gate_pass = fact_score >= args.fact_score_threshold and parents_covered and (binding_satisfied or bridge_satisfied)
        else:
            coverage_gate_pass = fact_score >= args.fact_score_threshold and (binding_satisfied or target_match >= args.root_target_threshold or not fact.get("critical"))

        item = dict(cand)
        item.update({
            "fact_id": fact["id"],
            "semantic_relevance": float(semantic_relevance),
            "entity_target_match": float(target_match),
            "time_quantity_consistency": float(constraint_consistency),
            "negation_compatibility": float(negation_compatibility),
            "dependency_compatibility": float(bridge["score"]),
            "critical_coverage_bonus": float(critical_bonus),
            "doc_rank_bonus": float(doc_rank_bonus),
            "fact_score": float(fact_score),
            "binding_score": float(binding["score"]),
            "binding_satisfied": bool(binding_satisfied),
            "bridge_score": float(bridge["score"]),
            "bridge_satisfied": bool(bridge_satisfied),
            "cross_doc_bridge_score": float(bridge["cross_doc"]),
            "coverage_score": float(coverage_score),
            "aggregate_score": float(aggregate_score),
            "coverage_gate_pass": bool(coverage_gate_pass),
        })
        scored.append(item)

    scored.sort(key=lambda x: (x["aggregate_score"], x["fact_score"], x["coverage_score"], x["semantic_relevance"]), reverse=True)
    return scored


def coverage_insufficient(fact, parent_results, scored_candidates, args):
    if not scored_candidates:
        return True
    if scored_candidates[0]["fact_score"] < args.fact_score_threshold:
        return True
    if fact.get("rely_on") and not all((parent.get("coverage_summary") or {}).get("covered", False) for parent in parent_results):
        return True

    need = args.critical_min_per_fact if fact.get("critical") else args.default_min_per_fact
    good = [cand for cand in scored_candidates if cand["coverage_gate_pass"]]
    if len(good) < need:
        return True
    if fact.get("critical") and not any(cand["binding_satisfied"] or cand["bridge_satisfied"] for cand in good):
        return True
    return False

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


def merge_candidate_lists(*candidate_lists):
    merged = {}
    for candidates in candidate_lists:
        for cand in candidates:
            sid = cand["sid"]
            prev = merged.get(sid)
            if prev is None or cand["aggregate_score"] > prev["aggregate_score"]:
                merged[sid] = cand
    return sorted(merged.values(), key=lambda x: (x["aggregate_score"], x["fact_score"], x["coverage_score"]), reverse=True)


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
    args,
):
    critical = bool(fact.get("critical"))
    entry_k_s = args.critical_fact_k if critical else args.fact_k
    critical_bonus = args.critical_bonus if critical else 0.0
    profile = build_fact_profile(fact, nlp, context["entity_nodes"], context["relation_nodes"])

    base_entry_s = semantic_entry_from_bank(biencoder, fact["text"], sentence_bank, topk=entry_k_s)
    constraint_entry_s = build_constraint_entry(profile, biencoder, sentence_bank, topk=args.constraint_k)
    dep_entry_n = {}
    dep_entry_r = {}
    parent_summary = build_parent_support_summary(parent_results)
    if parent_summary["eids"]:
        dep_entry_n = topk_normalize({eid: 1.0 for eid in parent_summary["eids"]}, 32)
    if parent_summary["rids"]:
        dep_entry_r = topk_normalize({rid: 1.0 for rid in parent_summary["rids"]}, 24)

    entry_s = merge_score_maps((1.0, base_entry_s), (args.constraint_entry_weight, constraint_entry_s))
    entry_n = merge_score_maps((1.0, profile["entry_n"]), (args.dependency_seed_weight, dep_entry_n))
    entry_r = merge_score_maps((1.0, profile["entry_r"]), (args.dependency_seed_weight, dep_entry_r))

    if not entry_s:
        entry_s = dict(sorted(claim_entry_s.items(), key=lambda x: x[1], reverse=True)[:entry_k_s])
    if not entry_n:
        entry_n = claim_entry_n

    personalization = make_personalization(
        entry_s,
        entry_n,
        entry_r,
        w_s=args.w_entry_s,
        w_n=args.w_entry_n,
        w_r=args.w_entry_r,
    )
    local_scores = ppr(graph, personalization, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter)
    local_candidates = select_sentence_candidates(context, local_scores, topk=args.fact_candidate_k)
    reranked = rerank_cross_encoder(crossencoder, fact["text"], local_candidates)
    scored = enrich_fact_candidates(fact, profile, reranked, parent_results, context, args, critical_bonus)

    expanded = False
    if coverage_insufficient(fact, parent_results, scored, args):
        expanded = True
        exp_s, exp_n, exp_r = build_chain_completion_entries(
            fact,
            profile,
            entry_s,
            entry_n,
            entry_r,
            scored,
            claim_entry_s,
            claim_entry_n,
            parent_results,
            context,
            args,
        )
        exp_personalization = make_personalization(
            exp_s,
            exp_n,
            exp_r,
            w_s=args.w_entry_s,
            w_n=args.w_entry_n,
            w_r=args.w_entry_r,
        )
        expanded_scores = ppr(graph, exp_personalization, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter)
        expanded_candidates = select_sentence_candidates(context, expanded_scores, topk=args.expanded_candidate_k)
        expanded_reranked = rerank_cross_encoder(crossencoder, fact["text"], expanded_candidates)
        expanded_scored = enrich_fact_candidates(fact, profile, expanded_reranked, parent_results, context, args, critical_bonus)
        scored = merge_candidate_lists(scored, expanded_scored)
        entry_s, entry_n, entry_r = exp_s, exp_n, exp_r

    coverage_candidates = [cand for cand in scored if cand["coverage_gate_pass"]]
    best_candidate = scored[0] if scored else None
    parent_covered = all((parent.get("coverage_summary") or {}).get("covered", False) for parent in parent_results)
    covered = bool(coverage_candidates) and best_candidate is not None and best_candidate["fact_score"] >= args.fact_score_threshold and (parent_covered or not fact.get("rely_on"))
    support_seed_candidates = coverage_candidates if coverage_candidates else scored

    return {
        "fact_id": fact["id"],
        "text": fact["text"],
        "rely_on": fact.get("rely_on", []),
        "critical": critical,
        "constraint": fact.get("constraint", {}),
        "expanded": expanded,
        "entry_s": entry_s,
        "entry_n": entry_n,
        "entry_r": entry_r,
        "fact_profile": profile,
        "coverage_summary": {
            "covered": bool(covered),
            "top_fact_score": float(best_candidate["fact_score"]) if best_candidate else 0.0,
            "num_coverage_candidates": len(coverage_candidates),
            "parent_covered": bool(parent_covered),
        },
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
            if cand["aggregate_score"] > prev["aggregate_score"]:
                updated = dict(cand)
                updated["source_fact_ids"] = prev["source_fact_ids"]
                sid2best[sid] = updated
    ranked = sorted(sid2best.values(), key=lambda x: (x["aggregate_score"], x["fact_score"], x["coverage_score"]), reverse=True)
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
                "time_quantity_consistency": float(cand["time_quantity_consistency"]),
                "negation_compatibility": float(cand["negation_compatibility"]),
                "dependency_compatibility": float(cand["dependency_compatibility"]),
                "binding_score": float(cand["binding_score"]),
                "binding_satisfied": bool(cand["binding_satisfied"]),
                "bridge_score": float(cand["bridge_score"]),
                "bridge_satisfied": bool(cand["bridge_satisfied"]),
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
    fact_candidates = defaultdict(list)
    for sid in selected_sids:
        for fact_id, support in sentence_pool[sid]["fact_support"].items():
            if support["fact_score"] >= args.fact_score_threshold:
                fact_candidates[fact_id].append((sid, support))

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

    for fact in ordered_facts:
        fid = fact["id"]
        parents = [pid for pid in fact.get("rely_on", []) if pid in fact_stats["id2fact"]]
        if parents and not all(pid in covered_facts for pid in parents):
            continue

        parent_witness_sids = [fact_witnesses[pid]["sid"] for pid in parents if pid in fact_witnesses]
        candidates = sorted(
            fact_candidates.get(fid, []),
            key=lambda x: (x[1]["fact_score"], x[1]["aggregate_score"], x[1]["coverage_score"]),
            reverse=True,
        )
        best = None
        for sid, support in candidates:
            if parents:
                bridge_eval = score_bridge_against_selected_sids(sid, parent_witness_sids, context, semantic_sim_map, args)
                coverage_ready = support["binding_satisfied"] or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold
            else:
                bridge_eval = {
                    "score": 0.0,
                    "same_doc": 0.0,
                    "entity_overlap": 0.0,
                    "relation_overlap": 0.0,
                    "semantic": 0.0,
                    "cross_doc": 0.0,
                    "satisfied": False,
                }
                coverage_ready = support["binding_satisfied"] or support["entity_target_match"] >= args.root_target_threshold or not fact.get("critical")
            if not coverage_ready:
                continue
            best = {"sid": sid, "support": support, "bridge": bridge_eval}
            break

        if best is None:
            continue

        covered_facts.add(fid)
        depth = fact_stats["depth_map"].get(fid, 1)
        fact_value = 1.0 + args.assembly_depth_gain * max(0, depth - 1) + args.assembly_child_gain * len(fact_stats["children"].get(fid, []))
        if fact.get("critical"):
            fact_value += 1.0
        coverage_value += fact_value + args.assembly_fact_score_weight * best["support"]["fact_score"]
        if parents:
            dependency_covered += 1
            if best["bridge"]["cross_doc"] > 0:
                cross_doc_bridge_count += 1
        fact_witnesses[fid] = {
            "sid": best["sid"],
            "fact_score": float(best["support"]["fact_score"]),
            "bridge_score": float(best["bridge"]["score"]),
            "cross_doc_bridge": float(best["bridge"]["cross_doc"]),
        }
        facts_by_sid[best["sid"]].append(fid)

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
    ranked_sids = [sid for sid, _ in sorted(sentence_pool.items(), key=lambda x: (x[1]["best_fact_score"], x[1]["score"]), reverse=True)]

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
                "covered": bool(witness and witness["sid"] == sid),
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
            cross_doc_bridge_score = max(s["cross_doc_bridge_score"] for s in supports)
            coverage_score = max(s["coverage_score"] for s in supports)
            critical_bonus = max(s["critical_coverage_bonus"] for s in supports)
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
            cross_doc_bridge_score = 0.0
            coverage_score = 0.0
            critical_bonus = 0.0

        selected.append({
            "sid": sid,
            "text": item["text"],
            "docid": item.get("docid"),
            "doc_rank": int(item.get("doc_rank", 10**9)),
            "score": float(score),
            "fact_score": float(fact_score),
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
                "critical": bool(fact.get("critical")),
                "rely_on": fact.get("rely_on", []),
                "constraint": fact.get("constraint", {}),
                "expanded": bool(fact_result["expanded"]),
                "covered": bool((fact_result.get("coverage_summary") or {}).get("covered", False)),
                "top_fact_score": float((fact_result.get("coverage_summary") or {}).get("top_fact_score", 0.0)),
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

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {args.out_path}")
    if missing_ids:
        print(f"Skipped {missing_ids} examples missing graph data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition_path", type=str, default="./data/plan4.2/dev_2_decomposed_0_4000.json")
    parser.add_argument("--nodes_path", type=str, default="./data/plan1/bm25_nodes_[SPLIT].json")
    parser.add_argument("--edges_path", type=str, default="./data/plan1/bm25_edges_[SPLIT].json")
    parser.add_argument("--semantic_edges_path", type=str, default="./data/plan1/bm25_semantic_edges_[SPLIT].json")
    parser.add_argument("--out_path", type=str, default="./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=0)

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
    parser.add_argument("--per_fact_output_k", type=int, default=12)
    parser.add_argument("--fact_trace_k", type=int, default=5)
    parser.add_argument("--max_evidence", type=int, default=8)

    parser.add_argument("--w_entry_s", type=float, default=0.55)
    parser.add_argument("--w_entry_n", type=float, default=0.25)
    parser.add_argument("--w_entry_r", type=float, default=0.20)
    parser.add_argument("--constraint_entry_weight", type=float, default=0.70)
    parser.add_argument("--dependency_seed_weight", type=float, default=0.85)

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
