import re
from collections import defaultdict, deque

import numpy as np
import torch

try:
    import spacy
except ModuleNotFoundError:
    spacy = None

from search_graph_hopaware import (
    entity_entry_n,
    extract_keywords_simple,
    extract_numbers,
    norm_text,
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


def candidate_rank_key(candidate):
    return (
        1 if candidate.get("support_type") == "direct_support" else 0,
        float(candidate.get("direct_support_score", 0.0)),
        float(candidate.get("aggregate_score", 0.0)),
        float(candidate.get("fact_score", 0.0)),
        float(candidate.get("bridge_support_score", 0.0)),
        float(candidate.get("coverage_score", 0.0)),
    )


def compute_fact_covered_hard(fact, fact_role, has_direct_support, dependency_closure_ready):
    if fact_role in {"verify", "anchor"}:
        return bool(has_direct_support and dependency_closure_ready)
    if requires_direct_support(fact_role, fact):
        return bool(has_direct_support and dependency_closure_ready)
    return bool(dependency_closure_ready)
