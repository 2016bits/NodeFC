import argparse
import json
import re
from collections import defaultdict

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher


ROLE_ORDER = {
    "claim": 0,
    "critical": 1,
    "leaf": 2,
    "bridge": 3,
}

FACT_ROLE_PRIORITY = {
    "critical": 0,
    "bridge": 1,
    "leaf": 2,
}


def normalize_ws(text):
    return " ".join(str(text if text is not None else "").split()).strip()


def flatten_constraint_value(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_ws(value)
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        text = normalize_ws(value)
        return [text] if text else []
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_constraint_value(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(flatten_constraint_value(item))
        return out
    return []


def render_constraint_text(constraint):
    if not isinstance(constraint, dict):
        return ""
    parts = []
    for key in ("time", "date", "year", "quantity", "number", "count", "negation"):
        values = flatten_constraint_value(constraint.get(key))
        if values:
            parts.append(" ".join(values))
    if not parts:
        parts = flatten_constraint_value(constraint)
    seen = set()
    ordered = []
    for part in parts:
        text = normalize_ws(part)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return " ".join(ordered)


def load_decomposition_map(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item.get("id")): item for item in data if isinstance(item, dict) and "id" in item}


def load_atomic_facts(data, decomposition_map):
    decomposition_item = decomposition_map.get(str(data.get("id"))) or {}
    atomic_facts = []

    if isinstance(data.get("decomposition"), dict):
        atomic_facts = (data.get("decomposition") or {}).get("atomic_facts") or []
    elif isinstance(data.get("atomic_facts"), list):
        atomic_facts = data.get("atomic_facts") or []
    elif isinstance(decomposition_item.get("decomposition"), dict):
        atomic_facts = (decomposition_item.get("decomposition") or {}).get("atomic_facts") or []
    elif isinstance(decomposition_item.get("atomic_facts"), list):
        atomic_facts = decomposition_item.get("atomic_facts") or []

    facts = []
    for idx, fact in enumerate(atomic_facts):
        if not isinstance(fact, dict):
            continue
        item = dict(fact)
        item["id"] = str(item.get("id") or f"atomic_{idx}")
        item["_source_order"] = idx
        item["_text"] = normalize_ws(item.get("text", ""))
        facts.append(item)
    return facts


def build_fact_graph_stats(fact_sequence):
    id2fact = {fact["id"]: fact for fact in fact_sequence}
    children = defaultdict(list)
    for fact in fact_sequence:
        for parent_id in fact.get("rely_on", []):
            parent_id = str(parent_id)
            if parent_id in id2fact:
                children[parent_id].append(fact["id"])

    depth_cache = {}

    def get_depth(fid, trail=None):
        if fid in depth_cache:
            return depth_cache[fid]
        parents = [str(pid) for pid in id2fact[fid].get("rely_on", []) if str(pid) in id2fact]
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
        "fact_count": len(fact_sequence),
        "critical_count": sum(1 for fact in fact_sequence if fact.get("critical")),
    }


def infer_fact_query_role(fact, fact_stats, args):
    fid = str(fact.get("id"))
    child_count = len((fact_stats.get("children") or {}).get(fid, []))
    parent_count = sum(1 for pid in fact.get("rely_on", []) if str(pid) in (fact_stats.get("id2fact") or {}))
    depth = int((fact_stats.get("depth_map") or {}).get(fid, 1))
    explicit_bridge = bool(fact.get("bridge") or fact.get("is_bridge") or fact.get("bridge_fact"))
    is_leaf_like = child_count == 0 or depth >= int(args.deep_fact_depth_threshold)

    if fact.get("critical"):
        return "critical"
    if explicit_bridge or parent_count > 0 or child_count > 0:
        return "bridge"
    if is_leaf_like:
        return "leaf"
    return "leaf"


def build_role_queries(data, decomposition_map, args):
    atomic_facts = load_atomic_facts(data, decomposition_map)
    fact_stats = build_fact_graph_stats(atomic_facts)
    grouped_queries = {}

    for fact in atomic_facts:
        fact_text = normalize_ws(fact.get("text"))
        constraint_text = render_constraint_text(fact.get("constraint") or {})
        query_text = normalize_ws(" ".join(part for part in (fact_text, constraint_text) if part))
        if not query_text:
            continue

        fid = str(fact["id"])
        role = infer_fact_query_role(fact, fact_stats, args)
        depth = int((fact_stats.get("depth_map") or {}).get(fid, 1))
        child_count = len((fact_stats.get("children") or {}).get(fid, []))
        parent_count = sum(1 for pid in fact.get("rely_on", []) if str(pid) in (fact_stats.get("id2fact") or {}))
        key = (role, query_text)

        item = grouped_queries.get(key)
        if item is None:
            item = {
                "query_id": f"{role}_{len(grouped_queries)}",
                "role": role,
                "query": query_text,
                "text": fact_text or query_text,
                "constraint_text": constraint_text,
                "fact_ids": [],
                "critical": False,
                "max_depth": depth,
                "max_parent_count": parent_count,
                "max_child_count": child_count,
                "source_order": int(fact.get("_source_order", 0)),
            }
            grouped_queries[key] = item

        item["fact_ids"].append(fid)
        item["critical"] = bool(item["critical"] or fact.get("critical"))
        item["max_depth"] = max(int(item["max_depth"]), depth)
        item["max_parent_count"] = max(int(item["max_parent_count"]), parent_count)
        item["max_child_count"] = max(int(item["max_child_count"]), child_count)
        item["source_order"] = min(int(item["source_order"]), int(fact.get("_source_order", 0)))

    queries = list(grouped_queries.values())
    queries.sort(
        key=lambda item: (
            FACT_ROLE_PRIORITY.get(item["role"], 99),
            -int(item["critical"]),
            -int(item["max_depth"]),
            -int(item["max_parent_count"]),
            0 if item["role"] == "leaf" and int(item["max_child_count"]) == 0 else 1,
            -len(item["query"]),
            int(item["source_order"]),
            item["query"],
        )
    )

    if args.max_atomic_fact_queries > 0:
        queries = queries[:args.max_atomic_fact_queries]

    role_queries = {"critical": [], "leaf": [], "bridge": []}
    for item in queries:
        role_queries.setdefault(item["role"], []).append(item)
    return role_queries, fact_stats


def run_query(searcher, query_text, topk):
    if topk <= 0 or not normalize_ws(query_text):
        return []
    hits = searcher.search(query_text, k=topk)
    results = []
    for rank, hit in enumerate(hits, start=1):
        try:
            doc = json.loads(hit.raw)
        except json.JSONDecodeError:
            doc = {}
        results.append({
            "docid": hit.docid,
            "raw_score": float(hit.score),
            "rank": rank,
            "text": doc.get("contents", ""),
        })
    return results


def resolve_claim_topk(args):
    return args.claim_topk if args.claim_topk > 0 else args.topk


def resolve_base_fact_topk(args, claim_topk):
    if args.atomic_topk > 0:
        return args.atomic_topk
    return max(4, max(1, claim_topk // 2))


def resolve_role_topk(args, claim_topk):
    base_fact_topk = resolve_base_fact_topk(args, claim_topk)
    critical_topk = args.critical_topk if args.critical_topk > 0 else base_fact_topk
    leaf_topk = args.leaf_topk if args.leaf_topk > 0 else base_fact_topk
    bridge_topk = args.bridge_topk if args.bridge_topk > 0 else base_fact_topk
    return {
        "critical": int(critical_topk),
        "leaf": int(leaf_topk),
        "bridge": int(bridge_topk),
    }


def resolve_final_topk(args, claim_topk, fact_query_count, role_topk_map):
    if args.final_topk > 0:
        return args.final_topk
    if args.union_topk > 0:
        return args.union_topk
    if fact_query_count <= 0:
        return claim_topk
    fact_topk_hint = max(role_topk_map.values(), default=0)
    extra = max(fact_topk_hint, min(claim_topk, fact_query_count * max(1, fact_topk_hint // 2)))
    return max(claim_topk, min(claim_topk * 2, claim_topk + extra))


def normalize_hit_score(raw_score, best_score):
    raw_score = float(raw_score)
    best_score = float(best_score)
    if best_score <= 0:
        return 0.0
    return max(0.0, raw_score / best_score)


def normalize_title_pattern(docid):
    text = normalize_ws(docid).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\b\d{3,4}\b", " # ", text)
    text = re.sub(r"[_/|]+", " ", text)
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return normalize_ws(text)


def infer_entity_cluster(docid):
    raw = normalize_ws(docid).lower()
    for sep in ("(", " - ", ":", ","):
        if sep in raw:
            raw = raw.split(sep, 1)[0]
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    raw = normalize_ws(raw)
    if not raw:
        return normalize_title_pattern(docid)
    toks = raw.split()
    return " ".join(toks[:4])


def compute_role_bonus(item, args):
    bonus = 0.0
    routes = item.get("retrieval_routes") or set()
    if "claim" in routes:
        bonus += float(args.claim_role_bonus)
    if "critical" in routes:
        bonus += float(args.critical_role_bonus)
    if "leaf" in routes:
        bonus += float(args.leaf_role_bonus)
    if "bridge" in routes:
        bonus += float(args.bridge_role_bonus)
    return bonus


def init_doc_record(hit):
    return {
        "docid": hit["docid"],
        "text": hit.get("text", ""),
        "score": 0.0,
        "claim_score": None,
        "claim_rank": None,
        "norm_bm25_claim": 0.0,
        "max_fact_score": 0.0,
        "atomic_fact_score": 0.0,
        "num_fact_hits": 0,
        "best_rank": 10**9,
        "retrieval_routes": set(),
        "matched_atomic_fact_ids": set(),
        "matched_atomic_fact_roles": set(),
        "matched_role_fact_ids": defaultdict(set),
        "fact_best_scores": {},
        "role_best_scores": defaultdict(float),
        "fact_query_hit_count": 0,
        "role_bonus": 0.0,
        "title_pattern": "",
        "entity_cluster": "",
    }


def get_doc_record(merged, hit):
    item = merged.get(hit["docid"])
    if item is None:
        item = init_doc_record(hit)
        merged[hit["docid"]] = item
    if not item["text"] and hit.get("text"):
        item["text"] = hit["text"]
    item["best_rank"] = min(int(item["best_rank"]), int(hit["rank"]))
    return item


def build_selection_state(selected):
    state = {
        "fact_counts": defaultdict(int),
        "title_pattern_counts": defaultdict(int),
        "entity_cluster_counts": defaultdict(int),
        "covered_facts": set(),
    }
    for item in selected:
        for fact_id in item.get("matched_atomic_fact_ids", set()):
            state["fact_counts"][fact_id] += 1
            state["covered_facts"].add(fact_id)
        title_pattern = item.get("title_pattern") or ""
        entity_cluster = item.get("entity_cluster") or ""
        if title_pattern:
            state["title_pattern_counts"][title_pattern] += 1
        if entity_cluster:
            state["entity_cluster_counts"][entity_cluster] += 1
    return state


def selection_respects_cluster_caps(state, args):
    limit = int(args.max_docs_per_cluster)
    if limit <= 0:
        return True
    if any(count > limit for count in state["title_pattern_counts"].values()):
        return False
    if any(count > limit for count in state["entity_cluster_counts"].values()):
        return False
    return True


def selection_respects_fact_caps(state, args):
    limit = int(args.max_docs_per_fact)
    if limit <= 0:
        return True
    if any(count > limit for count in state["fact_counts"].values()):
        return False
    return True


def selection_respects_caps(state, args):
    return selection_respects_fact_caps(state, args) and selection_respects_cluster_caps(state, args)


def get_selection_violations(state, args):
    violations = set()
    if not selection_respects_fact_caps(state, args):
        violations.add("fact_cap")
    if not selection_respects_cluster_caps(state, args):
        violations.add("cluster_cap")
    return violations


def candidate_adds_new_facts(candidate, selected_state):
    candidate_facts = set(candidate.get("matched_atomic_fact_ids") or set())
    return bool(candidate_facts - set(selected_state.get("covered_facts") or set()))


def choose_replacement_candidate(selected, candidate, args):
    if not selected:
        return None

    current_state = build_selection_state(selected)
    current_covered = set(current_state["covered_facts"])
    if not (set(candidate.get("matched_atomic_fact_ids") or set()) - current_covered):
        return None

    current_coverage_size = len(current_covered)
    best = None

    for idx, victim in enumerate(selected):
        trial = list(selected)
        trial[idx] = candidate
        trial_state = build_selection_state(trial)
        if not selection_respects_caps(trial_state, args):
            continue

        trial_coverage_size = len(trial_state["covered_facts"])
        coverage_gain = trial_coverage_size - current_coverage_size
        if coverage_gain <= 0:
            continue

        victim_unique_facts = sum(
            1
            for fact_id in victim.get("matched_atomic_fact_ids", set())
            if current_state["fact_counts"].get(fact_id, 0) == 1 and fact_id not in candidate.get("matched_atomic_fact_ids", set())
        )
        option = (
            int(coverage_gain),
            1 if victim_unique_facts == 0 else 0,
            -int(victim_unique_facts),
            -float(victim.get("score", 0.0)),
            -idx,
        )
        if best is None or option > best["option"]:
            best = {
                "index": idx,
                "victim_docid": victim.get("docid"),
                "coverage_gain": int(coverage_gain),
                "victim_unique_facts": int(victim_unique_facts),
                "option": option,
            }

    return best


def select_diverse_docs(ranked_docs, final_topk, args):
    if final_topk <= 0:
        return [], {
            "candidate_pool_size": len(ranked_docs),
            "selected_doc_count": 0,
            "covered_fact_count": 0,
            "skipped_fact_cap": 0,
            "skipped_cluster_cap": 0,
            "soft_fill_count": 0,
            "replacement_count": 0,
        }

    selected = []
    skipped = []
    stats = {
        "candidate_pool_size": len(ranked_docs),
        "selected_doc_count": 0,
        "covered_fact_count": 0,
        "skipped_fact_cap": 0,
        "skipped_cluster_cap": 0,
        "soft_fill_count": 0,
        "replacement_count": 0,
    }
    ranked_index = {item["docid"]: idx for idx, item in enumerate(ranked_docs)}

    for item in ranked_docs:
        if len(selected) >= final_topk:
            skipped.append(item)
            continue

        trial = selected + [item]
        trial_state = build_selection_state(trial)
        if selection_respects_caps(trial_state, args):
            selected.append(item)
            continue

        violations = get_selection_violations(trial_state, args)
        if "fact_cap" in violations:
            stats["skipped_fact_cap"] += 1
        if "cluster_cap" in violations:
            stats["skipped_cluster_cap"] += 1
        skipped.append(item)

    if len(selected) < final_topk:
        for item in skipped:
            if len(selected) >= final_topk:
                break
            trial = selected + [item]
            trial_state = build_selection_state(trial)
            if selection_respects_cluster_caps(trial_state, args):
                selected.append(item)
                stats["soft_fill_count"] += 1

    if len(selected) < final_topk:
        selected_docids = {item["docid"] for item in selected}
        for item in ranked_docs:
            if len(selected) >= final_topk:
                break
            if item["docid"] in selected_docids:
                continue
            selected.append(item)
            selected_docids.add(item["docid"])
            stats["soft_fill_count"] += 1

    selected_docids = {item["docid"] for item in selected}
    for item in ranked_docs:
        if item["docid"] in selected_docids:
            continue
        replacement = choose_replacement_candidate(selected, item, args)
        if replacement is None:
            continue
        victim_docid = selected[replacement["index"]]["docid"]
        selected[replacement["index"]] = item
        selected_docids.discard(victim_docid)
        selected_docids.add(item["docid"])
        stats["replacement_count"] += 1

    selected.sort(key=lambda item: ranked_index.get(item["docid"], 10**9))
    final_state = build_selection_state(selected)
    stats["selected_doc_count"] = len(selected)
    stats["covered_fact_count"] = len(final_state["covered_facts"])
    return selected[:final_topk], stats


def serialize_doc_record(item):
    retrieval_routes = sorted(item["retrieval_routes"], key=lambda x: ROLE_ORDER.get(x, 99))
    matched_atomic_fact_roles = sorted(item["matched_atomic_fact_roles"], key=lambda x: ROLE_ORDER.get(x, 99))
    matched_role_fact_ids = {
        role: sorted(fact_ids)
        for role, fact_ids in sorted(item["matched_role_fact_ids"].items(), key=lambda x: ROLE_ORDER.get(x[0], 99))
        if fact_ids
    }
    return {
        "docid": item["docid"],
        "score": float(item["score"]),
        "text": item["text"],
        "claim_score": None if item["claim_score"] is None else float(item["claim_score"]),
        "claim_rank": item["claim_rank"],
        "norm_bm25_claim": float(item["norm_bm25_claim"]),
        "max_fact_score": float(item["max_fact_score"]),
        "atomic_fact_score": float(item["atomic_fact_score"]),
        "num_fact_hits": int(item["num_fact_hits"]),
        "fact_query_hit_count": int(item["fact_query_hit_count"]),
        "role_bonus": float(item["role_bonus"]),
        "matched_atomic_fact_ids": sorted(item["matched_atomic_fact_ids"]),
        "matched_atomic_fact_roles": matched_atomic_fact_roles,
        "matched_role_fact_ids": matched_role_fact_ids,
        "retrieval_routes": retrieval_routes,
        "role_best_scores": {
            role: float(score)
            for role, score in sorted(item["role_best_scores"].items(), key=lambda x: ROLE_ORDER.get(x[0], 99))
            if float(score) > 0
        },
        "fusion_components": {
            "norm_bm25_claim": float(item["norm_bm25_claim"]),
            "max_fact_score": float(item["max_fact_score"]),
            "num_fact_hits": int(item["num_fact_hits"]),
            "role_bonus": float(item["role_bonus"]),
        },
        "title_pattern": item["title_pattern"],
        "entity_cluster": item["entity_cluster"],
    }


def merge_role_aware_hits(claim_hits, role_results, final_topk, args):
    merged = {}
    claim_best_score = max((float(hit["raw_score"]) for hit in claim_hits), default=0.0)

    for hit in claim_hits:
        item = get_doc_record(merged, hit)
        item["retrieval_routes"].add("claim")
        item["claim_score"] = float(hit["raw_score"]) if item["claim_score"] is None else max(item["claim_score"], float(hit["raw_score"]))
        item["claim_rank"] = int(hit["rank"]) if item["claim_rank"] is None else min(item["claim_rank"], int(hit["rank"]))
        item["norm_bm25_claim"] = max(item["norm_bm25_claim"], normalize_hit_score(hit["raw_score"], claim_best_score))

    for role, queries in role_results.items():
        for query_item in queries:
            hits = query_item.get("hits") or []
            best_score = max((float(hit["raw_score"]) for hit in hits), default=0.0)
            for hit in hits:
                item = get_doc_record(merged, hit)
                norm_score = normalize_hit_score(hit["raw_score"], best_score)
                item["retrieval_routes"].add(role)
                item["matched_atomic_fact_roles"].add(role)
                item["role_best_scores"][role] = max(float(item["role_best_scores"].get(role, 0.0)), norm_score)
                item["max_fact_score"] = max(float(item["max_fact_score"]), norm_score)
                item["fact_query_hit_count"] += 1

                for fact_id in query_item.get("fact_ids", []):
                    fact_id = str(fact_id)
                    item["matched_atomic_fact_ids"].add(fact_id)
                    item["matched_role_fact_ids"][role].add(fact_id)
                    item["fact_best_scores"][fact_id] = max(float(item["fact_best_scores"].get(fact_id, 0.0)), norm_score)

    for item in merged.values():
        item["num_fact_hits"] = len(item["matched_atomic_fact_ids"])
        item["atomic_fact_score"] = float(item["max_fact_score"])
        item["role_bonus"] = compute_role_bonus(item, args)
        item["title_pattern"] = normalize_title_pattern(item["docid"])
        item["entity_cluster"] = infer_entity_cluster(item["docid"])
        item["score"] = (
            float(args.w_claim) * float(item["norm_bm25_claim"])
            + float(args.w_fact) * float(item["max_fact_score"])
            + float(args.w_multi) * float(item["num_fact_hits"])
            + float(args.w_role) * float(item["role_bonus"])
        )

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -float(item["score"]),
            -int(item["num_fact_hits"]),
            -len(item["retrieval_routes"]),
            item["claim_rank"] if item["claim_rank"] is not None else 10**9,
            int(item["best_rank"]),
            item["docid"],
        ),
    )
    selected, diversity_stats = select_diverse_docs(ranked, final_topk, args)
    return [serialize_doc_record(item) for item in selected], diversity_stats


def main(args):
    if args.w_claim is None:
        args.w_claim = float(args.claim_rrf_weight)
    if args.w_fact is None:
        args.w_fact = float(args.atomic_rrf_weight)

    searcher = LuceneSearcher(args.index_path)
    searcher.set_bm25(0.9, 0.4)

    in_path = args.in_path.replace("[SPLIT]", args.split)
    with open(in_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    decomposition_path = args.decomposition_path.replace("[SPLIT]", args.split).replace("[PLAN]", args.plan) if args.decomposition_path else ""
    decomposition_map = load_decomposition_map(decomposition_path) if decomposition_path else {}

    claim_topk = resolve_claim_topk(args)
    role_topk_map = resolve_role_topk(args, claim_topk)

    results = []
    for data in tqdm(dataset):
        claim = data["claim"]
        gold_evidence = data.get("gold_evidence_list", data.get("gold_evidence", []))
        label = data.get("label")
        num_hops = data.get("num_hops")

        claim_hits = run_query(searcher, claim, topk=claim_topk)
        role_queries, fact_stats = build_role_queries(data, decomposition_map, args)

        role_results = {"critical": [], "leaf": [], "bridge": []}
        for role, queries in role_queries.items():
            for query_item in queries:
                role_results.setdefault(role, []).append({
                    **query_item,
                    "hits": run_query(searcher, query_item["query"], topk=role_topk_map.get(role, 0)),
                })

        fact_query_count = sum(len(items) for items in role_results.values())
        final_topk = resolve_final_topk(args, claim_topk, fact_query_count, role_topk_map)
        retrieved_docs, diversity_stats = merge_role_aware_hits(claim_hits, role_results, final_topk, args)

        result = {
            "id": data["id"],
            "claim": claim,
            "gold_evidence": gold_evidence,
            "label": label,
            "num_hops": num_hops,
            "retrieved_docs": retrieved_docs,
            "retrieval_summary": {
                "claim_topk": int(claim_topk),
                "critical_topk": int(role_topk_map.get("critical", 0)),
                "leaf_topk": int(role_topk_map.get("leaf", 0)),
                "bridge_topk": int(role_topk_map.get("bridge", 0)),
                "final_topk": int(final_topk),
                "fact_count": int(fact_stats.get("fact_count", 0)),
                "critical_fact_count": int(fact_stats.get("critical_count", 0)),
                "fact_query_count": int(fact_query_count),
                "fact_query_count_by_role": {
                    role: int(len(items))
                    for role, items in sorted(role_results.items(), key=lambda x: FACT_ROLE_PRIORITY.get(x[0], 99))
                },
                "used_role_fact_bm25": bool(fact_query_count),
                "fusion_weights": {
                    "w_claim": float(args.w_claim),
                    "w_fact": float(args.w_fact),
                    "w_multi": float(args.w_multi),
                    "w_role": float(args.w_role),
                },
                "role_bonus_weights": {
                    "claim": float(args.claim_role_bonus),
                    "critical": float(args.critical_role_bonus),
                    "leaf": float(args.leaf_role_bonus),
                    "bridge": float(args.bridge_role_bonus),
                },
                "diversity": {
                    "max_docs_per_fact": int(args.max_docs_per_fact),
                    "max_docs_per_cluster": int(args.max_docs_per_cluster),
                    **diversity_stats,
                },
            },
        }
        results.append(result)

    out_path = args.out_path.replace("[SPLIT]", args.split).replace("[PLAN]", args.plan)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Retrieve {len(results)} samples from {in_path} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="/mnt/data/yangjun/data/HOVER/data/converted_data/[SPLIT]_full.json")
    parser.add_argument("--out_path", type=str, default="./data/[PLAN]/bm25_[SPLIT].json")
    parser.add_argument("--index_path", type=str, default="/mnt/data/yangjun/data/HOVER/corpus/index")
    parser.add_argument("--decomposition_path", type=str, default="data/[PLAN]/dev_2_decomposed_0_4000.json")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--claim_topk", type=int, default=0)
    parser.add_argument("--atomic_topk", type=int, default=0)
    parser.add_argument("--critical_topk", type=int, default=0)
    parser.add_argument("--leaf_topk", type=int, default=0)
    parser.add_argument("--bridge_topk", type=int, default=0)
    parser.add_argument("--union_topk", type=int, default=0)
    parser.add_argument("--final_topk", type=int, default=0)
    parser.add_argument("--max_atomic_fact_queries", type=int, default=0)
    parser.add_argument("--deep_fact_depth_threshold", type=int, default=3)
    parser.add_argument("--max_docs_per_fact", type=int, default=2)
    parser.add_argument("--max_docs_per_cluster", type=int, default=2)
    parser.add_argument("--w_claim", type=float, default=None)
    parser.add_argument("--w_fact", type=float, default=None)
    parser.add_argument("--w_multi", type=float, default=0.15)
    parser.add_argument("--w_role", type=float, default=0.20)
    parser.add_argument("--claim_role_bonus", type=float, default=0.25)
    parser.add_argument("--critical_role_bonus", type=float, default=1.00)
    parser.add_argument("--leaf_role_bonus", type=float, default=0.70)
    parser.add_argument("--bridge_role_bonus", type=float, default=0.85)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--claim_rrf_weight", type=float, default=1.0)
    parser.add_argument("--atomic_rrf_weight", type=float, default=0.9)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--plan", type=str, default="plan5.1")

    args = parser.parse_args()
    main(args)
