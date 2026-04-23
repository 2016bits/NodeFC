from collections import defaultdict

from search_graph_hopaware import get_sim

from search_graph_decomposition_aware_modules.scoring import score_bridge_features
from search_graph_decomposition_aware_modules.shared import (
    candidate_rank_key,
    collect_support_from_sids,
    compute_fact_coverage_status,
    direct_support_tier_rank,
    get_fact_bridge_helper_budget,
    get_fact_direct_winner_budget,
)


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


def _normalize_feature(value, minimum, maximum):
    if maximum - minimum < 1e-12:
        return 0.0
    return (float(value) - float(minimum)) / (float(maximum) - float(minimum))


def _compute_global_uncovered_fact_gain(fact_support, fact_results):
    if not fact_support:
        return 0.0, [], [], []

    gains = []
    uncovered_fact_ids = []
    bridge_need_fact_ids = []
    critical_fact_ids = []
    for fact_id, support in fact_support.items():
        fact_result = fact_results.get(fact_id) or {}
        summary = fact_result.get("coverage_summary") or {}
        gain = 0.0
        if not summary.get("covered", False):
            uncovered_fact_ids.append(fact_id)
            gain = max(
                gain,
                1.0 if support.get("direct_support_pass") else (
                    0.78 if support.get("bridge_support_pass") else 0.52 * float(support.get("fact_match_score", 0.0))
                ),
            )
        if summary.get("requires_direct_support") and not summary.get("has_direct_support", False) and support.get("direct_support_pass"):
            gain = max(gain, 0.92)
        if (
            summary.get("needs_dependency_completion")
            or summary.get("needs_cross_doc_bridge_completion")
        ) and (
            support.get("bridge_support_pass")
            or support.get("bridge_assisted_direct_pass")
            or support.get("dependency_closure_ready")
        ):
            bridge_need_fact_ids.append(fact_id)
            gain = max(gain, 0.86)
        if fact_result.get("critical"):
            critical_fact_ids.append(fact_id)
            gain += 0.15
        gains.append(gain)

    denominator = max(1.0, min(3.0, float(len(gains))))
    return min(1.0, sum(gains) / denominator), sorted(set(uncovered_fact_ids)), sorted(set(bridge_need_fact_ids)), sorted(set(critical_fact_ids))


def _compute_global_redundancy_scores(candidates, semantic_sim_map, args):
    if not candidates:
        return {}

    ordered = sorted(candidates, key=lambda cand: float(cand.get("_base_rerank_score", 0.0)), reverse=True)
    redundancy_scores = {}
    previous = []
    for cand in ordered:
        max_sim = 0.0
        same_doc = 0.0
        for prev in previous:
            if cand.get("docid") and cand.get("docid") == prev.get("docid"):
                same_doc = 1.0
            max_sim = max(max_sim, get_sim(semantic_sim_map, cand["sid"], prev["sid"]))
        semantic_penalty = 0.0
        if max_sim > args.redundancy_threshold:
            semantic_penalty = (max_sim - args.redundancy_threshold) / max(1e-6, 1.0 - args.redundancy_threshold)
        redundancy_scores[cand["sid"]] = min(
            1.0,
            max(0.0, semantic_penalty + args.rerank_same_doc_redundancy_penalty * same_doc),
        )
        previous.append(cand)
    return redundancy_scores


def _candidate_keep_reason_priority(candidate):
    reasons = set(candidate.get("rerank_keep_reasons") or [])
    if "critical_bucket" in reasons:
        return 4
    if "bypass_bucket" in reasons:
        return 3
    if "direct_bucket" in reasons:
        return 2
    if "bridge_bucket" in reasons:
        return 1
    return 0


def _global_fact_support_sort_key(candidate, fact_id, bucket_type):
    support = (candidate.get("fact_support") or {}).get(fact_id) or {}
    if bucket_type == "bridge":
        return (
            1 if support.get("bridge_assisted_closure_rescue") else 0,
            float(support.get("bridge_potential_score", support.get("bridge_support_score", 0.0))),
            float(candidate.get("uncovered_fact_gain", 0.0)),
            float(support.get("bridge_support_score", 0.0)),
            float(candidate.get("rerank_score", candidate.get("_base_rerank_score", 0.0))),
            candidate_rank_key(candidate),
        )
    if bucket_type == "critical":
        return (
            direct_support_tier_rank(support.get("direct_support_tier", "none")),
            float(support.get("direct_support_score", 0.0)),
            float(support.get("fact_score", 0.0)),
            float(candidate.get("uncovered_fact_gain", 0.0)),
            float(support.get("bridge_support_score", 0.0)),
            float(candidate.get("rerank_score", candidate.get("_base_rerank_score", 0.0))),
            candidate_rank_key(candidate),
        )
    return (
        direct_support_tier_rank(support.get("direct_support_tier", "none")),
        float(support.get("direct_support_score", 0.0)),
        float(support.get("fact_match_score", 0.0)),
        float(candidate.get("uncovered_fact_gain", 0.0)),
        float(candidate.get("rerank_score", candidate.get("_base_rerank_score", 0.0))),
        candidate_rank_key(candidate),
    )


def _add_global_bucket_candidates(selected_map, candidates, limit, reason):
    if limit <= 0:
        return
    kept = 0
    for cand in candidates:
        sid = cand["sid"]
        existing = selected_map.get(sid)
        if existing is None:
            item = dict(cand)
            item["rerank_keep_reasons"] = list(item.get("rerank_keep_reasons") or [])
            if reason not in item["rerank_keep_reasons"]:
                item["rerank_keep_reasons"].append(reason)
            selected_map[sid] = item
            kept += 1
        else:
            reasons = list(existing.get("rerank_keep_reasons") or [])
            if reason not in reasons:
                reasons.append(reason)
                existing["rerank_keep_reasons"] = reasons
        if kept >= limit:
            break


def build_global_candidate_view(fact_sequence, fact_results, fact_stats, semantic_sim_map, args, topk):
    if topk <= 0:
        return []

    support_topk_per_fact = max(
        int(args.per_fact_output_k),
        int(args.rerank_keep_direct_per_fact),
        int(args.rerank_keep_bridge_per_fact),
        int(args.rerank_keep_critical_per_fact),
        int(args.rerank_keep_bypass_per_fact),
    )
    sentence_pool = build_sentence_support_pool(fact_results, topk_per_fact=support_topk_per_fact)
    if not sentence_pool:
        return []

    candidates = []
    for sid, item in sentence_pool.items():
        supports = list((item.get("fact_support") or {}).values())
        if not supports:
            continue
        uncovered_fact_gain, uncovered_fact_ids, bridge_need_fact_ids, critical_fact_ids = _compute_global_uncovered_fact_gain(
            item["fact_support"],
            fact_results,
        )
        candidate = {
            "sid": sid,
            "text": item["text"],
            "docid": item.get("docid"),
            "doc_rank": int(item.get("doc_rank", 10**9)),
            "is_title": bool(item.get("is_title", False)),
            "source_fact_ids": sorted(item.get("source_fact_ids") or []),
            "supporting_facts": sorted(item.get("source_fact_ids") or []),
            "fact_support": item.get("fact_support") or {},
            "ce_score": max(float(support.get("ce_score", 0.0)) for support in supports),
            "aggregate_score": max(float(support.get("aggregate_score", 0.0)) for support in supports),
            "fact_score": max(float(support.get("fact_score", 0.0)) for support in supports),
            "coverage_score": max(float(support.get("coverage_score", 0.0)) for support in supports),
            "direct_support_score": max(float(support.get("direct_support_score", 0.0)) for support in supports),
            "bridge_support_score": max(float(support.get("bridge_support_score", 0.0)) for support in supports),
            "fact_match_score": max(float(support.get("fact_match_score", 0.0)) for support in supports),
            "bridge_potential_score": max(float(support.get("bridge_potential_score", 0.0)) for support in supports),
            "uncovered_fact_gain": float(uncovered_fact_gain),
            "direct_support_tier": max(
                [support.get("direct_support_tier", "none") for support in supports],
                key=direct_support_tier_rank,
            ),
            "support_type": "direct_support" if any(support.get("support_type") == "direct_support" for support in supports) else (
                "bridge_support" if any(support.get("bridge_support_pass") for support in supports) else "candidate"
            ),
            "weak_direct_rescue": any(bool(support.get("weak_direct_rescue")) for support in supports),
            "bridge_assisted_closure_rescue": any(bool(support.get("bridge_assisted_closure_rescue")) for support in supports),
            "rerank_bypass_pass": any(bool(support.get("rerank_bypass_pass")) for support in supports),
            "critical_supportive_candidate": any(bool(support.get("critical_supportive_candidate")) for support in supports),
            "uncovered_fact_ids": uncovered_fact_ids,
            "bridge_need_fact_ids": bridge_need_fact_ids,
            "critical_fact_ids": critical_fact_ids,
            "rerank_keep_reasons": [],
        }
        candidates.append(candidate)

    if not candidates:
        return []

    ce_min = min(float(candidate.get("ce_score", 0.0)) for candidate in candidates)
    ce_max = max(float(candidate.get("ce_score", 0.0)) for candidate in candidates)
    for candidate in candidates:
        ce_norm = _normalize_feature(candidate.get("ce_score", 0.0), ce_min, ce_max)
        candidate["ce_norm"] = float(ce_norm)
        candidate["_base_rerank_score"] = float(
            args.rerank_weight_ce * ce_norm
            + args.rerank_weight_fact_match * float(candidate.get("fact_match_score", 0.0))
            + args.rerank_weight_direct_support * float(candidate.get("direct_support_score", 0.0))
            + args.rerank_weight_bridge_potential * float(candidate.get("bridge_potential_score", 0.0))
            + args.rerank_weight_uncovered_fact_gain * float(candidate.get("uncovered_fact_gain", 0.0))
        )

    redundancy_scores = _compute_global_redundancy_scores(candidates, semantic_sim_map or {}, args)
    for candidate in candidates:
        redundancy_penalty = float(redundancy_scores.get(candidate["sid"], 0.0))
        candidate["redundancy_penalty"] = redundancy_penalty
        candidate["rerank_score"] = float(
            float(candidate.get("_base_rerank_score", 0.0))
            - args.rerank_weight_redundancy * redundancy_penalty
        )

    ranked_candidates = sorted(candidates, key=candidate_rank_key, reverse=True)
    selected_map = {}
    result_by_id = fact_results if isinstance(fact_results, dict) else {}

    ordered_facts = list(fact_sequence or [])
    if not ordered_facts:
        ordered_facts = sorted(
            [result for result in result_by_id.values() if isinstance(result, dict) and "fact_id" in result],
            key=lambda item: (
                (fact_stats.get("depth_map") or {}).get(item.get("fact_id"), 1),
                0 if item.get("critical") else 1,
            ),
        )

    for fact in ordered_facts:
        fact_id = fact["id"] if isinstance(fact, dict) else fact.get("fact_id")
        fact_result = result_by_id.get(fact_id) or {}
        summary = fact_result.get("coverage_summary") or {}

        direct_bucket = sorted(
            [
                cand for cand in ranked_candidates
                if fact_id in (cand.get("fact_support") or {})
                and (
                    (cand["fact_support"][fact_id].get("direct_support_pass"))
                    or cand["fact_support"][fact_id].get("weak_direct_rescue")
                )
            ],
            key=lambda cand: _global_fact_support_sort_key(cand, fact_id, "direct"),
            reverse=True,
        )
        _add_global_bucket_candidates(selected_map, direct_bucket, args.rerank_keep_direct_per_fact, "direct_bucket")

        if fact_result.get("role") == "bridge" or fact_result.get("rely_on"):
            bridge_bucket = sorted(
                [
                    cand for cand in ranked_candidates
                    if fact_id in (cand.get("fact_support") or {})
                    and (
                        cand["fact_support"][fact_id].get("bridge_support_pass")
                        or cand["fact_support"][fact_id].get("bridge_assisted_closure_rescue")
                        or cand["fact_support"][fact_id].get("dependency_closure_ready")
                    )
                ],
                key=lambda cand: _global_fact_support_sort_key(cand, fact_id, "bridge"),
                reverse=True,
            )
            _add_global_bucket_candidates(selected_map, bridge_bucket, args.rerank_keep_bridge_per_fact, "bridge_bucket")

        if fact_result.get("critical"):
            critical_bucket = sorted(
                [
                    cand for cand in ranked_candidates
                    if fact_id in (cand.get("fact_support") or {})
                    and (
                        cand["fact_support"][fact_id].get("direct_support_pass")
                        or cand["fact_support"][fact_id].get("bridge_support_pass")
                        or cand["fact_support"][fact_id].get("rerank_bypass_pass")
                    )
                ],
                key=lambda cand: _global_fact_support_sort_key(cand, fact_id, "critical"),
                reverse=True,
            )
            _add_global_bucket_candidates(selected_map, critical_bucket, args.rerank_keep_critical_per_fact, "critical_bucket")

    bypass_bucket = sorted(
        [cand for cand in ranked_candidates if cand.get("rerank_bypass_pass")],
        key=candidate_rank_key,
        reverse=True,
    )
    _add_global_bucket_candidates(
        selected_map,
        bypass_bucket,
        args.rerank_keep_bypass_per_fact,
        "bypass_bucket",
    )

    protected = sorted(
        selected_map.values(),
        key=lambda cand: (_candidate_keep_reason_priority(cand), candidate_rank_key(cand)),
        reverse=True,
    )
    if len(protected) > topk:
        protected = protected[:topk]
        selected_map = {cand["sid"]: cand for cand in protected}

    global_pool = ranked_candidates[:args.rerank_keep_global]
    for cand in global_pool:
        if len(selected_map) >= topk:
            break
        if cand["sid"] in selected_map:
            continue
        item = dict(cand)
        item["rerank_keep_reasons"] = list(item.get("rerank_keep_reasons") or [])
        if "global_pool" not in item["rerank_keep_reasons"]:
            item["rerank_keep_reasons"].append("global_pool")
        selected_map[item["sid"]] = item

    if len(selected_map) < topk:
        for cand in ranked_candidates:
            if len(selected_map) >= topk:
                break
            if cand["sid"] in selected_map:
                continue
            item = dict(cand)
            item["rerank_keep_reasons"] = list(item.get("rerank_keep_reasons") or [])
            if not item["rerank_keep_reasons"]:
                item["rerank_keep_reasons"].append("score_fill")
            selected_map[item["sid"]] = item

    final_candidates = sorted(selected_map.values(), key=candidate_rank_key, reverse=True)[:topk]
    exported = []
    for cand in final_candidates:
        item = dict(cand)
        item.pop("fact_support", None)
        item.pop("_base_rerank_score", None)
        exported.append(item)
    return exported


def _candidate_is_title(candidate, sentence_pool=None):
    if not candidate:
        return False
    if "is_title" in candidate:
        return bool(candidate.get("is_title", False))
    sid = candidate.get("sid")
    if sid is None or sentence_pool is None:
        return False
    return bool((sentence_pool.get(sid) or {}).get("is_title", False))


def _assembly_candidate_sort_key(candidate, sentence_pool, args):
    return (
        0 if _candidate_is_title(candidate, sentence_pool) else 1,
        candidate_rank_key(candidate),
        -float(getattr(args, "assembly_title_penalty_weight", 0.0) or 0.0) if _candidate_is_title(candidate, sentence_pool) else 0.0,
    )


def _prioritize_assembly_candidates(candidates, sentence_pool, args):
    return sorted(candidates or [], key=lambda cand: _assembly_candidate_sort_key(cand, sentence_pool, args), reverse=True)


def _merge_sentence_pool_candidate(sentence_pool, fact_id, cand):
    sid = cand["sid"]
    item = sentence_pool.get(sid)
    if item is None:
        item = {
            "sid": sid,
            "text": cand["text"],
            "docid": cand.get("docid"),
            "doc_rank": int(cand.get("doc_rank", 10**9)),
            "is_title": bool(cand.get("is_title", False)),
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
        item["is_title"] = bool(cand.get("is_title", False))
        item["score"] = float(cand["aggregate_score"])
    item["best_fact_score"] = max(item["best_fact_score"], float(cand["fact_score"]))
    item["fact_support"][fact_id] = {
        "ce_score": float(cand.get("ce_score", 0.0)),
        "ce_norm": float(cand.get("ce_norm", 0.0)),
        "aggregate_score": float(cand["aggregate_score"]),
        "rerank_score": float(cand.get("rerank_score", cand["aggregate_score"])),
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
        "bridge_potential_score": float(cand.get("bridge_potential_score", cand["bridge_support_score"])),
        "direct_support_score": float(cand["direct_support_score"]),
        "direct_support_pass": bool(cand["direct_support_pass"]),
        "direct_support_tier": cand.get("direct_support_tier", "none"),
        "strong_direct_support_pass": bool(cand.get("strong_direct_support_pass", False)),
        "weak_direct_support_pass": bool(cand.get("weak_direct_support_pass", False)),
        "bridge_assisted_direct_pass": bool(cand.get("bridge_assisted_direct_pass", False)),
        "weak_direct_rescue": bool(cand.get("weak_direct_rescue", False)),
        "bridge_assisted_closure_rescue": bool(cand.get("bridge_assisted_closure_rescue", False)),
        "rerank_bypass_pass": bool(cand.get("rerank_bypass_pass", False)),
        "critical_supportive_candidate": bool(cand.get("critical_supportive_candidate", False)),
        "dependency_closure_ready": bool(cand["dependency_closure_ready"]),
        "support_type": cand["support_type"],
        "fact_role": cand["fact_role"],
        "cross_doc_bridge_score": float(cand["cross_doc_bridge_score"]),
        "critical_coverage_bonus": float(cand["critical_coverage_bonus"]),
        "doc_rank_bonus": float(cand["doc_rank_bonus"]),
        "fact_match_score": float(cand.get("fact_match_score", 0.0)),
        "uncovered_fact_gain": float(cand.get("uncovered_fact_gain", 0.0)),
        "fact_completeness_penalty": float(cand.get("fact_completeness_penalty", 0.0)),
        "is_title": bool(cand.get("is_title", False)),
        "title_score_penalty": float(cand.get("title_score_penalty", 0.0)),
        "redundancy_penalty": float(cand.get("redundancy_penalty", 0.0)),
    }


def build_sentence_support_pool(fact_results, topk_per_fact):
    sentence_pool = {}
    for fact_id, fact_result in fact_results.items():
        for cand in fact_result.get("candidates", [])[:topk_per_fact]:
            _merge_sentence_pool_candidate(sentence_pool, fact_id, cand)

        summary = fact_result.get("coverage_summary") or {}
        extra_candidates = []
        extra_candidates.extend(summary.get("direct_winners") or [])
        extra_candidates.extend(summary.get("bridge_winners") or [])
        extra_candidates.extend(summary.get("direct_candidates") or [])
        extra_candidates.extend(summary.get("bridge_candidates") or [])
        for cand in extra_candidates:
            if cand:
                _merge_sentence_pool_candidate(sentence_pool, fact_id, cand)
    return sentence_pool


def _get_effective_hop_count(fact_stats):
    return max(
        int((fact_stats or {}).get("max_depth", 1)),
        int((fact_stats or {}).get("claim_num_hops") or 0),
    )


def _estimate_fact_doc_floor(fact_stats):
    fact_count = int((fact_stats or {}).get("fact_count", 0))
    if fact_count <= 2:
        return 1
    return max(2, (fact_count + 1) // 2)


def compute_dynamic_doc_budget(fact_sequence, fact_stats, sentence_pool, args):
    candidate_doc_count = len({item["docid"] for item in sentence_pool.values() if item.get("docid")})
    hop_count = _get_effective_hop_count(fact_stats)
    fact_doc_floor = min(args.max_docs_per_claim_cap, _estimate_fact_doc_floor(fact_stats))
    budget = args.base_max_docs_per_claim
    if hop_count <= 2:
        budget = max(budget, min(fact_doc_floor, args.base_max_docs_per_claim + 1))
        if fact_stats["critical_count"] >= 2:
            budget += 1
    elif hop_count == 3:
        budget = max(budget, hop_count, fact_doc_floor)
        if fact_stats["critical_count"] >= 2:
            budget += 1
        if candidate_doc_count >= args.doc_budget_candidate_docs_threshold:
            budget += 1
    else:
        budget = max(budget, hop_count + 1, fact_doc_floor + 1)
        if fact_stats["critical_count"] >= 2:
            budget += 1
        if candidate_doc_count >= args.doc_budget_candidate_docs_threshold:
            budget += 1
    budget = max(1, min(args.max_docs_per_claim_cap, budget))
    return budget, {
        "base_max_docs_per_claim": int(args.base_max_docs_per_claim),
        "claim_num_hops": int((fact_stats or {}).get("claim_num_hops") or 0),
        "effective_hop_count": int(hop_count),
        "fact_count": int(fact_stats["fact_count"]),
        "fact_doc_floor": int(fact_doc_floor),
        "dag_depth": int(fact_stats["max_depth"]),
        "critical_fact_count": int(fact_stats["critical_count"]),
        "candidate_doc_count": int(candidate_doc_count),
        "dynamic_max_docs_per_claim": int(budget),
    }


def _scale_multihop_weight(base_weight, max_depth, hop3_multiplier, hop4_multiplier):
    weight = float(base_weight)
    if max_depth >= 4:
        return weight * float(hop4_multiplier)
    if max_depth >= 3:
        return weight * float(hop3_multiplier)
    return weight


def _compute_doc_soft_free_allowance(fact_stats, args):
    hop_count = _get_effective_hop_count(fact_stats)
    fact_doc_floor = _estimate_fact_doc_floor(fact_stats)
    free_docs = args.base_max_docs_per_claim
    if hop_count == 3:
        free_docs = max(free_docs, min(args.max_docs_per_claim_cap, max(3, fact_doc_floor)))
    elif hop_count >= 4:
        free_docs = max(free_docs, min(args.max_docs_per_claim_cap, max(4, fact_doc_floor + 1)))
    return int(max(1, min(args.max_docs_per_claim_cap, free_docs)))


def _compute_assembly_doc_penalty(selected_docs, fact_stats, args):
    free_docs = _compute_doc_soft_free_allowance(fact_stats, args)
    return float(max(0, len(selected_docs) - free_docs))


def _count_new_fact_carrying_docs(current_state, trial_state):
    current_covered = current_state.get("covered_facts") or set()
    trial_doc_fact_map = trial_state.get("doc_fact_map") or {}
    new_docids = (trial_state.get("docids") or set()) - (current_state.get("docids") or set())
    informative_docs = 0.0
    for docid in new_docids:
        new_facts = set(trial_doc_fact_map.get(docid) or set()) - set(current_covered)
        if new_facts:
            informative_docs += 1.0
    return float(informative_docs)


def _collect_direct_tier_flags(records):
    tiers = [item["support"].get("direct_support_tier", "none") for item in records]
    return {
        "has_strong_direct_support": any(tier == "strong" for tier in tiers),
        "has_weak_direct_support": any(tier == "weak" for tier in tiers),
        "has_bridge_assisted_direct": any(tier == "bridge_assisted" for tier in tiers),
        "best_direct_support_tier": max(tiers, key=direct_support_tier_rank) if tiers else "none",
    }


def _compute_assembly_utility_components(state, fact_stats, args):
    max_depth = _get_effective_hop_count(fact_stats)
    critical_weight = _scale_multihop_weight(
        args.assembly_critical_covered_weight,
        max_depth,
        args.assembly_3hop_critical_multiplier,
        args.assembly_4hop_critical_multiplier,
    )
    dependency_weight = _scale_multihop_weight(
        args.assembly_dependency_closed_weight,
        max_depth,
        args.assembly_3hop_dependency_multiplier,
        args.assembly_4hop_dependency_multiplier,
    )
    return {
        "fact_weight": float(args.assembly_fact_covered_weight),
        "critical_weight": float(critical_weight),
        "dependency_weight": float(dependency_weight),
        "bridge_weight": float(args.assembly_bridge_closed_weight),
        "anchor_weight": float(args.assembly_anchor_satisfied_weight),
        "redundancy_weight": float(args.assembly_redundancy_weight),
        "doc_weight": float(args.assembly_doc_penalty_weight),
        "title_weight": float(args.assembly_title_penalty_weight),
    }


def _compute_assembly_gain(current_state, trial_state, fact_stats, args):
    weights = _compute_assembly_utility_components(trial_state, fact_stats, args)
    new_fact_covered = len(trial_state["covered_facts"] - current_state["covered_facts"])
    new_critical_covered = len(trial_state["critical_covered_facts"] - current_state["critical_covered_facts"])
    new_dependency_closed = len(trial_state["dependency_closed_facts"] - current_state["dependency_closed_facts"])
    new_bridge_closed = len(trial_state["bridge_closed_facts"] - current_state["bridge_closed_facts"])
    new_anchor_satisfied = len(trial_state["anchor_satisfied_facts"] - current_state["anchor_satisfied_facts"])
    new_redundancy = max(0.0, float(trial_state["redundancy"]) - float(current_state["redundancy"]))
    new_doc_penalty = max(0.0, float(trial_state["doc_penalty"]) - float(current_state["doc_penalty"]))
    new_title_penalty = max(0.0, float(trial_state["title_penalty"]) - float(current_state["title_penalty"]))
    doc_fact_credit = args.assembly_doc_fact_credit * _count_new_fact_carrying_docs(current_state, trial_state)
    new_doc_penalty = max(0.0, new_doc_penalty - doc_fact_credit)
    gain = (
        weights["fact_weight"] * new_fact_covered
        + weights["critical_weight"] * new_critical_covered
        + weights["dependency_weight"] * new_dependency_closed
        + weights["bridge_weight"] * new_bridge_closed
        + weights["anchor_weight"] * new_anchor_satisfied
        - weights["redundancy_weight"] * new_redundancy
        - weights["doc_weight"] * new_doc_penalty
        - weights["title_weight"] * new_title_penalty
    )
    return {
        "gain": float(gain),
        "new_fact_covered": int(new_fact_covered),
        "new_critical_covered": int(new_critical_covered),
        "new_dependency_closed": int(new_dependency_closed),
        "new_bridge_closed": int(new_bridge_closed),
        "new_anchor_satisfied": int(new_anchor_satisfied),
        "new_redundancy": float(new_redundancy),
        "new_doc_penalty": float(new_doc_penalty),
        "new_title_penalty": float(new_title_penalty),
        "doc_fact_credit": float(doc_fact_credit),
        "weights": weights,
    }


def score_bridge_against_selected_sids(sid, selected_parent_sids, context, semantic_sim_map, args):
    parent_support = collect_support_from_sids(selected_parent_sids, context)
    return score_bridge_features(sid, parent_support, context, semantic_sim_map, args)


def _zero_bridge_eval():
    return {
        "score": 0.0,
        "same_doc": 0.0,
        "entity_overlap": 0.0,
        "relation_overlap": 0.0,
        "constraint_overlap": 0.0,
        "semantic": 0.0,
        "cross_doc": 0.0,
        "satisfied": False,
    }


def _sort_direct_support_items(items):
    return sorted(
        items,
        key=lambda x: (
            0 if x[1].get("is_title") else 1,
            direct_support_tier_rank(x[1].get("direct_support_tier")),
            float(x[1].get("direct_support_score", 0.0)),
            float(x[1].get("fact_score", 0.0)),
            -float(x[1].get("fact_completeness_penalty", 0.0)),
            float(x[1].get("aggregate_score", 0.0)),
        ),
        reverse=True,
    )


def _sort_bridge_support_items(items):
    return sorted(
        items,
        key=lambda x: (
            0 if x[1].get("is_title") else 1,
            float(x[1].get("bridge_support_score", 0.0)),
            float(x[1].get("binding_score", 0.0)),
            float(x[1].get("aggregate_score", 0.0)),
            float(x[1].get("fact_score", 0.0)),
        ),
        reverse=True,
    )


def _collect_parent_support_sids(parent_ids, fact_witnesses):
    ordered = []
    seen = set()
    for pid in parent_ids:
        witness = fact_witnesses.get(pid) or {}
        support_sids = witness.get("support_sids") or []
        if not support_sids and witness.get("sid"):
            support_sids = [witness["sid"]]
        helper_sids = witness.get("helper_sids") or []
        for sid in list(support_sids) + list(helper_sids):
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
    return ordered


def _can_add_sid_with_doc_budget(sid, sentence_pool, selected_docids, doc_budget):
    docid = sentence_pool.get(sid, {}).get("docid")
    if docid and docid not in selected_docids and len(selected_docids) >= doc_budget:
        return False
    return True


def _add_sid_to_selection(sid, selected_sids, selected_sid_set, selected_docids, sentence_pool):
    if sid in selected_sid_set:
        return
    selected_sids.append(sid)
    selected_sid_set.add(sid)
    docid = sentence_pool.get(sid, {}).get("docid")
    if docid:
        selected_docids.add(docid)


def _merge_selection_candidates(*candidate_lists):
    merged = {}
    for candidates in candidate_lists:
        for cand in candidates or []:
            sid = cand.get("sid")
            if not sid:
                continue
            prev = merged.get(sid)
            if prev is None or candidate_rank_key(cand) > candidate_rank_key(prev):
                merged[sid] = cand
    return sorted(merged.values(), key=candidate_rank_key, reverse=True)


def evaluate_selected_set(selected_sids, sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, args):
    selected_sids = [sid for sid in selected_sids if sid in sentence_pool]
    selected_docs = {sentence_pool[sid]["docid"] for sid in selected_sids if sentence_pool[sid].get("docid")}

    fact_direct_candidates = defaultdict(list)
    fact_bridge_candidates = defaultdict(list)
    for sid in selected_sids:
        for fact_id, support in sentence_pool[sid]["fact_support"].items():
            if support.get("direct_support_pass"):
                fact_direct_candidates[fact_id].append((sid, support))
            if support.get("bridge_support_pass"):
                fact_bridge_candidates[fact_id].append((sid, support))

    ordered_facts = sorted(
        fact_sequence,
        key=lambda fact: (fact_stats["depth_map"].get(fact["id"], 1), 0 if fact.get("critical") else 1),
    )
    covered_facts = set()
    fully_covered_facts = set()
    critical_covered_facts = set()
    dependency_closed_facts = set()
    bridge_closed_facts = set()
    anchor_satisfied_facts = set()
    fact_witnesses = {}
    facts_by_sid = defaultdict(list)
    doc_fact_map = defaultdict(set)
    coverage_value = 0.0
    dependency_covered = 0
    cross_doc_bridge_count = 0

    for fact in ordered_facts:
        fid = fact["id"]
        fact_role = fact.get("role", "verify")
        parents = [pid for pid in fact.get("rely_on", []) if pid in fact_stats["id2fact"]]
        if parents and not all(pid in covered_facts for pid in parents):
            continue

        parent_support_sids = _collect_parent_support_sids(parents, fact_witnesses)
        direct_candidates = _sort_direct_support_items(
            [item for item in fact_direct_candidates.get(fid, []) if item[1].get("direct_support_pass")]
        )
        bridge_candidates = _sort_bridge_support_items(
            [item for item in fact_bridge_candidates.get(fid, []) if item[1].get("bridge_support_pass")]
        )

        direct_winner_budget = get_fact_direct_winner_budget(fact, fact_role)
        direct_winners = direct_candidates[:direct_winner_budget]
        direct_records = []
        dependency_ready = not parents
        for sid, support in direct_winners:
            bridge_eval = _zero_bridge_eval() if not parents else score_bridge_against_selected_sids(
                sid,
                parent_support_sids,
                context,
                semantic_sim_map,
                args,
            )
            direct_records.append({"sid": sid, "support": support, "bridge_eval": bridge_eval})
            dependency_ready = dependency_ready or bool(
                support.get("dependency_closure_ready")
                or bridge_eval["satisfied"]
                or bridge_eval["score"] >= args.bridge_threshold
            )

        primary_sid = direct_records[0]["sid"] if direct_records else None
        primary_support = direct_records[0]["support"] if direct_records else None
        primary_bridge_eval = direct_records[0]["bridge_eval"] if direct_records else _zero_bridge_eval()

        helper_budget = get_fact_bridge_helper_budget(fact, fact_stats, args) if parents else 0
        helper_records = []
        direct_tier_flags = _collect_direct_tier_flags(direct_records)

        if fact_role == "bridge":
            primary_record = None
            for sid, support in bridge_candidates:
                bridge_eval = _zero_bridge_eval() if not parents else score_bridge_against_selected_sids(
                    sid,
                    parent_support_sids,
                    context,
                    semantic_sim_map,
                    args,
                )
                if support.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                    primary_record = {"sid": sid, "support": support, "bridge_eval": bridge_eval}
                    dependency_ready = (not parents) or bool(
                        support.get("dependency_closure_ready")
                        or bridge_eval["satisfied"]
                        or bridge_eval["score"] >= args.bridge_threshold
                    )
                    break

            if primary_record is None:
                continue

            primary_sid = primary_record["sid"]
            primary_support = primary_record["support"]
            primary_bridge_eval = primary_record["bridge_eval"]

            if parents and not dependency_ready and helper_budget > 1:
                for sid, support in bridge_candidates:
                    if sid == primary_sid or sid in {item["sid"] for item in helper_records}:
                        continue
                    bridge_eval = score_bridge_against_selected_sids(
                        sid,
                        parent_support_sids,
                        context,
                        semantic_sim_map,
                        args,
                    )
                    if support.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                        helper_records.append({"sid": sid, "support": support, "bridge_eval": bridge_eval})
                        dependency_ready = dependency_ready or bool(
                            support.get("dependency_closure_ready")
                            or bridge_eval["satisfied"]
                            or bridge_eval["score"] >= args.bridge_threshold
                        )
                        if len(helper_records) >= helper_budget - 1:
                            break

            coverage_status = compute_fact_coverage_status(
                fact=fact,
                fact_role=fact_role,
                has_direct_support=False,
                dependency_closure_ready=dependency_ready,
                has_bridge_support=True,
            )
            support_sids = [primary_sid] + [item["sid"] for item in helper_records]
            direct_sids = []
            direct_tier_flags = _collect_direct_tier_flags([])
        else:
            if parents and direct_records and not dependency_ready and helper_budget > 0:
                for sid, support in bridge_candidates:
                    if sid in {item["sid"] for item in direct_records} or sid in {item["sid"] for item in helper_records}:
                        continue
                    bridge_eval = score_bridge_against_selected_sids(
                        sid,
                        parent_support_sids,
                        context,
                        semantic_sim_map,
                        args,
                    )
                    if support.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                        helper_records.append({"sid": sid, "support": support, "bridge_eval": bridge_eval})
                        dependency_ready = dependency_ready or bool(
                            support.get("dependency_closure_ready")
                            or bridge_eval["satisfied"]
                            or bridge_eval["score"] >= args.bridge_threshold
                        )
                        if len(helper_records) >= helper_budget:
                            break

            coverage_status = compute_fact_coverage_status(
                fact=fact,
                fact_role=fact_role,
                has_direct_support=bool(direct_records),
                dependency_closure_ready=dependency_ready,
                has_bridge_support=bool(helper_records),
                has_strong_direct_support=direct_tier_flags["has_strong_direct_support"],
                has_weak_direct_support=direct_tier_flags["has_weak_direct_support"],
                has_bridge_assisted_direct=direct_tier_flags["has_bridge_assisted_direct"],
            )
            direct_sids = [item["sid"] for item in direct_records]
            support_sids = direct_sids + [item["sid"] for item in helper_records]

        if not coverage_status["covered"]:
            continue

        covered_facts.add(fid)
        if fact.get("critical"):
            critical_covered_facts.add(fid)
        if fact_role == "anchor":
            anchor_satisfied_facts.add(fid)
        if coverage_status["fully_covered"]:
            fully_covered_facts.add(fid)
        depth = fact_stats["depth_map"].get(fid, 1)
        fact_value = 1.0 + args.assembly_depth_gain * max(0, depth - 1) + args.assembly_child_gain * len(fact_stats["children"].get(fid, []))
        if fact.get("critical"):
            fact_value += 1.0
        coverage_value += fact_value
        coverage_value += args.assembly_fact_score_weight * float(primary_support.get("fact_score", 0.0))
        coverage_value += args.assembly_direct_support_weight * float(primary_support.get("direct_support_score", 0.0))
        if fact_role == "bridge":
            coverage_value += args.assembly_bridge_helper_gain * float(primary_support.get("bridge_support_score", 0.0))
        for helper in helper_records:
            coverage_value += args.assembly_bridge_helper_gain * float(helper["support"].get("bridge_support_score", 0.0))
        if coverage_status["fully_covered"]:
            coverage_value += args.assembly_fully_covered_gain

        bridge_evals = [item["bridge_eval"] for item in helper_records]
        closure_bridge_evals = [item["bridge_eval"] for item in direct_records] + bridge_evals
        if fact_role == "bridge":
            bridge_evals = [primary_bridge_eval] + bridge_evals
            closure_bridge_evals = list(bridge_evals)

        if parents and coverage_status["fully_covered"]:
            dependency_covered += 1
            dependency_closed_facts.add(fid)
            if any(bridge_eval.get("cross_doc", 0.0) > 0 for bridge_eval in closure_bridge_evals):
                cross_doc_bridge_count += 1
        bridge_closure_ready = bool(
            coverage_status["fully_covered"]
            and (
                fact_role == "bridge"
                or helper_records
                or any(
                    bridge_eval.get("satisfied")
                    or bridge_eval.get("score", 0.0) >= args.bridge_threshold
                    or bridge_eval.get("cross_doc", 0.0) > 0
                    for bridge_eval in closure_bridge_evals
                )
            )
        )
        if bridge_closure_ready:
            bridge_closed_facts.add(fid)

        fact_witnesses[fid] = {
            "sid": primary_sid,
            "direct_sids": direct_sids,
            "helper_sids": [item["sid"] for item in helper_records],
            "support_sids": support_sids,
            "fact_role": fact_role,
            "covered": bool(coverage_status["covered"]),
            "fully_covered": bool(coverage_status["fully_covered"]),
            "dependency_closed": bool(fid in dependency_closed_facts),
            "bridge_closed": bool(fid in bridge_closed_facts),
            "anchor_satisfied": bool(fid in anchor_satisfied_facts),
            "best_direct_support_tier": direct_tier_flags["best_direct_support_tier"],
            "has_strong_direct_support": bool(direct_tier_flags["has_strong_direct_support"]),
            "has_weak_direct_support": bool(direct_tier_flags["has_weak_direct_support"]),
            "has_bridge_assisted_direct": bool(direct_tier_flags["has_bridge_assisted_direct"]),
            "fact_score": float(primary_support.get("fact_score", 0.0)),
            "direct_support_score": float(max([item["support"].get("direct_support_score", 0.0) for item in direct_records] or [0.0])),
            "bridge_score": float(max(
                [item["support"].get("bridge_support_score", 0.0) for item in helper_records]
                + ([primary_support.get("bridge_support_score", 0.0)] if fact_role == "bridge" else [0.0])
            )),
            "cross_doc_bridge": float(max([bridge_eval.get("cross_doc", 0.0) for bridge_eval in bridge_evals] or [0.0])),
        }
        for sid in support_sids:
            facts_by_sid[sid].append(fid)
            docid = sentence_pool[sid].get("docid")
            if docid:
                doc_fact_map[docid].add(fid)

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

    critical_covered = len(critical_covered_facts)
    doc_penalty = _compute_assembly_doc_penalty(selected_docs, fact_stats, args)
    title_penalty = sum(1.0 for sid in selected_sids if sentence_pool.get(sid, {}).get("is_title"))
    utility_weights = _compute_assembly_utility_components({}, fact_stats, args)
    utility = 0.0
    utility += utility_weights["fact_weight"] * len(covered_facts)
    utility += utility_weights["critical_weight"] * critical_covered
    utility += utility_weights["dependency_weight"] * len(dependency_closed_facts)
    utility += utility_weights["bridge_weight"] * len(bridge_closed_facts)
    utility += utility_weights["anchor_weight"] * len(anchor_satisfied_facts)
    utility -= utility_weights["redundancy_weight"] * redundancy
    utility -= utility_weights["doc_weight"] * doc_penalty
    utility -= utility_weights["title_weight"] * title_penalty

    return {
        "utility": float(utility),
        "covered_facts": covered_facts,
        "fully_covered_facts": fully_covered_facts,
        "critical_covered_facts": critical_covered_facts,
        "dependency_closed_facts": dependency_closed_facts,
        "bridge_closed_facts": bridge_closed_facts,
        "anchor_satisfied_facts": anchor_satisfied_facts,
        "fact_witnesses": fact_witnesses,
        "facts_by_sid": {sid: fact_ids for sid, fact_ids in facts_by_sid.items()},
        "critical_covered": int(critical_covered),
        "fully_covered_count": int(len(fully_covered_facts)),
        "dependency_closed": int(len(dependency_closed_facts)),
        "dependency_covered": int(dependency_covered),
        "bridge_closed": int(len(bridge_closed_facts)),
        "anchor_satisfied": int(len(anchor_satisfied_facts)),
        "cross_doc_bridge_count": int(cross_doc_bridge_count),
        "docids": selected_docs,
        "doc_fact_map": {docid: set(fids) for docid, fids in doc_fact_map.items()},
        "doc_penalty": float(doc_penalty),
        "title_penalty": float(title_penalty),
        "coverage_value": float(coverage_value),
        "redundancy": float(redundancy),
    }


def rescue_uncovered_facts(selected_sids, sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, result_by_id, args):
    current_sids = [sid for sid in selected_sids if sid in sentence_pool]
    current_state = evaluate_selected_set(
        current_sids,
        sentence_pool,
        fact_sequence,
        fact_stats,
        context,
        semantic_sim_map,
        doc_budget,
        args,
    )
    current_sid_set = set(current_sids)

    ordered_facts = sorted(
        fact_sequence,
        key=lambda fact: (
            0 if fact.get("critical") else 1,
            -fact_stats["depth_map"].get(fact["id"], 1),
            -len(fact_stats["children"].get(fact["id"], [])),
        ),
    )
    hop_count = _get_effective_hop_count(fact_stats)

    while len(current_sids) < args.max_evidence:
        best_sid = None
        best_state = None
        best_rank = None

        for fact in ordered_facts:
            fid = fact["id"]
            fact_role = fact.get("role", "verify")
            parents = [pid for pid in fact.get("rely_on", []) if pid in fact_stats["id2fact"]]
            needs_repair = (
                fid not in current_state["covered_facts"]
                or (parents and fid not in current_state["dependency_closed_facts"])
                or (fact_role == "bridge" and fid not in current_state["bridge_closed_facts"])
                or (fact_role == "anchor" and fid not in current_state["anchor_satisfied_facts"])
            )
            if not needs_repair:
                continue

            summary = (result_by_id.get(fid) or {}).get("coverage_summary") or {}
            candidate_groups = []
            if fid not in current_state["covered_facts"]:
                candidate_groups.append(summary.get("direct_candidates") or [])
            if parents and fid not in current_state["dependency_closed_facts"]:
                candidate_groups.append(summary.get("bridge_candidates") or [])
            if summary.get("needs_cross_doc_bridge_completion"):
                candidate_groups.append(summary.get("bridge_candidates") or [])
            if fact_role == "anchor" and fid not in current_state["anchor_satisfied_facts"]:
                candidate_groups.append(summary.get("direct_candidates") or [])
            candidates = _merge_selection_candidates(*candidate_groups) if candidate_groups else _merge_selection_candidates(
                summary.get("direct_candidates") or [],
                summary.get("bridge_candidates") or [],
            )
            candidates = _prioritize_assembly_candidates(candidates, sentence_pool, args)
            if not candidates:
                continue

            for cand in candidates[:args.assembly_candidates_per_fact]:
                sid = cand["sid"]
                if sid in current_sid_set or sid not in sentence_pool:
                    continue

                trial_sids = current_sids + [sid]
                trial_state = evaluate_selected_set(
                    trial_sids,
                    sentence_pool,
                    fact_sequence,
                    fact_stats,
                    context,
                    semantic_sim_map,
                    doc_budget,
                    args,
                )
                gain_info = _compute_assembly_gain(current_state, trial_state, fact_stats, args)
                if (
                    gain_info["new_fact_covered"] <= 0
                    and gain_info["new_critical_covered"] <= 0
                    and gain_info["new_dependency_closed"] <= 0
                    and gain_info["new_bridge_closed"] <= 0
                    and gain_info["new_anchor_satisfied"] <= 0
                    and gain_info["gain"] < args.assembly_stop_gain
                ):
                    continue

                if hop_count >= 4:
                    priority_rank = (
                        int(gain_info["new_fact_covered"]),
                        int(gain_info["new_dependency_closed"]),
                        int(gain_info["new_critical_covered"]),
                    )
                elif hop_count >= 3:
                    priority_rank = (
                        int(gain_info["new_fact_covered"]),
                        int(gain_info["new_critical_covered"]),
                        int(gain_info["new_dependency_closed"]),
                    )
                else:
                    priority_rank = (
                        int(gain_info["new_fact_covered"]),
                        int(gain_info["new_critical_covered"]),
                        int(gain_info["new_dependency_closed"]),
                    )
                rank = (
                    priority_rank,
                    float(gain_info["gain"]),
                    int(gain_info["new_bridge_closed"]),
                    int(gain_info["new_anchor_satisfied"]),
                    -float(gain_info["new_doc_penalty"]),
                    -float(gain_info["new_title_penalty"]),
                    -float(gain_info["new_redundancy"]),
                    candidate_rank_key(cand),
                )
                if best_rank is None or rank > best_rank:
                    best_sid = sid
                    best_state = trial_state
                    best_rank = rank

        if best_sid is None:
            break

        current_sids.append(best_sid)
        current_sid_set.add(best_sid)
        current_state = best_state

    return current_sids, current_state


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
            "fully_covered_count": 0,
            "dependency_closed": 0,
            "dependency_covered": 0,
            "bridge_closed": 0,
            "anchor_satisfied": 0,
            "cross_doc_bridge_count": 0,
            "doc_penalty": 0.0,
            "utility": 0.0,
        }
        return [], {}, empty_summary

    doc_budget, budget_summary = compute_dynamic_doc_budget(fact_sequence, fact_stats, sentence_pool, args)

    fact_coverage = defaultdict(list)
    selected_sids = []
    selected_sid_set = set()
    selected_docids = set()
    covered_facts = set()
    fact_witnesses = {}
    facts_by_sid = defaultdict(list)

    ordered_facts = sorted(
        fact_sequence,
        key=lambda fact: (fact_stats["depth_map"].get(fact["id"], 1), 0 if fact.get("critical") else 1),
    )

    if isinstance(fact_results, dict):
        result_by_id = fact_results
    else:
        result_by_id = {}
        for fr in fact_results:
            if isinstance(fr, dict) and "fact_id" in fr:
                result_by_id[fr["fact_id"]] = fr

    for fact in ordered_facts:
        fid = fact["id"]
        fact_role = fact.get("role", "verify")
        parents = [pid for pid in fact.get("rely_on", []) if pid in fact_stats["id2fact"]]
        if parents and not all(pid in covered_facts for pid in parents):
            continue

        fr = result_by_id.get(fid) or {}
        summary = fr.get("coverage_summary") or {}
        direct_candidates = _prioritize_assembly_candidates(list(summary.get("direct_candidates") or []), sentence_pool, args)
        bridge_candidates = _prioritize_assembly_candidates(list(summary.get("bridge_candidates") or []), sentence_pool, args)
        parent_support_sids = _collect_parent_support_sids(parents, fact_witnesses)

        chosen_direct = []
        direct_budget = get_fact_direct_winner_budget(fact, fact_role)
        for cand in direct_candidates[:direct_budget]:
            sid = cand["sid"]
            if not _can_add_sid_with_doc_budget(sid, sentence_pool, selected_docids, doc_budget):
                continue
            _add_sid_to_selection(sid, selected_sids, selected_sid_set, selected_docids, sentence_pool)
            chosen_direct.append(cand)

        helper_budget = get_fact_bridge_helper_budget(fact, fact_stats, args) if parents else 0
        chosen_helpers = []
        dependency_ready = not parents
        primary = chosen_direct[0] if chosen_direct else None
        direct_tier_flags = _collect_direct_tier_flags([{"sid": cand["sid"], "support": cand} for cand in chosen_direct])

        if fact_role == "bridge":
            primary = None
            primary_bridge_eval = _zero_bridge_eval()
            for cand in bridge_candidates:
                sid = cand["sid"]
                if not _can_add_sid_with_doc_budget(sid, sentence_pool, selected_docids, doc_budget):
                    continue
                bridge_eval = _zero_bridge_eval() if not parents else score_bridge_against_selected_sids(
                    sid,
                    parent_support_sids,
                    context,
                    semantic_sim_map,
                    args,
                )
                if cand.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                    _add_sid_to_selection(sid, selected_sids, selected_sid_set, selected_docids, sentence_pool)
                    primary = cand
                    primary_bridge_eval = bridge_eval
                    dependency_ready = (not parents) or bool(
                        cand.get("dependency_closure_ready")
                        or bridge_eval["satisfied"]
                        or bridge_eval["score"] >= args.bridge_threshold
                    )
                    break

            if primary is None:
                continue

            if parents and not dependency_ready and helper_budget > 1:
                for cand in bridge_candidates:
                    sid = cand["sid"]
                    if sid == primary["sid"] or sid in {item["sid"] for item in chosen_helpers}:
                        continue
                    if not _can_add_sid_with_doc_budget(sid, sentence_pool, selected_docids, doc_budget):
                        continue
                    bridge_eval = score_bridge_against_selected_sids(
                        sid,
                        parent_support_sids,
                        context,
                        semantic_sim_map,
                        args,
                    )
                    if cand.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                        _add_sid_to_selection(sid, selected_sids, selected_sid_set, selected_docids, sentence_pool)
                        chosen_helpers.append({"candidate": cand, "bridge_eval": bridge_eval})
                        dependency_ready = dependency_ready or bool(
                            cand.get("dependency_closure_ready")
                            or bridge_eval["satisfied"]
                            or bridge_eval["score"] >= args.bridge_threshold
                        )
                        if len(chosen_helpers) >= helper_budget - 1:
                            break

            coverage_status = compute_fact_coverage_status(
                fact=fact,
                fact_role=fact_role,
                has_direct_support=False,
                dependency_closure_ready=dependency_ready,
                has_bridge_support=True,
            )
            direct_sids = []
            helper_sids = [item["candidate"]["sid"] for item in chosen_helpers]
            support_sids = [primary["sid"]] + helper_sids
            bridge_evals = [primary_bridge_eval] + [item["bridge_eval"] for item in chosen_helpers]
            direct_tier_flags = _collect_direct_tier_flags([])
        else:
            if not chosen_direct:
                continue

            for cand in chosen_direct:
                if not parents:
                    dependency_ready = True
                    break
                bridge_eval = score_bridge_against_selected_sids(
                    cand["sid"],
                    parent_support_sids,
                    context,
                    semantic_sim_map,
                    args,
                )
                dependency_ready = dependency_ready or bool(
                    cand.get("dependency_closure_ready")
                    or bridge_eval["satisfied"]
                    or bridge_eval["score"] >= args.bridge_threshold
                )

            if parents and not dependency_ready and helper_budget > 0:
                for cand in bridge_candidates:
                    sid = cand["sid"]
                    if sid in {item["sid"] for item in chosen_direct} or sid in {item["candidate"]["sid"] for item in chosen_helpers}:
                        continue
                    if not _can_add_sid_with_doc_budget(sid, sentence_pool, selected_docids, doc_budget):
                        continue
                    bridge_eval = score_bridge_against_selected_sids(
                        sid,
                        parent_support_sids,
                        context,
                        semantic_sim_map,
                        args,
                    )
                    if cand.get("bridge_support_pass") or bridge_eval["satisfied"] or bridge_eval["score"] >= args.bridge_threshold:
                        _add_sid_to_selection(sid, selected_sids, selected_sid_set, selected_docids, sentence_pool)
                        chosen_helpers.append({"candidate": cand, "bridge_eval": bridge_eval})
                        dependency_ready = dependency_ready or bool(
                            cand.get("dependency_closure_ready")
                            or bridge_eval["satisfied"]
                            or bridge_eval["score"] >= args.bridge_threshold
                        )
                        if len(chosen_helpers) >= helper_budget:
                            break

            coverage_status = compute_fact_coverage_status(
                fact=fact,
                fact_role=fact_role,
                has_direct_support=bool(chosen_direct),
                dependency_closure_ready=dependency_ready,
                has_bridge_support=bool(chosen_helpers),
                has_strong_direct_support=direct_tier_flags["has_strong_direct_support"],
                has_weak_direct_support=direct_tier_flags["has_weak_direct_support"],
                has_bridge_assisted_direct=direct_tier_flags["has_bridge_assisted_direct"],
            )
            direct_sids = [item["sid"] for item in chosen_direct]
            helper_sids = [item["candidate"]["sid"] for item in chosen_helpers]
            support_sids = direct_sids + helper_sids
            bridge_evals = [item["bridge_eval"] for item in chosen_helpers]

        if not coverage_status["covered"]:
            continue

        covered_facts.add(fid)
        fact_witnesses[fid] = {
            "sid": primary["sid"],
            "direct_sids": direct_sids,
            "helper_sids": helper_sids,
            "support_sids": support_sids,
            "covered": bool(coverage_status["covered"]),
            "fully_covered": bool(coverage_status["fully_covered"]),
            "best_direct_support_tier": direct_tier_flags["best_direct_support_tier"],
            "has_strong_direct_support": bool(direct_tier_flags["has_strong_direct_support"]),
            "has_weak_direct_support": bool(direct_tier_flags["has_weak_direct_support"]),
            "has_bridge_assisted_direct": bool(direct_tier_flags["has_bridge_assisted_direct"]),
            "fact_score": float(primary.get("fact_score", 0.0)),
            "direct_support_score": float(primary.get("direct_support_score", 0.0)),
            "bridge_score": float(max(
                [item["candidate"].get("bridge_support_score", 0.0) for item in chosen_helpers]
                + ([primary.get("bridge_support_score", 0.0)] if fact_role == "bridge" else [0.0])
            )),
            "cross_doc_bridge": float(max([bridge_eval.get("cross_doc", 0.0) for bridge_eval in bridge_evals] or [0.0])),
        }
        for sid in support_sids:
            facts_by_sid[sid].append(fid)
            fact_coverage[fid].append(sid)

    selected_sids = [sid for sid in selected_sids if sid in sentence_pool]
    state = evaluate_selected_set(selected_sids, sentence_pool, fact_sequence, fact_stats, context, semantic_sim_map, doc_budget, args)
    initial_selected_sids = list(selected_sids)
    selected_sids, state = rescue_uncovered_facts(
        selected_sids,
        sentence_pool,
        fact_sequence,
        fact_stats,
        context,
        semantic_sim_map,
        doc_budget,
        result_by_id,
        args,
    )
    rescue_added = max(0, len(selected_sids) - len(initial_selected_sids))
    selection_stage = "hard_two_stage_chain_completion" if rescue_added else "hard_two_stage"
    fact_coverage = defaultdict(list)
    for fid, witness in state["fact_witnesses"].items():
        fact_coverage[fid] = list(witness.get("support_sids") or [])

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
                "direct_support_tier": support.get("direct_support_tier", "none"),
                "direct_support_score": float(support.get("direct_support_score", 0.0)),
                "bridge_support_score": float(support.get("bridge_support_score", 0.0)),
                "covered": bool(witness and sid in (witness.get("direct_sids") or [witness.get("sid")])),
                "fully_covered": bool(witness and witness.get("fully_covered", False)),
                "dependency_helper": bool(witness and sid in (witness.get("helper_sids") or [])),
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
            direct_support_tier = max(
                [s.get("direct_support_tier", "none") for s in supports],
                key=direct_support_tier_rank,
            )
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
            direct_support_tier = "none"
            support_type = "bridge_support"

        selected.append({
            "sid": sid,
            "text": item["text"],
            "docid": item.get("docid"),
            "doc_rank": int(item.get("doc_rank", 10**9)),
            "is_title": bool(item.get("is_title", False)),
            "score": float(score),
            "fact_score": float(fact_score),
            "support_type": support_type,
            "direct_support_tier": direct_support_tier,
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
            "selection_stage": selection_stage,
        })

    selected.sort(
        key=lambda x: (
            0 if x.get("is_title") else 1,
            direct_support_tier_rank(x.get("direct_support_tier")),
            1 if x.get("support_type") == "direct_support" else 0,
            float(x.get("direct_support_score", 0.0)),
            float(x.get("fact_score", 0.0)),
            float(x.get("score", 0.0)),
        ),
        reverse=True,
    )

    assembly_summary = dict(budget_summary)
    assembly_summary.update({
        "selected_docids": sorted(state["docids"]),
        "selected_sids": list(selected_sids),
        "covered_facts": sorted(state["covered_facts"], key=lambda fid: fact_stats["depth_map"].get(fid, 1)),
        "critical_covered": int(state["critical_covered"]),
        "fully_covered_count": int(state["fully_covered_count"]),
        "dependency_closed": int(state["dependency_closed"]),
        "dependency_covered": int(state["dependency_covered"]),
        "bridge_closed": int(state["bridge_closed"]),
        "anchor_satisfied": int(state["anchor_satisfied"]),
        "cross_doc_bridge_count": int(state["cross_doc_bridge_count"]),
        "doc_penalty": float(state["doc_penalty"]),
        "title_penalty": float(state["title_penalty"]),
        "utility": float(state["utility"]),
        "redundancy": float(state["redundancy"]),
        "rescue_added": int(rescue_added),
        "completion_added": int(rescue_added),
        "selection_mode": selection_stage,
    })
    return selected, dict(fact_coverage), assembly_summary
