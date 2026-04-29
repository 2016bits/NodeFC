from collections import defaultdict

import numpy as np

from search_graph_hopaware import get_sim, make_personalization, norm_text, ppr

from search_graph_decomposition_aware_modules.scoring import (
    compute_direct_support_tier,
    score_background_penalty,
    score_binding_coverage,
    score_bridge_features,
    score_context_independence,
    score_entity_pair_presence,
    score_fact_completeness_penalty,
    score_keyword_overlap,
    score_negation_compatibility,
    score_relation_expression,
    score_target_match,
    score_time_quantity_consistency,
    score_upstream_bridge,
)
from search_graph_decomposition_aware_modules.shared import (
    build_constraint_entry,
    build_dependency_seed_maps,
    build_fact_profile,
    build_parent_support_summary,
    build_support_profile,
    candidate_rank_key,
    clamp_score,
    compute_fact_coverage_status,
    derive_binding_requirements,
    get_fact_direct_winner_budget,
    get_direct_support_threshold,
    get_role_candidate_budget,
    infer_fact_role,
    is_title_candidate,
    merge_score_maps,
    normalize_sentence_node_scores,
    requires_direct_support,
    rerank_cross_encoder,
    select_sentence_candidates,
    semantic_entry_from_bank,
    top_score_items,
    topk_normalize,
)


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
            score = 0.32 * bridge["score"] + 0.30 * binding["score"] + 0.18 * same_parent_doc + 0.12 * relation_match + 0.08 * keyword_overlap
            if fact_role == "bridge":
                score += 0.12 * bridge["entity_overlap"] + 0.08 * bridge["relation_overlap"] + 0.06 * bridge["constraint_overlap"]
        else:
            score = 0.24 * entity_pair + 0.18 * relation_match + 0.18 * keyword_overlap + 0.16 * target_match + 0.10 * constraint_consistency + 0.04 * negation_compatibility + 0.10 * binding["score"]
            if fact_role == "verify":
                score += 0.08 * entity_pair + 0.06 * keyword_overlap
            elif fact_role == "bridge":
                score += 0.10 * bridge["score"] + 0.06 * same_parent_doc
            elif fact_role == "anchor":
                score += 0.14 * constraint_consistency

        score -= candidate_title_penalty({"sid": sid}, context, args, "title_recall_penalty")
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


def build_fact_rerank_query_text(fact, profile, fact_role, parent_results, context):
    if fact_role == "bridge" or parent_results:
        return build_bridge_repair_query_text(fact, profile, parent_results, context)
    return build_direct_repair_query_text(fact, profile, context)


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


def candidate_title_penalty(candidate, context, args, field_name):
    if not is_title_candidate(candidate, context):
        return 0.0
    return float(getattr(args, field_name, 0.0) or 0.0)


def _seed_filter_targets(fact, profile, parent_results):
    binding_requirements = derive_binding_requirements(fact, profile, parent_results)
    core_eids = list(binding_requirements.get("eids") or [])
    if not core_eids:
        core_eids = [eid for eid, _ in top_score_items(profile.get("entry_n") or {}, 3)]

    core_rids = list(binding_requirements.get("rids") or [])
    if not core_rids:
        core_rids = [rid for rid, _ in top_score_items(profile.get("entry_r") or {}, 3)]

    return {
        "core_eids": set(core_eids),
        "entity_keywords": set(profile.get("entity_surface_keywords") or set()),
        "core_rids": set(core_rids),
        "relation_keywords": set(profile.get("relation_keywords") or set()),
    }


def _seed_has_core_entity(candidate, targets, context):
    sid = candidate["sid"]
    sent_eids = context["sid2eids"].get(sid, set())
    sent_keywords = context["sid2keywords"].get(sid, set())
    return bool(
        (targets["core_eids"] and sent_eids & targets["core_eids"])
        or (targets["entity_keywords"] and sent_keywords & targets["entity_keywords"])
    )


def _seed_has_relation_clue(candidate, targets, context, args):
    sid = candidate["sid"]
    sent_rids = context["sid2rids"].get(sid, set())
    sent_keywords = context["sid2keywords"].get(sid, set())
    relation_match = float(candidate.get("relation_match_score", 0.0))
    relation_keyword_hit = bool(targets["relation_keywords"] and sent_keywords & targets["relation_keywords"])
    relation_id_hit = bool(targets["core_rids"] and sent_rids & targets["core_rids"])
    return bool(relation_id_hit or relation_keyword_hit or relation_match >= args.seed_min_relation_match)


def primary_seed_candidate_is_clean(candidate, args):
    if candidate.get("is_title"):
        return False
    if candidate.get("direct_support_tier") != "strong":
        return False

    return float(candidate.get("direct_support_score", 0.0)) >= args.seed_min_direct_support_score


def secondary_seed_candidate_is_clean(candidate, fact, profile, parent_results, context, args):
    if candidate.get("is_title"):
        return False

    direct_tier = candidate.get("direct_support_tier")
    if direct_tier not in {"weak", "bridge_assisted"} and not candidate.get("bridge_support_pass"):
        return False

    entity_overlap = float(candidate.get("entity_pair_score", 0.0))
    binding_score = float(candidate.get("binding_score", 0.0))
    bridge_support_score = float(candidate.get("bridge_support_score", 0.0))
    targets = _seed_filter_targets(fact, profile, parent_results)

    if entity_overlap < args.seed_min_entity_overlap:
        return False
    if not _seed_has_core_entity(candidate, targets, context):
        return False
    if not _seed_has_relation_clue(candidate, targets, context, args):
        return False
    return bool(
        binding_score >= args.seed_min_binding_score
        or bridge_support_score >= args.bridge_threshold
    )


def _annotate_seed_candidate(candidate, seed_tier, seed_weight):
    item = dict(candidate)
    item["seed_tier"] = seed_tier
    item["seed_weight"] = float(seed_weight)
    return item


def select_support_seed_candidates(fact, profile, parent_results, direct_candidates, bridge_candidates, context, args):
    primary = [
        _annotate_seed_candidate(cand, "primary", args.primary_seed_weight)
        for cand in direct_candidates
        if primary_seed_candidate_is_clean(cand, args)
    ][:args.max_support_seed_candidates]

    secondary = [
        _annotate_seed_candidate(cand, "secondary", args.secondary_seed_weight)
        for cand in direct_candidates
        if secondary_seed_candidate_is_clean(cand, fact, profile, parent_results, context, args)
    ][:args.max_bridge_seed_candidates]

    if not secondary:
        secondary = [
            _annotate_seed_candidate(cand, "secondary", args.secondary_seed_weight)
            for cand in bridge_candidates
            if secondary_seed_candidate_is_clean(cand, fact, profile, parent_results, context, args)
        ][:args.max_bridge_seed_candidates]

    if primary or secondary:
        return primary, secondary

    fallback_direct = [
        _annotate_seed_candidate(cand, "primary", max(0.75, args.secondary_seed_weight))
        for cand in direct_candidates
        if not cand.get("is_title")
    ]
    if fallback_direct:
        return fallback_direct[:1], []

    fallback_bridge = [
        _annotate_seed_candidate(cand, "secondary", args.secondary_seed_weight)
        for cand in bridge_candidates
        if not cand.get("is_title")
    ]
    if fallback_bridge:
        return [], fallback_bridge[:1]

    if direct_candidates:
        return [_annotate_seed_candidate(direct_candidates[0], "primary", max(0.75, args.secondary_seed_weight))], []
    if bridge_candidates:
        return [], [_annotate_seed_candidate(bridge_candidates[0], "secondary", args.secondary_seed_weight)]
    return [], []


def select_chain_seed_candidates(fact, profile, parent_results, scored_candidates, context, args):
    ordered = sorted(scored_candidates or [], key=candidate_rank_key, reverse=True)
    direct = [cand for cand in ordered if cand.get("direct_support_pass")]
    bridge = [cand for cand in ordered if cand.get("bridge_support_pass")]
    primary, secondary = select_support_seed_candidates(
        fact,
        profile,
        parent_results,
        direct,
        bridge,
        context,
        args,
    )
    return primary[:args.chain_seed_k], secondary[:args.max_bridge_seed_candidates]


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


def compute_fact_match_score(
    fact_role,
    target_match,
    entity_pair,
    relation_match,
    keyword_overlap,
    constraint_consistency,
    negation_compatibility,
):
    if fact_role == "anchor":
        weights = {
            "target": 0.22,
            "entity": 0.20,
            "relation": 0.12,
            "keyword": 0.12,
            "constraint": 0.24,
            "negation": 0.10,
        }
    elif fact_role == "bridge":
        weights = {
            "target": 0.20,
            "entity": 0.24,
            "relation": 0.18,
            "keyword": 0.12,
            "constraint": 0.10,
            "negation": 0.06,
        }
    else:
        weights = {
            "target": 0.24,
            "entity": 0.24,
            "relation": 0.16,
            "keyword": 0.14,
            "constraint": 0.12,
            "negation": 0.10,
        }
    return clamp_score(
        weights["target"] * target_match
        + weights["entity"] * entity_pair
        + weights["relation"] * relation_match
        + weights["keyword"] * keyword_overlap
        + weights["constraint"] * max(0.0, constraint_consistency)
        + weights["negation"] * max(0.0, negation_compatibility)
    )


def compute_bridge_potential_score(bridge, binding, bridge_support_score, dependency_norm, ppr_norm, dependency_closure_ready):
    return clamp_score(
        0.42 * max(0.0, bridge_support_score)
        + 0.24 * max(0.0, bridge.get("score", 0.0))
        + 0.18 * max(0.0, binding.get("score", 0.0))
        + 0.08 * max(0.0, dependency_norm)
        + 0.04 * max(0.0, ppr_norm)
        + 0.04 * float(bool(dependency_closure_ready))
    )


def compute_uncovered_fact_gain(
    fact,
    fact_role,
    candidate_coverage,
    strong_direct_support_pass,
    weak_direct_support_pass,
    bridge_assisted_direct_pass,
    bridge_support_pass,
    fact_match_score,
    bridge_potential_score,
):
    gain = 0.12 * fact_match_score + 0.10 * bridge_potential_score
    if strong_direct_support_pass:
        gain = max(gain, 1.0)
    elif weak_direct_support_pass:
        gain = max(gain, 0.84)
    elif bridge_assisted_direct_pass:
        gain = max(gain, 0.78 if candidate_coverage.get("closure_ready") else 0.70)
    elif bridge_support_pass:
        gain = max(gain, 0.60 if candidate_coverage.get("closure_ready") else 0.48)

    if candidate_coverage.get("fully_covered"):
        gain += 0.08
    elif candidate_coverage.get("covered"):
        gain += 0.04
    if fact.get("critical"):
        gain += 0.08
    if fact_role == "bridge":
        gain += 0.06 * bridge_potential_score
    return clamp_score(gain)


def _compute_local_redundancy_scores(scored_candidates, context, args):
    if not scored_candidates:
        return {}

    semantic_sim_map = (context or {}).get("semantic_sim_map") or {}
    ordered = sorted(scored_candidates, key=candidate_rank_key, reverse=True)
    previous = []
    redundancy_scores = {}
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
        redundancy_scores[cand["sid"]] = clamp_score(
            semantic_penalty + args.rerank_same_doc_redundancy_penalty * same_doc,
            lower=0.0,
            upper=1.0,
        )
        previous.append(cand)
    return redundancy_scores


def _candidate_keep_reason_priority(candidate):
    reasons = set(candidate.get("rerank_keep_reasons") or [])
    if "critical_supportive" in reasons:
        return 4
    if "bypass" in reasons:
        return 3
    if "direct_bucket" in reasons:
        return 2
    if "bridge_bucket" in reasons:
        return 1
    return 0


def _fact_bridge_bucket_sort_key(candidate):
    return (
        1 if candidate.get("bridge_assisted_closure_rescue") else 0,
        float(candidate.get("bridge_potential_score", candidate.get("bridge_support_score", 0.0))),
        float(candidate.get("rerank_score", candidate.get("aggregate_score", 0.0))),
        float(candidate.get("binding_score", 0.0)),
        float(candidate.get("bridge_support_score", 0.0)),
        candidate_rank_key(candidate),
    )


def _fact_critical_support_sort_key(candidate):
    return (
        1 if candidate.get("direct_support_pass") else 0,
        float(candidate.get("direct_support_score", 0.0)),
        float(candidate.get("fact_score", 0.0)),
        float(candidate.get("uncovered_fact_gain", 0.0)),
        float(candidate.get("bridge_support_score", 0.0)),
        float(candidate.get("rerank_score", candidate.get("aggregate_score", 0.0))),
        candidate_rank_key(candidate),
    )


def _add_bucket_candidates(selected_map, candidates, limit, reason):
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
            if reason == "critical_supportive":
                item["critical_supportive_candidate"] = True
            selected_map[sid] = item
            kept += 1
        else:
            reasons = list(existing.get("rerank_keep_reasons") or [])
            if reason not in reasons:
                reasons.append(reason)
                existing["rerank_keep_reasons"] = reasons
            if reason == "critical_supportive":
                existing["critical_supportive_candidate"] = True
        if kept >= limit:
            break


def select_fact_preserved_candidates(fact, fact_role, scored_candidates, coverage_summary, args):
    if not scored_candidates:
        return []

    limit = min(len(scored_candidates), max(1, int(args.per_fact_output_k)))
    ordered = sorted(scored_candidates, key=candidate_rank_key, reverse=True)
    selected_map = {}

    if fact.get("critical"):
        critical_supportive = sorted(
            [
                cand for cand in ordered
                if cand.get("direct_support_pass")
                or cand.get("bridge_support_pass")
                or cand.get("rerank_bypass_pass")
            ],
            key=_fact_critical_support_sort_key,
            reverse=True,
        )
        _add_bucket_candidates(selected_map, critical_supportive, args.rerank_keep_critical_per_fact, "critical_supportive")

    bypass_candidates = sorted(
        [cand for cand in ordered if cand.get("rerank_bypass_pass")],
        key=candidate_rank_key,
        reverse=True,
    )
    _add_bucket_candidates(selected_map, bypass_candidates, args.rerank_keep_bypass_per_fact, "bypass")

    direct_candidates = sorted(
        list((coverage_summary or {}).get("direct_candidates") or [cand for cand in ordered if cand.get("direct_support_pass")]),
        key=candidate_rank_key,
        reverse=True,
    )
    _add_bucket_candidates(selected_map, direct_candidates, args.rerank_keep_direct_per_fact, "direct_bucket")

    if fact_role == "bridge" or fact.get("rely_on"):
        bridge_candidates = sorted(
            [
                cand for cand in ((coverage_summary or {}).get("bridge_candidates") or ordered)
                if cand.get("bridge_support_pass")
                or cand.get("bridge_assisted_closure_rescue")
                or cand.get("dependency_closure_ready")
            ],
            key=_fact_bridge_bucket_sort_key,
            reverse=True,
        )
        _add_bucket_candidates(selected_map, bridge_candidates, args.rerank_keep_bridge_per_fact, "bridge_bucket")

    protected = sorted(
        selected_map.values(),
        key=lambda cand: (_candidate_keep_reason_priority(cand), candidate_rank_key(cand)),
        reverse=True,
    )
    if len(protected) > limit:
        protected = protected[:limit]
        selected_map = {cand["sid"]: cand for cand in protected}

    for cand in ordered:
        if len(selected_map) >= limit:
            break
        if cand["sid"] in selected_map:
            continue
        item = dict(cand)
        item["rerank_keep_reasons"] = list(item.get("rerank_keep_reasons") or [])
        if not item["rerank_keep_reasons"]:
            item["rerank_keep_reasons"].append("score_pool")
        selected_map[item["sid"]] = item

    final_candidates = sorted(selected_map.values(), key=candidate_rank_key, reverse=True)
    return final_candidates[:limit]


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
        title_score_penalty = candidate_title_penalty(cand, context, args, "title_score_penalty")
        title_bridge_penalty = candidate_title_penalty(cand, context, args, "title_bridge_penalty")

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
            - title_score_penalty
        )
        bridge_support_score = clamp_score(
            0.44 * max(0.0, bridge["score"])
            + 0.30 * binding["score"]
            + 0.18 * dependency_norm
            + 0.08 * ppr_norm
            - title_bridge_penalty
        )

        bridge_support_pass = bridge_support_score >= args.bridge_threshold or bridge["satisfied"] or binding["score"] >= args.binding_threshold
        dependency_closure_ready = (
            not fact.get("rely_on")
            or binding["direct_hit"]
            or binding["score"] >= args.binding_threshold
            or bridge["satisfied"]
            or bridge["score"] >= args.bridge_threshold
        )
        direct_support_tier = compute_direct_support_tier(
            fact=fact,
            profile=profile,
            fact_role=fact_role,
            cand=cand,
            relation_match=relation_match,
            entity_pair=entity_pair,
            target_match=target_match,
            keyword_overlap=keyword_overlap,
            constraint_consistency=constraint_consistency,
            negation_compatibility=negation_compatibility,
            context_independence=context_independence,
            binding=binding,
            direct_support_score=direct_support_score,
            bridge_support_pass=bridge_support_pass,
            dependency_closure_ready=dependency_closure_ready,
            args=args,
        )
        direct_support_pass = direct_support_tier != "none"
        strong_direct_support_pass = direct_support_tier == "strong"
        weak_direct_support_pass = direct_support_tier == "weak"
        bridge_assisted_direct_pass = direct_support_tier == "bridge_assisted"

        binding_satisfied = bool(binding["direct_hit"] or binding["score"] >= args.binding_threshold)
        completeness_penalty = score_fact_completeness_penalty({
            "entity_pair_score": entity_pair,
            "relation_match_score": relation_match,
            "binding_satisfied": binding_satisfied,
            "context_independence": context_independence,
        }, fact_role, args)

        support_type = "direct_support" if direct_support_pass else ("bridge_support" if bridge_support_pass else "candidate")

        fact_score = clamp_score(
            0.62 * direct_support_score
            + 0.16 * max(0.0, constraint_consistency)
            + 0.10 * context_independence
            + 0.12 * max(0.0, negation_compatibility)
            - args.fact_completeness_penalty_weight * completeness_penalty
            - 0.55 * title_score_penalty
        )
        aggregate_score = (
            1.55 * direct_support_score
            + 0.60 * ce_norm
            + 0.22 * lexical_norm
            + 0.18 * dense_norm
            + (0.30 if fact_role == "anchor" else 0.18) * max(0.0, constraint_consistency)
            + 0.12 * max(0.0, negation_compatibility)
            + 0.08 * context_independence
            + weights["bridge_bonus"] * bridge_support_score
            + 0.05 * ppr_norm
            + args.doc_rank_weight * doc_rank_bonus
            + critical_bonus
            - 0.20 * background_penalty
            - args.fact_completeness_penalty_weight * completeness_penalty
            - 1.15 * title_score_penalty
        )
        coverage_score = 0.74 * direct_support_score + 0.16 * bridge_support_score + 0.10 * max(0.0, constraint_consistency)

        if fact_role == "verify":
            if direct_support_tier == "none":
                aggregate_score -= args.verify_no_direct_support_margin
                fact_score = max(0.0, fact_score - 0.5 * args.verify_no_direct_support_margin)
            elif direct_support_tier == "bridge_assisted":
                aggregate_score -= 0.35 * args.verify_no_direct_support_margin
            elif direct_support_tier == "weak":
                aggregate_score -= 0.15 * args.verify_no_direct_support_margin

        candidate_coverage = compute_fact_coverage_status(
            fact=fact,
            fact_role=fact_role,
            has_direct_support=bool(direct_support_pass),
            dependency_closure_ready=bool(parents_covered and dependency_closure_ready),
            has_bridge_support=bool(bridge_support_pass),
            has_strong_direct_support=bool(strong_direct_support_pass),
            has_weak_direct_support=bool(weak_direct_support_pass),
            has_bridge_assisted_direct=bool(bridge_assisted_direct_pass),
        )
        coverage_gate_pass = bool(parents_covered and candidate_coverage["fully_covered"])
        fact_match_score = compute_fact_match_score(
            fact_role=fact_role,
            target_match=target_match,
            entity_pair=entity_pair,
            relation_match=relation_match,
            keyword_overlap=keyword_overlap,
            constraint_consistency=constraint_consistency,
            negation_compatibility=negation_compatibility,
        )
        bridge_potential_score = compute_bridge_potential_score(
            bridge=bridge,
            binding=binding,
            bridge_support_score=bridge_support_score,
            dependency_norm=dependency_norm,
            ppr_norm=ppr_norm,
            dependency_closure_ready=dependency_closure_ready,
        )
        uncovered_fact_gain = compute_uncovered_fact_gain(
            fact=fact,
            fact_role=fact_role,
            candidate_coverage=candidate_coverage,
            strong_direct_support_pass=strong_direct_support_pass,
            weak_direct_support_pass=weak_direct_support_pass,
            bridge_assisted_direct_pass=bridge_assisted_direct_pass,
            bridge_support_pass=bridge_support_pass,
            fact_match_score=fact_match_score,
            bridge_potential_score=bridge_potential_score,
        )
        weak_direct_rescue = bool(weak_direct_support_pass)
        bridge_assisted_closure_rescue = bool(
            bridge_assisted_direct_pass
            and parents_covered
            and dependency_closure_ready
        )

        item = dict(cand)
        item.update({
            "fact_id": fact["id"],
            "fact_role": fact_role,
            "ce_norm": float(ce_norm),
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
            "binding_satisfied": bool(binding_satisfied),
            "bridge_score": float(bridge["score"]),
            "bridge_satisfied": bool(bridge["satisfied"] or bridge["score"] >= args.bridge_threshold),
            "bridge_support_score": float(bridge_support_score),
            "bridge_support_pass": bool(bridge_support_pass),
            "direct_support_score": float(direct_support_score),
            "direct_support_pass": bool(direct_support_pass),
            "direct_support_tier": direct_support_tier,
            "strong_direct_support_pass": bool(strong_direct_support_pass),
            "weak_direct_support_pass": bool(weak_direct_support_pass),
            "bridge_assisted_direct_pass": bool(bridge_assisted_direct_pass),
            "direct_threshold": float(direct_threshold),
            "dependency_closure_ready": bool(dependency_closure_ready),
            "support_type": support_type,
            "cross_doc_bridge_score": float(bridge["cross_doc"]),
            "coverage_score": float(coverage_score),
            "aggregate_score": float(aggregate_score),
            "fact_match_score": float(fact_match_score),
            "bridge_potential_score": float(bridge_potential_score),
            "uncovered_fact_gain": float(uncovered_fact_gain),
            "coverage_gate_pass": bool(coverage_gate_pass),
            "fact_completeness_penalty": float(completeness_penalty),
            "is_title": bool(cand.get("is_title", False)),
            "title_score_penalty": float(title_score_penalty),
            "redundancy_penalty": 0.0,
            "rerank_score": float(aggregate_score),
            "weak_direct_rescue": bool(weak_direct_rescue),
            "bridge_assisted_closure_rescue": bool(bridge_assisted_closure_rescue),
            "rerank_bypass_pass": bool(weak_direct_rescue or bridge_assisted_closure_rescue),
            "critical_supportive_candidate": False,
            "rerank_keep_reasons": [],
        })
        scored.append(item)

    redundancy_scores = _compute_local_redundancy_scores(scored, context, args)
    for item in scored:
        redundancy_penalty = float(redundancy_scores.get(item["sid"], 0.0))
        item["redundancy_penalty"] = redundancy_penalty
        item["rerank_score"] = float(
            args.rerank_weight_ce * float(item.get("ce_norm", 0.0))
            + args.rerank_weight_fact_match * float(item.get("fact_match_score", 0.0))
            + args.rerank_weight_direct_support * float(item.get("direct_support_score", 0.0))
            + args.rerank_weight_bridge_potential * float(item.get("bridge_potential_score", 0.0))
            + args.rerank_weight_uncovered_fact_gain * float(item.get("uncovered_fact_gain", 0.0))
            - args.rerank_weight_redundancy * redundancy_penalty
        )

    scored.sort(key=candidate_rank_key, reverse=True)
    return scored


def build_fact_coverage_summary(fact, profile, fact_role, parent_results, scored_candidates, context, args):
    parent_covered = all((parent.get("coverage_summary") or {}).get("covered", False) for parent in parent_results)
    parent_summary = build_parent_support_summary(parent_results)
    direct_candidates = sorted([cand for cand in scored_candidates if cand.get("direct_support_pass")], key=candidate_rank_key, reverse=True)
    strong_direct_candidates = [cand for cand in direct_candidates if cand.get("direct_support_tier") == "strong"]
    weak_direct_candidates = [cand for cand in direct_candidates if cand.get("direct_support_tier") == "weak"]
    bridge_assisted_direct_candidates = [cand for cand in direct_candidates if cand.get("direct_support_tier") == "bridge_assisted"]
    bridge_candidates = sorted([cand for cand in scored_candidates if cand.get("bridge_support_pass")], key=candidate_rank_key, reverse=True)
    has_direct_support = bool(direct_candidates)
    has_strong_direct_support = bool(strong_direct_candidates)
    has_weak_direct_support = bool(weak_direct_candidates)
    has_bridge_assisted_direct = bool(bridge_assisted_direct_candidates)
    has_bridge_support = bool(bridge_candidates)
    dependency_closure = (not fact.get("rely_on")) or any(
        cand.get("dependency_closure_ready") for cand in (direct_candidates + bridge_candidates)
    )
    anchor_constraint_ready = fact_role != "anchor" or any(
        cand.get("time_quantity_consistency", 0.0) >= args.min_constraint_consistency_for_anchor
        for cand in direct_candidates
    )
    cross_doc_bridge_ready = bool(fact.get("rely_on")) and any(
        cand.get("cross_doc_bridge_score", 0.0) > 0
        and (
            cand.get("bridge_support_pass")
            or cand.get("dependency_closure_ready")
            or cand.get("direct_support_pass")
        )
        for cand in (direct_candidates + bridge_candidates)
    )
    coverage_status = compute_fact_coverage_status(
        fact=fact,
        fact_role=fact_role,
        has_direct_support=has_direct_support,
        dependency_closure_ready=bool(parent_covered and dependency_closure),
        has_bridge_support=has_bridge_support,
        has_strong_direct_support=has_strong_direct_support,
        has_weak_direct_support=has_weak_direct_support,
        has_bridge_assisted_direct=has_bridge_assisted_direct,
    )

    best_candidate = scored_candidates[0] if scored_candidates else None
    best_direct = direct_candidates[0] if direct_candidates else None
    best_bridge = bridge_candidates[0] if bridge_candidates else None

    direct_winner_budget = get_fact_direct_winner_budget(fact, fact_role)
    direct_winners = direct_candidates[:direct_winner_budget]
    bridge_winners = bridge_candidates[:1] if fact_role == "bridge" else []
    primary_seed_candidates, secondary_seed_candidates = select_support_seed_candidates(
        fact,
        profile,
        parent_results,
        direct_candidates,
        bridge_candidates,
        context,
        args,
    )
    support_seed_candidates = primary_seed_candidates + secondary_seed_candidates

    covered = bool(parent_covered and coverage_status["covered"])
    fully_covered = bool(parent_covered and coverage_status["fully_covered"])
    needs_fact_completion = bool(
        (not covered and not coverage_status["support_ready"])
        or (coverage_status["requires_direct_support"] and not has_direct_support)
        or (fact.get("critical") and not covered)
    )
    needs_critical_fact_completion = bool(fact.get("critical") and not covered)
    needs_dependency_completion = bool(fact.get("rely_on")) and bool(parent_covered) and not bool(dependency_closure)
    needs_cross_doc_bridge_completion = bool(fact.get("rely_on")) and bool(parent_covered) and not bool(cross_doc_bridge_ready) and bool(
        not fully_covered or len(parent_summary["docids"]) > 1
    )
    needs_anchor_completion = bool(fact_role == "anchor" and not anchor_constraint_ready)

    return {
        "covered": covered,
        "fully_covered": fully_covered,
        "parent_covered": bool(parent_covered),
        "requires_direct_support": bool(coverage_status["requires_direct_support"]),
        "relaxed_direct_allowed": bool(coverage_status["relaxed_direct_allowed"]),
        "support_ready": bool(coverage_status["support_ready"]),
        "closure_ready": bool(coverage_status["closure_ready"]),
        "has_direct_support": bool(has_direct_support),
        "has_strong_direct_support": bool(has_strong_direct_support),
        "has_weak_direct_support": bool(has_weak_direct_support),
        "has_bridge_assisted_direct": bool(has_bridge_assisted_direct),
        "best_direct_support_tier": (best_direct or {}).get("direct_support_tier", "none"),
        "has_bridge_support": bool(has_bridge_support),
        "dependency_closure": bool(dependency_closure),
        "anchor_constraint_ready": bool(anchor_constraint_ready),
        "cross_doc_bridge_ready": bool(cross_doc_bridge_ready),
        "needs_fact_completion": bool(needs_fact_completion),
        "needs_critical_fact_completion": bool(needs_critical_fact_completion),
        "needs_dependency_completion": bool(needs_dependency_completion),
        "needs_cross_doc_bridge_completion": bool(needs_cross_doc_bridge_completion),
        "needs_anchor_completion": bool(needs_anchor_completion),
        "needs_direct_repair": bool(needs_fact_completion),
        "needs_bridge_repair": bool(needs_dependency_completion or needs_cross_doc_bridge_completion),
        "top_fact_score": float(best_candidate["fact_score"]) if best_candidate else 0.0,
        "top_direct_support_score": float(best_direct["direct_support_score"]) if best_direct else 0.0,
        "top_bridge_support_score": float(best_bridge["bridge_support_score"]) if best_bridge else 0.0,
        "top_direct_support_tier": (best_direct or {}).get("direct_support_tier", "none"),
        "num_coverage_candidates": len(direct_candidates if coverage_status["requires_direct_support"] else bridge_candidates),
        "num_direct_candidates": len(direct_candidates),
        "num_strong_direct_candidates": len(strong_direct_candidates),
        "num_weak_direct_candidates": len(weak_direct_candidates),
        "num_bridge_assisted_direct_candidates": len(bridge_assisted_direct_candidates),
        "num_bridge_candidates": len(bridge_candidates),
        "best_direct_sid": best_direct["sid"] if best_direct else None,
        "best_bridge_sid": best_bridge["sid"] if best_bridge else None,
        "direct_candidates": direct_candidates,
        "strong_direct_candidates": strong_direct_candidates,
        "weak_direct_candidates": weak_direct_candidates,
        "bridge_assisted_direct_candidates": bridge_assisted_direct_candidates,
        "bridge_candidates": bridge_candidates,
        "direct_winners": direct_winners,
        "bridge_winners": bridge_winners,
        "primary_seed_candidates": primary_seed_candidates,
        "secondary_seed_candidates": secondary_seed_candidates,
        "support_seed_candidates": support_seed_candidates,
    }


def coverage_insufficient(fact, profile, fact_role, parent_results, scored_candidates, context, args):
    summary = build_fact_coverage_summary(fact, profile, fact_role, parent_results, scored_candidates, context, args)
    return bool(
        not summary["fully_covered"]
        or summary["needs_fact_completion"]
        or summary["needs_dependency_completion"]
        or summary["needs_cross_doc_bridge_completion"]
        or summary["needs_anchor_completion"]
    )


def has_clear_structural_bridge_constraints(fact, profile, parent_results):
    binding_requirements = derive_binding_requirements(fact, profile, parent_results)
    entity_ready = bool(binding_requirements["eids"] or profile["entry_n"])
    relation_ready = bool(binding_requirements["rids"] or profile["entry_r"] or profile.get("relation_keywords"))
    return bool(entity_ready and relation_ready)


def build_chain_bridge_sentence_map(binding_requirements, parent_results, context, args, allow_semantic=False):
    scores = defaultdict(float)
    parent_summary = build_parent_support_summary(parent_results)
    for sid in context["sid_list"]:
        binding = score_binding_coverage(binding_requirements, {"sid": sid}, context)
        bridge = score_bridge_features(
            sid,
            parent_summary,
            context,
            context["semantic_sim_map"],
            args,
            allow_semantic=allow_semantic,
        )
        score = 0.38 * binding["score"] + 0.34 * bridge["score"] + 0.18 * bridge["constraint_overlap"] + 0.10 * bridge["cross_doc"]
        if score > 0:
            scores[sid] = score
    return topk_normalize(scores, args.chain_seed_k)


def build_chain_completion_entries(
    fact,
    profile,
    base_entry_s,
    base_entry_n,
    base_entry_r,
    dep_entry_s,
    dep_entry_n,
    dep_entry_r,
    scored_candidates,
    claim_entry_s,
    claim_entry_n,
    parent_results,
    context,
    args,
    structure_focus=False,
):
    expanded_s = defaultdict(float, base_entry_s)
    expanded_n = defaultdict(float, base_entry_n)
    expanded_r = defaultdict(float, base_entry_r)
    binding_requirements = derive_binding_requirements(fact, profile, parent_results)

    for sid, score in (dep_entry_s or {}).items():
        expanded_s[sid] += args.chain_parent_sentence_weight * float(score)
    for eid, score in (dep_entry_n or {}).items():
        expanded_n[eid] += args.chain_parent_neighbor_weight * float(score)
    for rid, score in (dep_entry_r or {}).items():
        expanded_r[rid] += args.chain_parent_neighbor_weight * float(score)

    for sid, score in build_chain_bridge_sentence_map(
        binding_requirements,
        parent_results,
        context,
        args,
        allow_semantic=not structure_focus,
    ).items():
        expanded_s[sid] += args.chain_binding_sentence_weight * float(score)

    for eid in binding_requirements["eids"]:
        expanded_n[eid] += args.chain_binding_anchor_weight
    for rid in binding_requirements["rids"]:
        expanded_r[rid] += args.chain_binding_anchor_weight

    primary_seed_candidates, secondary_seed_candidates = select_chain_seed_candidates(
        fact,
        profile,
        parent_results,
        scored_candidates,
        context,
        args,
    )
    chain_seed_candidates = primary_seed_candidates + secondary_seed_candidates

    for cand in primary_seed_candidates:
        if cand["bridge_score"] >= args.bridge_threshold or cand["binding_score"] >= args.binding_threshold or cand["fact_score"] >= args.fact_score_threshold:
            seed_weight = args.chain_primary_seed_weight * float(cand.get("seed_weight", 1.0)) * max(
                cand["fact_score"],
                cand["direct_support_score"],
                cand["binding_score"],
            )
            if cand.get("is_title"):
                seed_weight = max(0.0, seed_weight - args.title_bridge_penalty)
            expanded_s[cand["sid"]] += seed_weight
            for eid in context["sid2eids"].get(cand["sid"], set()):
                expanded_n[eid] += 0.25 * seed_weight
            for rid in context["sid2rids"].get(cand["sid"], set()):
                expanded_r[rid] += 0.25 * seed_weight

    for cand in secondary_seed_candidates:
        if cand["bridge_score"] >= args.bridge_threshold or cand["binding_score"] >= args.binding_threshold or cand["fact_score"] >= args.fact_score_threshold:
            seed_weight = args.chain_secondary_seed_weight * float(cand.get("seed_weight", 1.0)) * max(
                cand["fact_score"],
                cand["bridge_score"],
                cand["binding_score"],
            )
            if cand.get("is_title"):
                seed_weight = max(0.0, seed_weight - args.title_bridge_penalty)
            expanded_s[cand["sid"]] += seed_weight
            for eid in context["sid2eids"].get(cand["sid"], set()):
                expanded_n[eid] += 0.25 * seed_weight
            for rid in context["sid2rids"].get(cand["sid"], set()):
                expanded_r[rid] += 0.25 * seed_weight

    if fact.get("critical"):
        for cand in chain_seed_candidates:
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


def _graph_bundle_variant(graph, variant):
    if isinstance(graph, dict) and any(key in graph for key in ("structural", "local", "full")):
        return graph.get(variant) or graph.get("full") or graph.get("local") or graph.get("structural") or {}
    return graph or {}


def _count_stage_hits(score_map, min_relative_score):
    return sum(1 for score in (score_map or {}).values() if float(score) >= min_relative_score)


def build_bridge_stage_ppr_scores(graph, personalization, topk, budget, structure_focus, args):
    structural_graph = _graph_bundle_variant(graph, "structural")
    local_graph = _graph_bundle_variant(graph, "local")
    full_graph = _graph_bundle_variant(graph, "full")

    structural_scores = normalize_sentence_node_scores(
        ppr(structural_graph, personalization, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter),
        topk,
    )
    min_hits = min(max(1, budget), max(1, int(args.bridge_stage_min_hits)))
    structural_ready = _count_stage_hits(structural_scores, args.bridge_stage_min_relative_score) >= min_hits

    local_scores = {}
    if not structure_focus or not structural_ready:
        local_scores = normalize_sentence_node_scores(
            ppr(local_graph, personalization, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter),
            topk,
        )
    local_ready = _count_stage_hits(local_scores, args.bridge_stage_min_relative_score) >= min_hits

    semantic_scores = {}
    if not structure_focus or (not structural_ready and not local_ready):
        semantic_scores = normalize_sentence_node_scores(
            ppr(full_graph, personalization, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter),
            topk,
        )

    combined_scores = topk_normalize(
        merge_score_maps(
            (args.ppr_structural_expand_weight, structural_scores),
            (args.ppr_local_expand_weight if local_scores else 0.0, local_scores),
            (args.ppr_semantic_expand_weight if semantic_scores else 0.0, semantic_scores),
        ),
        topk,
    )
    return combined_scores, {
        "structural": structural_scores,
        "local": local_scores,
        "semantic": semantic_scores,
        "structural_ready": bool(structural_ready),
        "local_ready": bool(local_ready),
    }


def build_targeted_fact_completion_candidates(
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
        args=args,
    )
    if fact_role == "anchor":
        candidates = filter_anchor_candidates(candidates, profile, context, args)
    return candidates


def build_targeted_dependency_completion_candidates(
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
    structure_focus = has_clear_structural_bridge_constraints(fact, profile, parent_results)
    bridge_lexical = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="bridge", topk=topk)
    query_text = build_bridge_repair_query_text(fact, profile, parent_results, context)
    bridge_dense = topk_normalize(semantic_entry_from_bank(biencoder, query_text, sentence_bank, topk=topk), topk)

    dep_entry_s, dep_entry_n, dep_entry_r = build_dependency_seed_maps(parent_results)
    exp_s, exp_n, exp_r = build_chain_completion_entries(
        fact,
        profile,
        bridge_lexical,
        merge_score_maps((1.0, profile["entry_n"]), (args.dependency_seed_weight, dep_entry_n)),
        merge_score_maps((1.0, profile["entry_r"]), (args.dependency_seed_weight, dep_entry_r)),
        dep_entry_s,
        dep_entry_n,
        dep_entry_r,
        scored_candidates,
        claim_entry_s,
        claim_entry_n,
        parent_results,
        context,
        args,
        structure_focus=structure_focus,
    )
    exp_personalization = make_personalization(exp_s, exp_n, exp_r, w_s=args.w_entry_s, w_n=args.w_entry_n, w_r=args.w_entry_r)
    ppr_scores, _ppr_debug = build_bridge_stage_ppr_scores(
        graph,
        exp_personalization,
        topk,
        budget,
        structure_focus,
        args,
    )
    combined = merge_score_maps(
        (args.bridge_repair_lexical_weight, bridge_lexical),
        (args.bridge_repair_dense_weight, bridge_dense),
        (1.0, ppr_scores),
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
        args=args,
    )
    if fact_role == "anchor":
        candidates = filter_anchor_candidates(candidates, profile, context, args)
    return candidates


def build_targeted_cross_doc_bridge_candidates(
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

    parent_summary = build_parent_support_summary(parent_results)
    budget = args.bridge_repair_candidate_k
    topk = max(budget * 2, args.chain_seed_k)
    structure_focus = has_clear_structural_bridge_constraints(fact, profile, parent_results)
    bridge_lexical = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="bridge", topk=topk)
    query_text = build_bridge_repair_query_text(fact, profile, parent_results, context)
    bridge_dense = topk_normalize(semantic_entry_from_bank(biencoder, query_text, sentence_bank, topk=topk), topk)

    dep_entry_s, dep_entry_n, dep_entry_r = build_dependency_seed_maps(parent_results)
    exp_s, exp_n, exp_r = build_chain_completion_entries(
        fact,
        profile,
        bridge_lexical,
        merge_score_maps((1.0, profile["entry_n"]), (args.dependency_seed_weight, dep_entry_n)),
        merge_score_maps((1.0, profile["entry_r"]), (args.dependency_seed_weight, dep_entry_r)),
        dep_entry_s,
        dep_entry_n,
        dep_entry_r,
        scored_candidates,
        claim_entry_s,
        claim_entry_n,
        parent_results,
        context,
        args,
        structure_focus=structure_focus,
    )
    exp_personalization = make_personalization(exp_s, exp_n, exp_r, w_s=args.w_entry_s, w_n=args.w_entry_n, w_r=args.w_entry_r)
    ppr_scores, _ppr_debug = build_bridge_stage_ppr_scores(
        graph,
        exp_personalization,
        topk,
        budget,
        structure_focus,
        args,
    )
    cross_doc_bias = topk_normalize({
        sid: 1.0
        for sid in context["sid_list"]
        if context["sid2meta"].get(sid, {}).get("docid")
        and context["sid2meta"][sid]["docid"] not in parent_summary["docids"]
    }, topk)
    combined = merge_score_maps(
        (args.bridge_repair_lexical_weight, bridge_lexical),
        (args.bridge_repair_dense_weight, bridge_dense),
        (1.0, ppr_scores),
        (args.cross_doc_completion_weight, cross_doc_bias),
    )
    candidates = select_sentence_candidates(
        context,
        combined,
        topk=budget * 2,
        component_maps={
            "dense_score": bridge_dense,
            "lexical_score": bridge_lexical,
            "constraint_score": {},
            "dependency_score": bridge_lexical,
            "ppr_score": ppr_scores,
        },
        args=args,
    )
    cross_doc_candidates = [
        cand for cand in candidates
        if cand.get("docid") and cand["docid"] not in parent_summary["docids"]
    ]
    return cross_doc_candidates[:budget] if cross_doc_candidates else candidates[:budget]


def build_targeted_anchor_completion_candidates(
    fact,
    profile,
    fact_role,
    context,
    sentence_bank,
    biencoder,
    parent_results,
    args,
):
    budget = max(args.anchor_candidate_k, args.direct_repair_candidate_k)
    topk = max(budget * 2, args.constraint_k)
    anchor_lexical = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="direct", topk=topk)
    query_text = build_direct_repair_query_text(fact, profile, context)
    anchor_dense = topk_normalize(semantic_entry_from_bank(biencoder, query_text, sentence_bank, topk=topk), topk)
    constraint_entry = build_constraint_entry(profile, biencoder, sentence_bank, topk=topk)
    combined = merge_score_maps(
        (args.direct_repair_lexical_weight, anchor_lexical),
        (args.direct_repair_dense_weight, anchor_dense),
        (args.anchor_completion_weight, constraint_entry),
    )
    candidates = select_sentence_candidates(
        context,
        combined,
        topk=budget,
        component_maps={
            "dense_score": anchor_dense,
            "lexical_score": anchor_lexical,
            "constraint_score": constraint_entry,
            "dependency_score": {},
            "ppr_score": {},
        },
        args=args,
    )
    return filter_anchor_candidates(candidates, profile, context, args)


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
                for field in (
                    "aggregate_score",
                    "rerank_score",
                    "fact_score",
                    "coverage_score",
                    "direct_support_score",
                    "bridge_support_score",
                    "semantic_relevance",
                    "fact_match_score",
                    "bridge_potential_score",
                    "uncovered_fact_gain",
                    "ce_score",
                    "ce_norm",
                ):
                    prev[field] = max(float(prev.get(field, 0.0)), float(cand.get(field, 0.0)))
                for field in (
                    "direct_support_pass",
                    "strong_direct_support_pass",
                    "weak_direct_support_pass",
                    "bridge_assisted_direct_pass",
                    "bridge_support_pass",
                    "dependency_closure_ready",
                    "weak_direct_rescue",
                    "bridge_assisted_closure_rescue",
                    "rerank_bypass_pass",
                    "critical_supportive_candidate",
                ):
                    prev[field] = bool(prev.get(field, False) or cand.get(field, False))
                prev["direct_support_tier"] = max(
                    [prev.get("direct_support_tier", "none"), cand.get("direct_support_tier", "none")],
                    key=lambda tier: {"none": 0, "bridge_assisted": 1, "weak": 2, "strong": 3}.get(tier, 0),
                )
                if cand.get("support_type") == "direct_support" or prev.get("support_type") != "direct_support":
                    prev["support_type"] = cand.get("support_type", prev.get("support_type"))
                reasons = list(prev.get("rerank_keep_reasons") or [])
                for reason in cand.get("rerank_keep_reasons") or []:
                    if reason not in reasons:
                        reasons.append(reason)
                prev["rerank_keep_reasons"] = reasons
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

    depth = fact_stats["depth_map"].get(fact["id"], 1)
    entry_k_s = args.critical_fact_k if critical else args.fact_k
    candidate_k = get_role_candidate_budget(fact_role, critical, args)
    if critical or depth >= args.multi_bridge_depth_threshold:
        candidate_k = max(candidate_k, args.expanded_candidate_k)
    elif (
        fact_role == "bridge"
        or depth >= 3
        or len(fact.get("rely_on", [])) > 1
        or fact_stats["max_depth"] >= args.multi_bridge_depth_threshold
    ):
        candidate_k = max(candidate_k, args.fact_candidate_k)
    entry_k_s = max(entry_k_s, min(candidate_k, args.expanded_candidate_k))

    base_entry_s = topk_normalize(semantic_entry_from_bank(biencoder, fact["text"], sentence_bank, topk=entry_k_s), entry_k_s)
    lexical_entry_s = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="base", topk=max(entry_k_s, candidate_k * 2))
    constraint_entry_s = build_constraint_entry(profile, biencoder, sentence_bank, topk=max(args.constraint_k, candidate_k))
    dependency_entry_s = build_sentence_recall_entry(fact, profile, fact_role, parent_results, context, args, mode="bridge", topk=max(args.constraint_k, candidate_k)) if parent_results or fact_role == "bridge" else {}

    _dep_entry_s, dep_entry_n, dep_entry_r = build_dependency_seed_maps(parent_results)
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
        args=args,
    )
    if fact_role == "anchor":
        local_candidates = filter_anchor_candidates(local_candidates, profile, context, args)

    rerank_query_text = build_fact_rerank_query_text(fact, profile, fact_role, parent_results, context)
    reranked = rerank_cross_encoder(crossencoder, rerank_query_text, local_candidates)
    scored = enrich_fact_candidates(fact, profile, fact_role, reranked, parent_results, context, args, critical_bonus)
    coverage_summary = build_fact_coverage_summary(fact, profile, fact_role, parent_results, scored, context, args)

    expanded = False
    completion_scored = []
    completion_steps = []
    if coverage_summary["needs_critical_fact_completion"] or coverage_summary["needs_fact_completion"]:
        expanded = True
        completion_steps.append("critical_fact" if coverage_summary["needs_critical_fact_completion"] else "fact_completion")
        fact_completion_candidates = build_targeted_fact_completion_candidates(
            fact,
            profile,
            fact_role,
            context,
            sentence_bank,
            biencoder,
            parent_results,
            args,
        )
        if fact_completion_candidates:
            fact_reranked = rerank_cross_encoder(
                crossencoder,
                build_direct_repair_query_text(fact, profile, context),
                fact_completion_candidates,
            )
            completion_scored.extend(enrich_fact_candidates(fact, profile, fact_role, fact_reranked, parent_results, context, args, critical_bonus))

    if coverage_summary["needs_dependency_completion"]:
        expanded = True
        completion_steps.append("dependency_closure")
        dependency_completion_candidates = build_targeted_dependency_completion_candidates(
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
        if dependency_completion_candidates:
            dependency_reranked = rerank_cross_encoder(
                crossencoder,
                build_bridge_repair_query_text(fact, profile, parent_results, context),
                dependency_completion_candidates,
            )
            completion_scored.extend(enrich_fact_candidates(fact, profile, fact_role, dependency_reranked, parent_results, context, args, critical_bonus))

    if coverage_summary["needs_cross_doc_bridge_completion"]:
        expanded = True
        completion_steps.append("cross_doc_bridge")
        cross_doc_candidates = build_targeted_cross_doc_bridge_candidates(
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
        if cross_doc_candidates:
            cross_doc_reranked = rerank_cross_encoder(
                crossencoder,
                build_bridge_repair_query_text(fact, profile, parent_results, context),
                cross_doc_candidates,
            )
            completion_scored.extend(enrich_fact_candidates(fact, profile, fact_role, cross_doc_reranked, parent_results, context, args, critical_bonus))

    if coverage_summary["needs_anchor_completion"]:
        expanded = True
        completion_steps.append("anchor_constraint")
        anchor_completion_candidates = build_targeted_anchor_completion_candidates(
            fact,
            profile,
            fact_role,
            context,
            sentence_bank,
            biencoder,
            parent_results,
            args,
        )
        if anchor_completion_candidates:
            anchor_reranked = rerank_cross_encoder(
                crossencoder,
                build_direct_repair_query_text(fact, profile, context),
                anchor_completion_candidates,
            )
            completion_scored.extend(enrich_fact_candidates(fact, profile, fact_role, anchor_reranked, parent_results, context, args, critical_bonus))

    if completion_scored:
        scored = merge_candidate_lists(scored, completion_scored)
        coverage_summary = build_fact_coverage_summary(fact, profile, fact_role, parent_results, scored, context, args)

    support_seed_candidates = coverage_summary["support_seed_candidates"] if coverage_summary["support_seed_candidates"] else scored
    support_profile_k = max(
        args.parent_support_k,
        len(coverage_summary.get("direct_winners") or []),
        len(coverage_summary.get("bridge_winners") or []),
        len(coverage_summary.get("primary_seed_candidates") or []),
        len(coverage_summary.get("secondary_seed_candidates") or []),
    )
    preserved_candidates = select_fact_preserved_candidates(fact, fact_role, scored, coverage_summary, args)
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
        "completion_steps": completion_steps,
        "support_profile": build_support_profile(support_seed_candidates, context, max_candidates=support_profile_k),
        "candidates": preserved_candidates,
    }
