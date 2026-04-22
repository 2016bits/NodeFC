from collections import defaultdict

import numpy as np

from search_graph_hopaware import make_personalization, norm_text, ppr

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


def seed_candidate_is_clean(candidate, args):
    if candidate.get("is_title"):
        return False

    entity_overlap = float(candidate.get("entity_pair_score", 0.0))
    relation_match = float(candidate.get("relation_match_score", 0.0))
    constraint_match = max(0.0, float(candidate.get("time_quantity_consistency", 0.0)))
    binding_score = float(candidate.get("binding_score", 0.0))
    direct_support_score = float(candidate.get("direct_support_score", 0.0))

    shared_signal = (
        entity_overlap >= args.seed_min_entity_overlap
        or constraint_match >= args.seed_min_constraint_match
        or (relation_match >= args.seed_min_relation_match and binding_score >= args.seed_min_binding_score)
    )
    if candidate.get("direct_support_pass"):
        return shared_signal and direct_support_score >= args.seed_min_direct_support_score
    if not candidate.get("bridge_support_pass"):
        return False
    return shared_signal and binding_score >= args.seed_min_binding_score


def select_support_seed_candidates(direct_candidates, bridge_candidates, args):
    filtered_direct = [cand for cand in direct_candidates if seed_candidate_is_clean(cand, args)]
    filtered_bridge = [cand for cand in bridge_candidates if seed_candidate_is_clean(cand, args)]

    if filtered_direct:
        return filtered_direct[:args.max_support_seed_candidates]
    if filtered_bridge:
        return filtered_bridge[:args.max_bridge_seed_candidates]

    fallback_direct = [cand for cand in direct_candidates if not cand.get("is_title")]
    if fallback_direct:
        return fallback_direct[:1]
    fallback_bridge = [cand for cand in bridge_candidates if not cand.get("is_title")]
    if fallback_bridge:
        return fallback_bridge[:1]
    if direct_candidates:
        return direct_candidates[:1]
    return bridge_candidates[:1]


def select_chain_seed_candidates(scored_candidates, args):
    ordered = sorted(scored_candidates or [], key=candidate_rank_key, reverse=True)
    direct = [cand for cand in ordered if cand.get("direct_support_pass") and seed_candidate_is_clean(cand, args)]
    if direct:
        return direct[:args.chain_seed_k]
    bridge = [cand for cand in ordered if cand.get("bridge_support_pass") and seed_candidate_is_clean(cand, args)]
    if bridge:
        return bridge[:min(args.chain_seed_k, args.max_bridge_seed_candidates)]
    fallback = [cand for cand in ordered if not cand.get("is_title")]
    if fallback:
        return fallback[:1]
    return ordered[:1]


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
            "coverage_gate_pass": bool(coverage_gate_pass),
            "fact_completeness_penalty": float(completeness_penalty),
            "is_title": bool(cand.get("is_title", False)),
            "title_score_penalty": float(title_score_penalty),
        })
        scored.append(item)

    scored.sort(key=candidate_rank_key, reverse=True)
    return scored


def build_fact_coverage_summary(fact, fact_role, parent_results, scored_candidates, args):
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
    support_seed_candidates = select_support_seed_candidates(direct_winners, bridge_winners, args)

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
        "support_seed_candidates": support_seed_candidates,
    }


def coverage_insufficient(fact, fact_role, parent_results, scored_candidates, args):
    summary = build_fact_coverage_summary(fact, fact_role, parent_results, scored_candidates, args)
    return bool(
        not summary["fully_covered"]
        or summary["needs_fact_completion"]
        or summary["needs_dependency_completion"]
        or summary["needs_cross_doc_bridge_completion"]
        or summary["needs_anchor_completion"]
    )


def build_chain_bridge_sentence_map(binding_requirements, parent_results, context, args):
    scores = defaultdict(float)
    parent_summary = build_parent_support_summary(parent_results)
    for sid in context["sid_list"]:
        binding = score_binding_coverage(binding_requirements, {"sid": sid}, context)
        bridge = score_bridge_features(sid, parent_summary, context, context["semantic_sim_map"], args)
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

    for cand in select_chain_seed_candidates(scored_candidates, args):
        if cand["bridge_score"] >= args.bridge_threshold or cand["binding_score"] >= args.binding_threshold or cand["fact_score"] >= args.fact_score_threshold:
            seed_weight = 0.5 * max(cand["fact_score"], cand["bridge_score"], cand["binding_score"])
            if cand.get("is_title"):
                seed_weight = max(0.0, seed_weight - args.title_bridge_penalty)
            expanded_s[cand["sid"]] += seed_weight
            for eid in context["sid2eids"].get(cand["sid"], set()):
                expanded_n[eid] += 0.25 * seed_weight
            for rid in context["sid2rids"].get(cand["sid"], set()):
                expanded_r[rid] += 0.25 * seed_weight

    if fact.get("critical"):
        for cand in select_chain_seed_candidates(scored_candidates, args):
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
    cross_doc_bias = topk_normalize({
        sid: 1.0
        for sid in context["sid_list"]
        if context["sid2meta"].get(sid, {}).get("docid")
        and context["sid2meta"][sid]["docid"] not in parent_summary["docids"]
    }, topk)
    combined = merge_score_maps(
        (args.bridge_repair_lexical_weight, bridge_lexical),
        (args.bridge_repair_dense_weight, bridge_dense),
        (args.ppr_expand_weight, ppr_scores),
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
        args=args,
    )
    if fact_role == "anchor":
        local_candidates = filter_anchor_candidates(local_candidates, profile, context, args)

    rerank_query_text = build_fact_rerank_query_text(fact, profile, fact_role, parent_results, context)
    reranked = rerank_cross_encoder(crossencoder, rerank_query_text, local_candidates)
    scored = enrich_fact_candidates(fact, profile, fact_role, reranked, parent_results, context, args, critical_bonus)
    coverage_summary = build_fact_coverage_summary(fact, fact_role, parent_results, scored, args)

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
        coverage_summary = build_fact_coverage_summary(fact, fact_role, parent_results, scored, args)

    support_seed_candidates = coverage_summary["support_seed_candidates"] if coverage_summary["support_seed_candidates"] else scored
    support_profile_k = max(
        args.parent_support_k,
        len(coverage_summary.get("direct_winners") or []),
        len(coverage_summary.get("bridge_winners") or []),
    )
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
        "candidates": scored[:args.per_fact_output_k],
    }
