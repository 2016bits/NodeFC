import re

from search_graph_hopaware import get_sim, norm_text

from search_graph_decomposition_aware_modules.shared import (
    CONTEXT_DEPENDENT_STARTS,
    GENERIC_BACKGROUND_PATTERNS,
    allow_relaxed_direct_tiers,
    build_parent_support_summary,
    clamp_score,
    get_direct_support_threshold,
    has_negation,
    is_title_sid,
)


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


def score_bridge_features(sid, target_support, context, semantic_sim_map, args, allow_semantic=True):
    if (
        not target_support["sids"]
        and not target_support["docids"]
        and not target_support["eids"]
        and not target_support["rids"]
        and not target_support.get("numbers")
        and not target_support.get("time_tokens")
        and not target_support.get("quantity_tokens")
    ):
        return {
            "score": 0.0,
            "entity_overlap": 0.0,
            "relation_overlap": 0.0,
            "constraint_overlap": 0.0,
            "same_doc": 0.0,
            "semantic": 0.0,
            "cross_doc": 0.0,
            "satisfied": False,
        }

    sent_eids = context["sid2eids"].get(sid, set())
    sent_rids = context["sid2rids"].get(sid, set())
    sent_numbers = context["sid2numbers"].get(sid, set())
    sent_time_tokens = context["sid2time_tokens"].get(sid, set())
    sent_quantity_tokens = context["sid2quantity_tokens"].get(sid, set())
    docid = context["sid2meta"].get(sid, {}).get("docid")
    is_title = is_title_sid(context, sid)

    entity_overlap = 0.0 if not target_support["eids"] else len(sent_eids & target_support["eids"]) / max(1, len(target_support["eids"]))
    relation_overlap = 0.0 if not target_support["rids"] else len(sent_rids & target_support["rids"]) / max(1, len(target_support["rids"]))
    same_doc = 1.0 if docid and docid in target_support["docids"] else 0.0
    constraint_parts = []
    if target_support.get("numbers"):
        constraint_parts.append(len(sent_numbers & target_support["numbers"]) / max(1, len(target_support["numbers"])))
    if target_support.get("time_tokens"):
        constraint_parts.append(len(sent_time_tokens & target_support["time_tokens"]) / max(1, len(target_support["time_tokens"])))
    if target_support.get("quantity_tokens"):
        constraint_parts.append(len(sent_quantity_tokens & target_support["quantity_tokens"]) / max(1, len(target_support["quantity_tokens"])))
    constraint_overlap = sum(constraint_parts) / len(constraint_parts) if constraint_parts else 0.0

    semantic = 0.0
    if allow_semantic:
        for target_sid in target_support["sids"]:
            semantic = max(semantic, get_sim(semantic_sim_map, sid, target_sid))

    has_entity_bridge = bool(sent_eids & target_support["eids"])
    has_relation_bridge = bool(sent_rids & target_support["rids"])
    constraint_ready = constraint_overlap >= args.bridge_constraint_threshold
    cross_doc = 1.0 if docid and target_support["docids"] and docid not in target_support["docids"] and (
        has_entity_bridge
        or has_relation_bridge
        or constraint_ready
        or (allow_semantic and semantic >= args.bridge_semantic_threshold)
    ) else 0.0
    score = (
        0.44 * entity_overlap
        + 0.20 * relation_overlap
        + 0.14 * constraint_overlap
        + 0.10 * same_doc
        + 0.06 * semantic
        + 0.06 * cross_doc
    )
    if is_title:
        score = max(0.0, score - args.title_bridge_penalty)
    satisfied = bool(
        has_entity_bridge
        or has_relation_bridge
        or constraint_ready
        or (allow_semantic and semantic >= args.bridge_semantic_threshold)
        or cross_doc
        or (same_doc and not is_title)
    )
    return {
        "score": float(score),
        "entity_overlap": float(entity_overlap),
        "relation_overlap": float(relation_overlap),
        "constraint_overlap": float(constraint_overlap),
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


def score_fact_completeness_penalty(candidate_features, fact_role, args):
    penalty = 0.0

    entity_pair = float(candidate_features.get("entity_pair_score", 0.0))
    relation_match = float(candidate_features.get("relation_match_score", 0.0))
    binding_satisfied = bool(candidate_features.get("binding_satisfied", False))
    context_independence = float(candidate_features.get("context_independence", 0.0))

    if entity_pair < args.penalty_entity_pair_floor:
        penalty += args.penalty_entity_pair_weight * (args.penalty_entity_pair_floor - entity_pair)
    if relation_match <= 0.0:
        penalty += args.penalty_relation_zero_weight
    if not binding_satisfied:
        penalty += args.penalty_binding_unsatisfied_weight
    if context_independence < args.penalty_context_independent_floor:
        penalty += args.penalty_context_independent_weight * (
            args.penalty_context_independent_floor - context_independence
        )

    if fact_role == "verify":
        penalty *= args.verify_penalty_boost
    elif fact_role == "anchor":
        penalty *= args.anchor_penalty_boost
    return float(max(0.0, penalty))


def compute_direct_support_tier(
    fact,
    profile,
    fact_role,
    cand,
    relation_match,
    entity_pair,
    target_match,
    keyword_overlap,
    constraint_consistency,
    negation_compatibility,
    context_independence,
    binding,
    direct_support_score,
    bridge_support_pass,
    dependency_closure_ready,
    args,
):
    threshold = get_direct_support_threshold(fact_role, args)
    relation_or_keyword_ready = (
        relation_match > 0.0
        or keyword_overlap >= args.min_keyword_overlap_for_direct_fallback
    )
    if bool(
        direct_support_score >= threshold
        and entity_pair >= args.min_entity_pair_for_direct
        and relation_or_keyword_ready
    ):
        return "strong"

    if not allow_relaxed_direct_tiers(fact_role, fact):
        return "none"

    weak_threshold = max(0.0, threshold - args.weak_direct_support_margin)
    weak_relation_or_keyword_ready = (
        relation_match >= args.min_relation_match_for_weak_direct
        or keyword_overlap >= args.min_keyword_overlap_for_weak_direct
    )
    if bool(
        direct_support_score >= weak_threshold
        and entity_pair >= args.min_entity_pair_for_weak_direct
        and weak_relation_or_keyword_ready
        and context_independence >= args.min_context_independence_for_weak_direct
        and negation_compatibility >= args.min_negation_compat_for_direct
    ):
        return "weak"

    bridge_assisted_threshold = max(0.0, threshold - args.bridge_assisted_direct_margin)
    bridge_assisted_ready = (
        bool(fact.get("rely_on"))
        and (
            bridge_support_pass
            or dependency_closure_ready
            or binding["score"] >= args.binding_threshold
        )
    )
    if bool(
        bridge_assisted_ready
        and direct_support_score >= bridge_assisted_threshold
        and entity_pair >= args.min_entity_pair_for_bridge_assisted_direct
        and relation_or_keyword_ready
        and context_independence >= args.min_context_independence_for_bridge_assisted_direct
    ):
        return "bridge_assisted"
    return "none"


def compute_direct_support_pass(
    fact,
    profile,
    fact_role,
    cand,
    relation_match,
    entity_pair,
    target_match,
    keyword_overlap,
    constraint_consistency,
    negation_compatibility,
    context_independence,
    binding,
    direct_support_score,
    bridge_support_pass,
    dependency_closure_ready,
    args,
):
    return compute_direct_support_tier(
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
    ) != "none"
