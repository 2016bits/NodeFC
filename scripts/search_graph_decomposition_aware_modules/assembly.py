from collections import defaultdict

from search_graph_hopaware import get_sim

from search_graph_decomposition_aware_modules.scoring import score_bridge_features
from search_graph_decomposition_aware_modules.shared import (
    candidate_rank_key,
    collect_support_from_sids,
    compute_fact_coverage_status,
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


def _merge_sentence_pool_candidate(sentence_pool, fact_id, cand):
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
        "fact_completeness_penalty": float(cand.get("fact_completeness_penalty", 0.0)),
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


def _zero_bridge_eval():
    return {
        "score": 0.0,
        "same_doc": 0.0,
        "entity_overlap": 0.0,
        "relation_overlap": 0.0,
        "semantic": 0.0,
        "cross_doc": 0.0,
        "satisfied": False,
    }


def _sort_direct_support_items(items):
    return sorted(
        items,
        key=lambda x: (
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
    fact_witnesses = {}
    facts_by_sid = defaultdict(list)
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
            )
            direct_sids = [item["sid"] for item in direct_records]
            support_sids = direct_sids + [item["sid"] for item in helper_records]

        if not coverage_status["covered"]:
            continue

        covered_facts.add(fid)
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
        if fact_role == "bridge":
            bridge_evals = [primary_bridge_eval] + bridge_evals

        if parents and coverage_status["fully_covered"]:
            dependency_covered += 1
            if any(bridge_eval.get("cross_doc", 0.0) > 0 for bridge_eval in bridge_evals):
                cross_doc_bridge_count += 1

        fact_witnesses[fid] = {
            "sid": primary_sid,
            "direct_sids": direct_sids,
            "helper_sids": [item["sid"] for item in helper_records],
            "support_sids": support_sids,
            "covered": bool(coverage_status["covered"]),
            "fully_covered": bool(coverage_status["fully_covered"]),
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
        "fully_covered_facts": fully_covered_facts,
        "fact_witnesses": fact_witnesses,
        "facts_by_sid": {sid: fact_ids for sid, fact_ids in facts_by_sid.items()},
        "critical_covered": int(critical_covered),
        "fully_covered_count": int(len(fully_covered_facts)),
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
            "fully_covered_count": 0,
            "dependency_covered": 0,
            "cross_doc_bridge_count": 0,
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
        direct_candidates = list(summary.get("direct_candidates") or [])
        bridge_candidates = list(summary.get("bridge_candidates") or [])
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
            "selection_stage": "hard_two_stage",
        })

    selected.sort(key=lambda x: (1 if x.get("support_type") == "direct_support" else 0, float(x.get("direct_support_score", 0.0)), float(x.get("fact_score", 0.0)), float(x.get("score", 0.0))), reverse=True)

    assembly_summary = dict(budget_summary)
    assembly_summary.update({
        "selected_docids": sorted(state["docids"]),
        "selected_sids": list(selected_sids),
        "covered_facts": sorted(state["covered_facts"], key=lambda fid: fact_stats["depth_map"].get(fid, 1)),
        "critical_covered": int(state["critical_covered"]),
        "fully_covered_count": int(state["fully_covered_count"]),
        "dependency_covered": int(state["dependency_covered"]),
        "cross_doc_bridge_count": int(state["cross_doc_bridge_count"]),
        "redundancy": float(state["redundancy"]),
        "selection_mode": "hard_two_stage",
    })
    return selected, dict(fact_coverage), assembly_summary
