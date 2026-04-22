import argparse
import json

from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from search_graph_hopaware import build_hetero_graph, build_semantic_sim_map, entity_entry_n
from search_graph_decomposition_aware_modules.assembly import (
    aggregate_entry_ids,
    aggregate_top_evidences,
    build_global_candidate_view,
)
from search_graph_decomposition_aware_modules.retrieval import retrieve_one_fact
from search_graph_decomposition_aware_modules.shared import (
    build_example_context,
    build_fact_graph_stats,
    encode_sentence_bank,
    load_spacy_model,
    resolve_device,
    semantic_entry_from_bank,
    topological_sort_facts,
)

def main(args):
    with open(args.decomposition_path.replace('[PLAN]', args.plan), "r", encoding="utf-8") as f:
        decomposed_data = json.load(f)
    with open(args.nodes_path.replace("[SPLIT]", args.split).replace('[PLAN]', args.plan), "r", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(args.edges_path.replace("[SPLIT]", args.split).replace('[PLAN]', args.plan), "r", encoding="utf-8") as f:
        edges = json.load(f)
    with open(args.semantic_edges_path.replace("[SPLIT]", args.split).replace('[PLAN]', args.plan), "r", encoding="utf-8") as f:
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
            w_ss_entity_bonus=args.w_ss_entity_bonus,
            w_ss_relation_bonus=args.w_ss_relation_bonus,
            w_ss_number_bonus=args.w_ss_number_bonus,
            w_ss_title_penalty=args.w_ss_title_penalty,
        )

        claim = sample["claim"]
        claim_entry_s = semantic_entry_from_bank(biencoder, claim, sentence_bank, topk=args.claim_entry_k)
        claim_entry_n = entity_entry_n(nlp, claim, context["entity_nodes"])

        fact_sequence = topological_sort_facts((sample.get("decomposition") or {}).get("atomic_facts", []))
        fact_stats = build_fact_graph_stats(fact_sequence)
        fact_stats["claim_num_hops"] = int(sample.get("num_hops") or 0)
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
                "has_strong_direct_support": bool((fact_result.get("coverage_summary") or {}).get("has_strong_direct_support", False)),
                "has_weak_direct_support": bool((fact_result.get("coverage_summary") or {}).get("has_weak_direct_support", False)),
                "has_bridge_assisted_direct": bool((fact_result.get("coverage_summary") or {}).get("has_bridge_assisted_direct", False)),
                "best_direct_support_tier": (fact_result.get("coverage_summary") or {}).get("best_direct_support_tier", "none"),
                "dependency_closure": bool((fact_result.get("coverage_summary") or {}).get("dependency_closure", False)),
                "top_fact_score": float((fact_result.get("coverage_summary") or {}).get("top_fact_score", 0.0)),
                "top_direct_support_score": float((fact_result.get("coverage_summary") or {}).get("top_direct_support_score", 0.0)),
                "completion_steps": list(fact_result.get("completion_steps") or []),
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

    out_path = args.out_path.replace("[PLAN]", args.plan).replace("[SPLIT]", args.split).replace('[PLAN]', args.plan)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {out_path}")
    if missing_ids:
        print(f"Skipped {missing_ids} examples missing graph data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition_path", type=str, default="./data/[PLAN]/dev_2_decomposed_0_4000.json")
    parser.add_argument("--nodes_path", type=str, default="./data/[PLAN]/bm25_nodes_[SPLIT].json")
    parser.add_argument("--edges_path", type=str, default="./data/[PLAN]/bm25_edges_[SPLIT].json")
    parser.add_argument("--semantic_edges_path", type=str, default="./data/[PLAN]/bm25_semantic_edges_[SPLIT].json")
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
    parser.add_argument("--w_ss_entity_bonus", type=float, default=0.20)
    parser.add_argument("--w_ss_relation_bonus", type=float, default=0.14)
    parser.add_argument("--w_ss_number_bonus", type=float, default=0.08)
    parser.add_argument("--w_ss_title_penalty", type=float, default=0.45)
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
    parser.add_argument("--bridge_constraint_threshold", type=float, default=0.30)
    parser.add_argument("--direct_support_threshold", type=float, default=0.58)
    parser.add_argument("--verify_direct_support_threshold", type=float, default=0.62)
    parser.add_argument("--anchor_direct_support_threshold", type=float, default=0.60)
    parser.add_argument("--bridge_direct_support_threshold", type=float, default=0.52)
    parser.add_argument("--weak_direct_support_margin", type=float, default=0.08)
    parser.add_argument("--bridge_assisted_direct_margin", type=float, default=0.16)
    parser.add_argument("--min_direct_relation_score", type=float, default=0.16)
    parser.add_argument("--min_entity_pair_for_direct", type=float, default=0.45)
    parser.add_argument("--min_entity_pair_for_weak_direct", type=float, default=0.32)
    parser.add_argument("--min_entity_pair_for_bridge_assisted_direct", type=float, default=0.24)
    parser.add_argument("--min_relation_match_for_direct", type=float, default=0.16)
    parser.add_argument("--min_relation_match_for_weak_direct", type=float, default=0.08)
    parser.add_argument("--min_keyword_overlap_for_direct_fallback", type=float, default=0.05)
    parser.add_argument("--min_keyword_overlap_for_weak_direct", type=float, default=0.03)
    parser.add_argument("--min_negation_compat_for_direct", type=float, default=0.00)
    parser.add_argument("--min_context_independence_for_direct", type=float, default=0.18)
    parser.add_argument("--min_context_independence_for_weak_direct", type=float, default=0.12)
    parser.add_argument("--min_context_independence_for_bridge_assisted_direct", type=float, default=0.08)
    parser.add_argument("--min_constraint_consistency_for_anchor", type=float, default=0.20)
    parser.add_argument("--min_keyword_overlap_for_critical_direct", type=float, default=0.05)
    parser.add_argument("--fact_completeness_penalty_weight", type=float, default=0.18)
    parser.add_argument("--penalty_entity_pair_floor", type=float, default=0.45)
    parser.add_argument("--penalty_entity_pair_weight", type=float, default=0.30)
    parser.add_argument("--penalty_relation_zero_weight", type=float, default=0.35)
    parser.add_argument("--penalty_binding_unsatisfied_weight", type=float, default=0.25)
    parser.add_argument("--penalty_context_independent_floor", type=float, default=0.40)
    parser.add_argument("--penalty_context_independent_weight", type=float, default=0.20)
    parser.add_argument("--verify_penalty_boost", type=float, default=1.20)
    parser.add_argument("--anchor_penalty_boost", type=float, default=1.05)
    parser.add_argument("--verify_no_direct_support_margin", type=float, default=0.40)
    parser.add_argument("--anchor_prefilter_threshold", type=float, default=0.00)
    parser.add_argument("--default_min_per_fact", type=int, default=1)
    parser.add_argument("--critical_min_per_fact", type=int, default=2)
    parser.add_argument("--parent_support_k", type=int, default=2)
    parser.add_argument("--max_bridge_per_fact", type=int, default=2)
    parser.add_argument("--max_bridge_per_complex_fact", type=int, default=3)
    parser.add_argument("--multi_bridge_depth_threshold", type=int, default=4)
    parser.add_argument("--redundancy_threshold", type=float, default=0.88)
    parser.add_argument("--seed_min_entity_overlap", type=float, default=0.25)
    parser.add_argument("--seed_min_relation_match", type=float, default=0.10)
    parser.add_argument("--seed_min_constraint_match", type=float, default=0.10)
    parser.add_argument("--seed_min_binding_score", type=float, default=0.25)
    parser.add_argument("--seed_min_direct_support_score", type=float, default=0.55)
    parser.add_argument("--max_support_seed_candidates", type=int, default=2)
    parser.add_argument("--max_bridge_seed_candidates", type=int, default=1)
    parser.add_argument("--title_recall_penalty", type=float, default=0.12)
    parser.add_argument("--title_candidate_recall_penalty", type=float, default=0.08)
    parser.add_argument("--title_score_penalty", type=float, default=0.18)
    parser.add_argument("--title_bridge_penalty", type=float, default=0.20)

    parser.add_argument("--chain_seed_k", type=int, default=8)
    parser.add_argument("--chain_parent_sentence_weight", type=float, default=0.55)
    parser.add_argument("--chain_parent_neighbor_weight", type=float, default=0.35)
    parser.add_argument("--chain_binding_sentence_weight", type=float, default=0.50)
    parser.add_argument("--chain_binding_anchor_weight", type=float, default=0.65)
    parser.add_argument("--chain_critical_seed_weight", type=float, default=0.45)
    parser.add_argument("--chain_claim_weight", type=float, default=0.18)
    parser.add_argument("--cross_doc_completion_weight", type=float, default=0.55)
    parser.add_argument("--anchor_completion_weight", type=float, default=0.95)

    parser.add_argument("--assembly_candidates_per_fact", type=int, default=6)
    parser.add_argument("--base_max_docs_per_claim", type=int, default=2)
    parser.add_argument("--max_docs_per_claim_cap", type=int, default=7)
    parser.add_argument("--doc_budget_candidate_docs_threshold", type=int, default=6)
    parser.add_argument("--assembly_depth_gain", type=float, default=0.30)
    parser.add_argument("--assembly_child_gain", type=float, default=0.12)
    parser.add_argument("--assembly_fact_score_weight", type=float, default=0.65)
    parser.add_argument("--assembly_direct_support_weight", type=float, default=0.85)
    parser.add_argument("--assembly_bridge_helper_gain", type=float, default=0.35)
    parser.add_argument("--assembly_dependency_gain", type=float, default=0.80)
    parser.add_argument("--assembly_cross_doc_gain", type=float, default=0.60)
    parser.add_argument("--assembly_fully_covered_gain", type=float, default=0.85)
    parser.add_argument("--assembly_fact_covered_weight", type=float, default=3.50)
    parser.add_argument("--assembly_critical_covered_weight", type=float, default=2.90)
    parser.add_argument("--assembly_dependency_closed_weight", type=float, default=2.15)
    parser.add_argument("--assembly_bridge_closed_weight", type=float, default=1.10)
    parser.add_argument("--assembly_anchor_satisfied_weight", type=float, default=0.80)
    parser.add_argument("--assembly_redundancy_weight", type=float, default=0.45)
    parser.add_argument("--assembly_doc_penalty_weight", type=float, default=0.18)
    parser.add_argument("--assembly_title_penalty_weight", type=float, default=1.40)
    parser.add_argument("--assembly_doc_fact_credit", type=float, default=0.75)
    parser.add_argument("--assembly_same_doc_penalty", type=float, default=0.10)
    parser.add_argument("--assembly_3hop_critical_multiplier", type=float, default=1.40)
    parser.add_argument("--assembly_4hop_critical_multiplier", type=float, default=1.75)
    parser.add_argument("--assembly_3hop_dependency_multiplier", type=float, default=1.60)
    parser.add_argument("--assembly_4hop_dependency_multiplier", type=float, default=2.10)
    parser.add_argument("--assembly_stop_gain", type=float, default=0.02)

    parser.add_argument("--max_export_entry_s", type=int, default=40)
    parser.add_argument("--max_export_entry_n", type=int, default=32)
    parser.add_argument("--max_export_entry_r", type=int, default=24)
    parser.add_argument("--max_export_candidates", type=int, default=64)
    args = parser.parse_args()
    main(args)
