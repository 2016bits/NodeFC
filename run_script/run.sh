# python scripts/bm25_retrieve.py --claim_topk 12 \
#   --atomic_topk 8 \
#   --union_topk 18 \
#   --max_atomic_fact_queries 6 \
#   --rrf_k 60 \
#   --claim_rrf_weight 1.0 \
#   --atomic_rrf_weight 0.9   \
#   --plan plan5.1
# python scripts/split_sentence.py --plan plan5.1
# CUDA_VISIBLE_DEVICES=7 python scripts/construct_graph.py --plan plan5.1
CUDA_VISIBLE_DEVICES=7 python scripts/add_semantic_edge.py --topk 10 --min_sim 0.25 --plan plan5.1
CUDA_VISIBLE_DEVICES=7 python scripts/search_graph_decomposition_aware.py \
  --split dev \
  --plan plan5.1 \
  --decomposition_path "data/[PLAN]/dev_2_decomposed_0_4000.json" \
  --nodes_path "data/[PLAN]/bm25_nodes_[SPLIT].json" \
  --edges_path "data/[PLAN]/bm25_edges_[SPLIT].json" \
  --semantic_edges_path "data/[PLAN]/bm25_semantic_edges_[SPLIT].json" \
  --out_path "data/[PLAN]/nodefc_decomposition_aware_dev_0_4000.json" \
  --device auto \
  --w_ss 0.6 \
  --w_ss_local 0.60 \
  --w_ss_semantic 0.22 \
  --local_sentence_window 2 \
  --w_ss_entity_bonus 0.20 \
  --w_ss_relation_bonus 0.14 \
  --w_ss_number_bonus 0.08 \
  --w_ss_title_penalty 0.45 \
  --bridge_constraint_threshold 0.30 \
  --seed_min_entity_overlap 0.25 \
  --seed_min_relation_match 0.10 \
  --seed_min_constraint_match 0.10 \
  --seed_min_binding_score 0.25 \
  --seed_min_direct_support_score 0.55 \
  --max_support_seed_candidates 2 \
  --max_bridge_seed_candidates 1 \
  --primary_seed_weight 1.00 \
  --secondary_seed_weight 0.35 \
  --title_recall_penalty 0.12 \
  --title_candidate_recall_penalty 0.08 \
  --title_score_penalty 0.18 \
  --title_bridge_penalty 0.20 \
  --assembly_title_penalty_weight 1.40 \
  --chain_primary_seed_weight 0.52 \
  --chain_secondary_seed_weight 0.22 \
  --ppr_structural_expand_weight 0.22 \
  --ppr_local_expand_weight 0.12 \
  --ppr_semantic_expand_weight 0.06 \
  --bridge_stage_min_relative_score 0.25 \
  --bridge_stage_min_hits 3
