#!/usr/bin/env bash
SPLIT="${SPLIT:-dev}"
PLAN="${PLAN:-plan5.2}"
LIMIT="${LIMIT:-0}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-6}"
RUN_EVAL="${RUN_EVAL:-0}"

INPUT_PATH="${INPUT_PATH:-/mnt/data/yangjun/data/HOVER/data/converted_data/${SPLIT}_full.json}"
INDEX_PATH="${INDEX_PATH:-/mnt/data/yangjun/data/HOVER/corpus/index}"
DECOMPOSITION_PATH="${DECOMPOSITION_PATH:-data/${PLAN}/dev_2_decomposed_0_4000.json}"
GOLD_PATH="${GOLD_PATH:-data/plan4.2/gold_evidence_${SPLIT}.json}"

echo "[1/6] BM25 role-aware retrieval"
python scripts/bm25_retrieve.py \
  --split "${SPLIT}" \
  --plan "${PLAN}" \
  --in_path "${INPUT_PATH}" \
  --out_path "data/${PLAN}/bm25_${SPLIT}.json" \
  --index_path "${INDEX_PATH}" \
  --decomposition_path "${DECOMPOSITION_PATH}" \
  --claim_topk 12 \
  --critical_topk 8 \
  --leaf_topk 6 \
  --bridge_topk 6 \
  --final_topk 18 \
  --max_atomic_fact_queries 6 \
  --max_docs_per_fact 2 \
  --max_docs_per_cluster 2 \
  --w_claim 1.0 \
  --w_fact 0.9 \
  --w_multi 0.15 \
  --w_role 0.20

echo "[2/6] Split retrieved documents into sentence nodes"
python scripts/split_sentence.py \
  --split "${SPLIT}" \
  --plan "${PLAN}" \
  --in_path "data/${PLAN}/bm25_${SPLIT}.json" \
  --out_path "data/${PLAN}/bm25_sentnodes_${SPLIT}.json"

echo "[3/6] Construct graph"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/construct_graph.py \
  --split "${SPLIT}" \
  --plan "${PLAN}" \
  --in_path "data/${PLAN}/bm25_sentnodes_${SPLIT}.json" \
  --out_nodes_path "data/${PLAN}/bm25_nodes_${SPLIT}.json" \
  --out_edges_path "data/${PLAN}/bm25_edges_${SPLIT}.json"

echo "[4/6] Add semantic edges"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/add_semantic_edge.py \
  --split "${SPLIT}" \
  --plan "${PLAN}" \
  --nodes_path "data/${PLAN}/bm25_nodes_${SPLIT}.json" \
  --semantic_edges_path "data/${PLAN}/bm25_semantic_edges_${SPLIT}.json" \
  --topk 10 \
  --min_sim 0.25

echo "[5/6] Decomposition-aware graph retrieval"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/search_graph_decomposition_aware.py \
  --split "${SPLIT}" \
  --plan "${PLAN}" \
  --limit "${LIMIT}" \
  --decomposition_path "${DECOMPOSITION_PATH}" \
  --nodes_path "data/${PLAN}/bm25_nodes_${SPLIT}.json" \
  --edges_path "data/${PLAN}/bm25_edges_${SPLIT}.json" \
  --semantic_edges_path "data/${PLAN}/bm25_semantic_edges_${SPLIT}.json" \
  --out_path "data/${PLAN}/nodefc_decomposition_aware_${SPLIT}_0_4000.json" \
  --device auto \
  --w_ss 0.6 \
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
  --title_recall_penalty 0.12 \
  --title_candidate_recall_penalty 0.08 \
  --title_score_penalty 0.18 \
  --title_bridge_penalty 0.20 \
  --assembly_title_penalty_weight 1.40

echo "[6/6] Extract predicted evidence and evaluate"
python scripts/utils/extract_hover_predicted_evidence.py \
  --plan "${PLAN}" \
  --retrieval_path "data/${PLAN}/nodefc_decomposition_aware_${SPLIT}_0_4000.json" \
  --raw_path "data/${PLAN}/bm25_${SPLIT}.json" \
  --output_path "data/${PLAN}/nodefc_decomposition_aware_${SPLIT}_0_4000_pred_evidence.json" \
  --stats_path "data/${PLAN}/nodefc_decomposition_aware_${SPLIT}_0_4000_pred_evidence_stats.json"

python scripts/utils/evaluate_evidence_retrieval.py \
  --plan "${PLAN}" \
  --gold_path "${GOLD_PATH}" \
  --pred_path "data/${PLAN}/nodefc_decomposition_aware_${SPLIT}_0_4000_pred_evidence.json" \
  --save_path "data/${PLAN}/hover_eval_results_${SPLIT}.json"

