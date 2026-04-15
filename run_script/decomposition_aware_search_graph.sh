CUDA_VISIBLE_DEVICES=0 python scripts/search_graph_decomposition_aware.py \
  --split dev \
  --decomposition_path ./data/plan4.2/dev_2_decomposed_0_4000.json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json \
  --device cuda
python scripts/utils/convert_id2text.py \
  --in_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json \
  --node_path ./data/plan1/bm25_nodes_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_text.json \
  --split dev
python scripts/utils/construct_verify_data.py \
  --evidence_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_text.json \
  --raw_path ./data/plan1/bm25_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_verifying_data.json \
  --split dev \
  --max_evidence 5
