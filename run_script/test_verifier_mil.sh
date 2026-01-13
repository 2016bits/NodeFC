python scripts/verify/train_verifier_mil.py \
  --t nodefc_model \
  --data_path ./data/plan2/[T]_[TYPE]_verifying_data.json \
  --checkpoint ./save_models/bert_mil/bert_mil_nodefc_model_best.pth \
  --device cuda \
  --batch_size 16 \
  --max_ev 5 \
  --max_len 192 \
  --pooling max
