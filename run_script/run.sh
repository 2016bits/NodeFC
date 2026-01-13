torchrun --nproc_per_node=8 scripts/verify/train_verifier_mil.py \
  --do_train \
  --t nodefc_heuristic \
  --bert_model_name bert-base-uncased \
  --max_ev 5 \
  --pooling max \
  --batch_size 16 \
  --max_len 256

torchrun --nproc_per_node=8 scripts/verify/train_verifier_mil.py \
  --do_train \
  --t nodefc_model \
  --bert_model_name bert-base-uncased \
  --max_ev 5 \
  --pooling max \
  --batch_size 16 \
  --max_len 256

