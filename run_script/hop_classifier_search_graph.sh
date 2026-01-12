CUDA_VISIBLE_DEVICES=7 python scripts/search_graph_hopaware.py \
  --split train \
  --hop_source model \
  --hop_model ./save_models/hop_cls/ \
  --device cuda \
  --w_entry_r 0.30 --w_entry_n 0.15 --w_entry_s 0.55
