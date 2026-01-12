CUDA_VISIBLE_DEVICES=0 python scripts/search_graph_hopaware.py \
  --split train \
  --hop_source gold \
  --device cuda \
  --w_entry_r 0.30 --w_entry_n 0.15 --w_entry_s 0.55
python scripts/convert_id2text.py --split train --t nodefc_gold
python scripts/construct_verify_data.py --split train --t nodefc_gold