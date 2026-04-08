python scripts/bm25_retrieve.py --topk 20
python scripts/split_sentence.py
CUDA_VISIBLE_DEVICES=0 python scripts/construct_graph.py
