# python scripts/bm25_retrieve.py --split train
# echo "BM25 retrieval completed."

# python scripts/split_sentence.py --split train
# echo "Sentence splitting completed."

# CUDA_VISIBLE_DEVICES=0 python scripts/construct_graph.py --split train
# echo "Graph construction completed."

CUDA_VISIBLE_DEVICES=0 python scripts/add_semantic_edge.py --split train
echo "Semantic edge addition completed."

CUDA_VISIBLE_DEVICES=0 python scripts/search_graph.py --split train
echo "Graph search completed."
