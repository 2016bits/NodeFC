python scripts/bm25_retrieve.py --split dev
echo "BM25 retrieval completed."

python scripts/split_sentence.py --split dev
echo "Sentence splitting completed."

CUDA_VISIBLE_DEVICES=1 python scripts/construct_graph.py --split dev
echo "Graph construction completed."

CUDA_VISIBLE_DEVICES=1 python scripts/add_semantic_edge.py --split dev
echo "Semantic edge addition completed."

CUDA_VISIBLE_DEVICES=1 python scripts/search_graph.py --split dev
echo "Graph search completed."
