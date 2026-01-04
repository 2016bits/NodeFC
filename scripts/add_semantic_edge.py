import argparse
import json
import numpy as np
import hnswlib
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def build_hnsw_index(vectors, sids, k, ef):
    """
    用 HNSW 在局部句子向量上构建语义近邻边。
    返回无向去重边（u < v）。
    注意：hnswlib cosine 距离 dist = 1 - cos_sim
    """
    n = vectors.shape[0]    # 句子数量
    if n <= 1:
        return []
    
    dim = vectors.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    
    # HNSW 参数：M / ef_construction 影响构建质量与速度
    index.init_index(max_elements=n, ef_construction=200, M=32)
    index.add_items(vectors, np.arange(n))
    index.set_ef(ef)
    
    # 查 knn：取 k+1，跳过自己
    kq = min(k + 1, n)
    labels, dists = index.knn_query(vectors, k=kq)
    
    edges = []
    seen = set()
    for i in range(n):
        for j, dist in zip(labels[i], dists[i]):
            if i == j:
                continue
            u, v = sorted((sids[i], sids[j]))
            if (u, v) in seen:
                continue
            seen.add((u, v))
            sim = 1.0 - dist
            edges.append({'sid1': u, 'sid2': v, 'sim': float(sim)})
    return edges

def main(args):
    nodes_path = args.nodes_path.replace('[SPLIT]', args.split)
    with open(nodes_path, 'r') as f:
        nodes_data = json.load(f)
    print("Loaded nodes data.")
    
    model = SentenceTransformer(args.embedding_model)
    print("Initialized embedding model.")
    
    semantic_edges = []
    for node in tqdm(nodes_data):
        sent_nodes = node['sent_nodes']
        
        sids = []
        texts = []
        for s in sent_nodes:
            txt = s['sentences'].strip()
            if not txt:
                continue
            sids.append(s['sid'])
            texts.append(txt)
        
        if len(texts) == 0:
            semantic_edges.append({'id': node['id'], 'semantic_edges': []})
            continue
        
        vecs = model.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)
        
        edges = build_hnsw_index(vecs, sids, args.topk, args.ef)
        
        edges = [e for e in edges if e['sim'] >= args.min_sim]
        semantic_edges.append({'id': node['id'], 'semantic_edges': edges})
    
    semantic_edges_path = args.semantic_edges_path.replace('[SPLIT]', args.split)
    with open(semantic_edges_path, 'w') as f:
        json.dump(semantic_edges, f, indent=4)
    print("Saved semantic edges.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes_path', type=str, default='./data/bm25_nodes_[SPLIT].json')
    parser.add_argument('--semantic_edges_path', type=str, default='./data/bm25_semantic_edges_[SPLIT].json')
    parser.add_argument('--split', type=str, default='dev')
    
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--ef', type=int, default=100, help="HNSW ef (query time/quality tradeoff)")
    parser.add_argument('--min_sim', type=float, default=0.25, help="filter edges with sim < min_sim; set None to disable")
    
    args = parser.parse_args()
    main(args)
