# retrieve with BM25
```bash
python scripts/bm25_retrieve.py
# role-aware coarse retrieval:
# claim BM25 + critical-fact BM25 + leaf/deep-fact BM25 + bridge/rely_on-fact BM25
# docs are merged by fusion score and filtered with simple diversity constraints
# if you already have decomposition results, pass --decomposition_path
# example:
# python scripts/bm25_retrieve.py --split dev \
#   --decomposition_path data/plan4.2/dev_2_decomposed_0_4000.json \
#   --claim_topk 12 --critical_topk 8 --leaf_topk 6 --bridge_topk 6 \
#   --final_topk 18 --max_docs_per_fact 2 --max_docs_per_cluster 2
```
We can get retrieved documents: bm25_dev.json
```python
result = {
    'id': data['id'],
    'claim': claim,
    'gold_evidence': gold_evidence[{"title", "index", "sentences"}]
    'label': label("SPPORTS"/"REFUTES"),
    'num_hops': num_hops,
    'retrieved_docs': retrieved_docs[{
        "docid", "score", "text",
        "retrieval_routes",
        "matched_atomic_fact_ids",
        "matched_atomic_fact_roles",
        "fusion_components",
    }],
    'retrieval_summary': {
        "claim_topk", "critical_topk", "leaf_topk", "bridge_topk", "final_topk",
        "fact_query_count_by_role",
        "fusion_weights",
        "diversity",
    },
}
```

# split documents into chunk
```bash
python scripts/split_sentence.py
```
We can get sentence nodes: bm25_sentnodes_dev.json
```python
results.append({
    'id': data['id'],
    'claim': data['claim'],
    'label': data['label'],
    'sent_nodes': sent_nodes,
})
```
sent_nodes:
```python
{
    'sid': f"s{qid}_{sid_counter}",     # qid: claim number
    'docid': title,
    'doc_rank': rank,   # retrieve top number
    'doc_score': score,
    'sent_idx': sent_idx,
    'is_title': bool(sent_idx == 0),
    'sentences': sent,
}
```

# construct graph
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/construct_graph.py
```
We can get graph nodes (bm25_nodes_dev.json) and edges (bm25_edges_dev.json)
nodes:
```python
{
    'id': data['id'],
    'sent_nodes': sent_nodes,
    'entity_nodes': entity_nodes,   # {"eid": eid, "name": name, "norm": key}
    'relation_nodes': relation_nodes,   # {"rid": rid, "name": name, "norm": key}
}
```
edges:
```python
{
    'id': data['id'],
    'sn_edges': sn_edges,       # {"sid": sid, "eid": eid}
    'sr_edges': sr_edges,       # {"sid": sid, "rid": rid}
    'nrn_edges': nrn_edges,     # {"head_eid": eid_h, "rid": rid, "tail_eid": eid_t}
}
```

# add semantic edges
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/add_semantic_edge.py
```
We can get added semantic nodes: bm25_semantic_edges_dev.json
```python
{'id': node['id'], 'semantic_edges': edges}
```
edges:
```python
{'sid1': u, 'sid2': v, 'sim': float(sim)}
```

# search graph
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/search_graph_decomposition_aware.py
```

# evaluate
## extract evidence text for evaluation
```bash
python scripts
python scripts/utils/extract_hover_predicted_evidence.py --plan plan4.3
```

## evaluate retrieved_evidence
```bash
python scripts/utils/evaluate_evidence_retrieval.py --plan plan4.3
```
