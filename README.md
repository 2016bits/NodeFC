# retrieve with BM25
```bash
python scripts/bm25_retrieve.py
```
We can get retrieved documents: bm25_dev.json
```python
result = {
    'id': data['id'],
    'claim': claim,
    'gold_evidence': gold_evidence[{"title", "index", "sentences"}]
    'label': label("SPPORTS"/"REFUTES"),
    'num_hops': num_hops,
    'retrieved_docs': retrieved_docs[{"docid", "score", "text"}],
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
CUDA_VISIBLE_DEVICES=0 python scripts/search_graph.py
```

# convert node id and evidence id to texts
```bash
python scripts/convert_id2text.py
```
data:
```
{
    'id': index,
    'claim': claim,
    'entry_semantic_texts': entry_semantic_texts,   # text
    'entry_entity_texts': entry_entity_texts,       # text
    'top_evidence_texts': top_evidence_texts        # text
}
```

# construct data for crwon verification
```bash
python scripts/construct_verify_data.py
```
data:
```
{
    'id': id,
    'claim': claim,
    'gold_evidence': gold_evidence,
    'retrieved_evidence': retrieved_evidence,
    'num_hops': num_hops,
    'label': label
}
```

# improvement plan

## train verifier model with GNN
train gnn:
```bash
cd scripts/gnn
CUDA_VISIBLE_DEVICES=1 python train_gnn.py
```

## train bert+cls model for verifier
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/verify/verify_with_bert_cls.py
```


