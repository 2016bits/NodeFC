import json

split = 'train'
t = 'greedy_select_contra_rerank_'
evidence_path = 'data/bm25_noderag_searched_[T][SPLIT].json'.replace('[SPLIT]', split).replace('[T]', t)
raw_path = 'data/bm25_[SPLIT].json'.replace('[SPLIT]', split)

with open(evidence_path, 'r', encoding='utf-8') as f:
    evidences = json.load(f)

with open(raw_path, 'r', encoding='utf-8') as f:
    raws = json.load(f)

id2evidence = {item['id']: item['entry_semantic_texts'][:5] for item in evidences}

results = []
for data in raws:
    id = data['id']
    claim = data['claim']
    gold_evidence = data['gold_evidence']
    retrieved_evidence = id2evidence[id]
    num_hops = data['num_hops']
    label = data['label']
    
    results.append({
        'id': id,
        'claim': claim,
        'gold_evidence': gold_evidence,
        'retrieved_evidence': retrieved_evidence,
        'num_hops': num_hops,
        'label': label
    })

out_path = './data/noderag_retrieved_[T][SPLIT].json'.replace('[SPLIT]', split).replace('[T]', t)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"save path: {out_path}")