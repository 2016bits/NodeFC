import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default='data/plan2/[T]_[SPLIT].json')
parser.add_argument('--node_path', type=str, default='data/plan1/bm25_nodes_[SPLIT].json')
parser.add_argument('--out_path', type=str, default='./data/plan2/[T]_[SPLIT]_text.json')
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--t', type=str, default='')
args = parser.parse_args()

split = args.split
t = args.t
in_path = args.in_path.replace('[SPLIT]', split).replace('[T]', t)
node_path = args.node_path.replace('[SPLIT]', split)

with open(in_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

with open(node_path, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

node_id2text = {}
for item in nodes:
    for sent_node in item['sent_nodes']:
        node_id2text[sent_node['sid']] = sent_node['sentences'].strip()
    
    for entity_node in item['entity_nodes']:
        node_id2text[entity_node['eid']] = entity_node['name'].strip()
    
    for relation_node in item['relation_nodes']:
        node_id2text[relation_node['rid']] = relation_node['name'].strip()

results = []
for data in dataset:
    index = data['id']
    claim = data['claim']
    entry_semantic_texts = []
    for id in data['entry_sids']:
        entry_semantic_texts.append(node_id2text[id])
    entry_entity_texts = []
    for id in data['entry_nids']:
        entry_entity_texts.append(node_id2text[id])
    top_evidence_texts = []
    for evi in data['top_evidences']:
        top_evidence_texts.append(evi['text'])
    
    results.append({
        'id': index,
        'claim': claim,
        'entry_semantic_texts': entry_semantic_texts,
        'entry_entity_texts': entry_entity_texts,
        'top_evidence_texts': top_evidence_texts
    })

out_path = args.out_path.replace('[T]', t).replace('[SPLIT]', split)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"save path: {out_path}")