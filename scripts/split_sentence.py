import argparse
import json
import re
from tqdm import tqdm

from nltk.tokenize import sent_tokenize

def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_into_sentences(title, text):
    sentences = []
    if title:
        sentences.append(normalize_ws(title))
    
    if text:
        sents = sent_tokenize(text)
        for s in sents:
            s = normalize_ws(s)
            if s and len(s) > 10:
                # filter small sentences
                sentences.append(s)
    return sentences

def main(args):
    in_path = args.in_path.replace('[SPLIT]', args.split)
    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    for qid, data in enumerate(tqdm(dataset)):
        retrieved = data['retrieved_docs']
        
        sent_nodes = []
        sid_counter = 0
        for rank, doc in enumerate(retrieved):
            title = doc['docid']
            score = doc['score']
            text = doc['text']
            sentences = split_into_sentences(title, text)
            
            for sent_idx, sent in enumerate(sentences):
                sent_nodes.append({
                    'sid': f"s{qid}_{sid_counter}",
                    'docid': title,
                    'doc_rank': rank,
                    'doc_score': score,
                    'sent_idx': sent_idx,
                    'sentences': sent,
                })
                sid_counter += 1
        
        results.append({
            'id': data['id'],
            'claim': data['claim'],
            'label': data['label'],
            'sent_nodes': sent_nodes,
        })
    
    out_path = args.out_path.replace('[SPLIT]', args.split)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"save path: {args.out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/bm25_[SPLIT].json')
    parser.add_argument('--out_path', type=str, default='data/bm25_sentnodes_[SPLIT].json')
    parser.add_argument('--split', type=str, default='dev')

    args = parser.parse_args()
    main(args)
