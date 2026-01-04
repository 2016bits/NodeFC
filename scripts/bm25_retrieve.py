import argparse
import json
from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher

def main(args):
    searcher = LuceneSearcher(args.index_path)
    searcher.set_bm25(0.9, 0.4)

    in_path = args.in_path.replace('[SPLIT]', args.split)
    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    for data in tqdm(dataset):
        claim = data['claim']
        gold_evidence = data['gold_evidence_list']
        label = data['label']
        num_hops = data['num_hops']
        hits = searcher.search(claim, k=args.topk)
        retrieved_docs = []
        for hit in hits:
            doc = json.loads(hit.raw)
            retrieved_docs.append(
                {
                    "docid": hit.docid,
                    "score": float(hit.score),
                    "text": doc['contents'],
                }
            )

        result = {
            'id': data['id'],
            'claim': claim,
            'gold_evidence': gold_evidence,
            'label': label,
            'num_hops': num_hops,
            'retrieved_docs': retrieved_docs,
        }
        results.append(result)
    
    out_path = args.out_path.replace('[SPLIT]', args.split)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Retrieve {len(results)} samples from {in_path} to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='/mnt/data/yangjun/data/HOVER/data/converted_data/[SPLIT]_full.json')
    parser.add_argument('--out_path', type=str, default='./data/bm25_[SPLIT].json')
    parser.add_argument('--index_path', type=str, default='/mnt/data/yangjun/data/HOVER/corpus/index')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--split', type=str, default='dev')

    args = parser.parse_args()
    main(args)
