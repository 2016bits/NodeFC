import json
import argparse
import re
from collections import defaultdict

def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def sent_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    if pred in gold or gold in pred:
        return True
    return False

def main(args):
    in_path = args.in_path
    with open(in_path, 'r') as f:
        dataset = json.load(f)
    
    ks = [1, 5]
    stats = {
        hop: {
            k: {
                "sent_any": 0.0,
                "sent_all": 0.0,
                "count": 0
            } for k in ks
        } for hop in [2, 3, 4]
    }
    
    for data in dataset:
        hop = data['num_hops']
        retrieved_evidence = data['retrieved_evidence']
        gold_evidence = [evi['sentences'] for evi in data['gold_evidence']]
        
        if len(gold_evidence) == 0:
            continue
        
        for k in ks:
            topk = retrieved_evidence[:k]
            matched = 0
            
            for gs in gold_evidence:
                for ps in topk:
                    if sent_match(norm_text(ps), norm_text(gs)):
                        matched += 1
                        break
            
            sent_any = 1.0 if matched > 0 else 0.0
            sent_all = matched / len(gold_evidence)
            
            stats[hop][k]["sent_any"] += sent_any
            stats[hop][k]["sent_all"] += sent_all
            stats[hop][k]["count"] += 1
    
    print("\n===== Evidence Retrieval Recall by Hop =====")
    for hop in [2, 3, 4]:
        print(f"\n--- {hop}-hop ---")
        for k in ks:
            c = stats[hop][k]["count"]
            if c == 0:
                print(f"K={k}: no samples")
                continue
            any_r = stats[hop][k]["sent_any"] / c
            all_r = stats[hop][k]["sent_all"] / c
            print(f"K={k:2d} | Sentence Recall(any)={any_r:.4f} | Sentence Recall(all)={all_r:.4f} | N={c}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/plan2/nodefc_model_dev_verifying_data.json')

    args = parser.parse_args()
    main(args)