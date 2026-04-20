#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(x: Any) -> str:
    if x is None:
        return ''
    x = unicodedata.normalize('NFKC', str(x)).strip()
    return x


def normalize_list(items: Iterable[Any]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        nx = normalize_text(x)
        if nx and nx not in seen:
            seen.add(nx)
            out.append(nx)
    return out


def compute_set_metrics(gold_items: List[str], pred_items: List[str]) -> Dict[str, float]:
    gold_set = set(normalize_list(gold_items))
    pred_set = set(normalize_list(pred_items))

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    em = 1.0 if gold_set == pred_set else 0.0

    return {
        'em': em,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gold_count': len(gold_set),
        'pred_count': len(pred_set),
        'hit_count': tp,
    }


def compute_oracle_metrics(gold_items: List[str], observed_items: List[str]) -> Dict[str, float]:
    gold_set = set(normalize_list(gold_items))
    obs_set = set(normalize_list(observed_items))

    if not gold_set:
        return {
            'any_hit': 1.0,
            'all_hit': 1.0,
            'recall': 1.0,
            'hit_count': 0,
            'gold_count': 0,
        }

    hit_count = len(gold_set & obs_set)
    return {
        'any_hit': 1.0 if hit_count > 0 else 0.0,
        'all_hit': 1.0 if hit_count == len(gold_set) else 0.0,
        'recall': hit_count / len(gold_set),
        'hit_count': hit_count,
        'gold_count': len(gold_set),
    }


def aggregate_set_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {
            'em': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'count': 0,
            'avg_gold_count': 0.0,
            'avg_pred_count': 0.0,
            'avg_hit_count': 0.0,
        }
    n = len(metric_list)
    return {
        'em': sum(x['em'] for x in metric_list) / n,
        'precision': sum(x['precision'] for x in metric_list) / n,
        'recall': sum(x['recall'] for x in metric_list) / n,
        'f1': sum(x['f1'] for x in metric_list) / n,
        'count': n,
        'avg_gold_count': sum(x['gold_count'] for x in metric_list) / n,
        'avg_pred_count': sum(x['pred_count'] for x in metric_list) / n,
        'avg_hit_count': sum(x['hit_count'] for x in metric_list) / n,
    }


def aggregate_oracle_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {
            'any_hit': 0.0,
            'all_hit': 0.0,
            'recall': 0.0,
            'count': 0,
            'avg_gold_count': 0.0,
            'avg_hit_count': 0.0,
        }
    n = len(metric_list)
    return {
        'any_hit': sum(x['any_hit'] for x in metric_list) / n,
        'all_hit': sum(x['all_hit'] for x in metric_list) / n,
        'recall': sum(x['recall'] for x in metric_list) / n,
        'count': n,
        'avg_gold_count': sum(x['gold_count'] for x in metric_list) / n,
        'avg_hit_count': sum(x['hit_count'] for x in metric_list) / n,
    }


def build_gold_map(gold_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item['id']: item for item in gold_data}


def evaluate_retrieval(gold_data: List[Dict[str, Any]], pred_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold_map = build_gold_map(gold_data)

    grouped = defaultdict(lambda: {
        'doc_final': [],
        'sent_final': [],
        'gold_doc_in_candidates': [],
        'gold_sent_in_candidates': [],
        'gold_sent_in_rerank_topk': [],
        'gold_sent_in_final_selected': [],
    })

    overall = {
        'doc_final': [],
        'sent_final': [],
        'gold_doc_in_candidates': [],
        'gold_sent_in_candidates': [],
        'gold_sent_in_rerank_topk': [],
        'gold_sent_in_final_selected': [],
    }

    missing_ids = []
    per_claim = []

    for pred_item in pred_data:
        sample_id = pred_item.get('id')
        if sample_id not in gold_map:
            missing_ids.append(sample_id)
            continue

        gold_item = gold_map[sample_id]
        num_hops = gold_item.get('num_hops', len(gold_item.get('gold_doc_titles', [])))

        gold_docs = gold_item.get('gold_doc_titles', []) or []
        gold_sents = gold_item.get('gold_sent_keys', []) or []

        pred_docs = pred_item.get('pred_doc_titles', []) or []
        pred_sents = pred_item.get('pred_sent_keys', []) or []
        candidate_docs = pred_item.get('candidate_doc_titles', []) or []
        candidate_sents = pred_item.get('candidate_sent_keys', []) or []
        rerank_topk_sents = pred_item.get('rerank_topk_sent_keys', []) or []

        doc_final = compute_set_metrics(gold_docs, pred_docs)
        sent_final = compute_set_metrics(gold_sents, pred_sents)

        gold_doc_in_candidates = compute_oracle_metrics(gold_docs, candidate_docs)
        gold_sent_in_candidates = compute_oracle_metrics(gold_sents, candidate_sents)
        gold_sent_in_rerank_topk = compute_oracle_metrics(gold_sents, rerank_topk_sents)
        gold_sent_in_final_selected = compute_oracle_metrics(gold_sents, pred_sents)

        grouped[num_hops]['doc_final'].append(doc_final)
        grouped[num_hops]['sent_final'].append(sent_final)
        grouped[num_hops]['gold_doc_in_candidates'].append(gold_doc_in_candidates)
        grouped[num_hops]['gold_sent_in_candidates'].append(gold_sent_in_candidates)
        grouped[num_hops]['gold_sent_in_rerank_topk'].append(gold_sent_in_rerank_topk)
        grouped[num_hops]['gold_sent_in_final_selected'].append(gold_sent_in_final_selected)

        for key, val in [
            ('doc_final', doc_final),
            ('sent_final', sent_final),
            ('gold_doc_in_candidates', gold_doc_in_candidates),
            ('gold_sent_in_candidates', gold_sent_in_candidates),
            ('gold_sent_in_rerank_topk', gold_sent_in_rerank_topk),
            ('gold_sent_in_final_selected', gold_sent_in_final_selected),
        ]:
            overall[key].append(val)

        per_claim.append({
            'id': sample_id,
            'num_hops': num_hops,
            'document_final_metrics': doc_final,
            'sentence_final_metrics': sent_final,
            'gold_doc_in_candidates': gold_doc_in_candidates,
            'gold_sent_in_candidates': gold_sent_in_candidates,
            'gold_sent_in_rerank_topk': gold_sent_in_rerank_topk,
            'gold_sent_in_final_selected': gold_sent_in_final_selected,
        })

    results = {
        'meta': {
            'total_gold': len(gold_data),
            'total_pred': len(pred_data),
            'evaluated': len(per_claim),
            'missing_pred_ids_in_gold': missing_ids,
        },
        'document_final': aggregate_set_metrics(overall['doc_final']),
        'sentence_final': aggregate_set_metrics(overall['sent_final']),
        'oracle': {
            'gold_doc_in_candidates': aggregate_oracle_metrics(overall['gold_doc_in_candidates']),
            'gold_sent_in_candidates': aggregate_oracle_metrics(overall['gold_sent_in_candidates']),
            'gold_sent_in_rerank_topk': aggregate_oracle_metrics(overall['gold_sent_in_rerank_topk']),
            'gold_sent_in_final_selected': aggregate_oracle_metrics(overall['gold_sent_in_final_selected']),
        },
        'by_num_hops': {},
        'per_claim': per_claim,
    }

    for hop in sorted(grouped.keys()):
        results['by_num_hops'][str(hop)] = {
            'document_final': aggregate_set_metrics(grouped[hop]['doc_final']),
            'sentence_final': aggregate_set_metrics(grouped[hop]['sent_final']),
            'oracle': {
                'gold_doc_in_candidates': aggregate_oracle_metrics(grouped[hop]['gold_doc_in_candidates']),
                'gold_sent_in_candidates': aggregate_oracle_metrics(grouped[hop]['gold_sent_in_candidates']),
                'gold_sent_in_rerank_topk': aggregate_oracle_metrics(grouped[hop]['gold_sent_in_rerank_topk']),
                'gold_sent_in_final_selected': aggregate_oracle_metrics(grouped[hop]['gold_sent_in_final_selected']),
            },
        }

    return results


def print_set_block(title: str, metrics: Dict[str, float]):
    print(title)
    print(f"  EM         : {metrics['em'] * 100:.2f}")
    print(f"  Precision  : {metrics['precision'] * 100:.2f}")
    print(f"  Recall     : {metrics['recall'] * 100:.2f}")
    print(f"  F1         : {metrics['f1'] * 100:.2f}")
    print(f"  Avg Gold   : {metrics['avg_gold_count']:.4f}")
    print(f"  Avg Pred   : {metrics['avg_pred_count']:.4f}")
    print(f"  Avg Hit    : {metrics['avg_hit_count']:.4f}")
    print(f"  Count      : {metrics['count']}")



def print_oracle_block(title: str, metrics: Dict[str, float]):
    print(title)
    print(f"  Any Hit    : {metrics['any_hit'] * 100:.2f}")
    print(f"  All Hit    : {metrics['all_hit'] * 100:.2f}")
    print(f"  Recall     : {metrics['recall'] * 100:.2f}")
    print(f"  Avg Gold   : {metrics['avg_gold_count']:.4f}")
    print(f"  Avg Hit    : {metrics['avg_hit_count']:.4f}")
    print(f"  Count      : {metrics['count']}")


def pretty_print(results: Dict[str, Any]):
    print('=' * 72)
    print('Meta')
    print('=' * 72)
    print(json.dumps(results['meta'], ensure_ascii=False, indent=2))
    print()

    print('=' * 72)
    print('Final Retrieval Metrics')
    print('=' * 72)
    print_set_block('[Document Final]', results['document_final'])
    print()
    print_set_block('[Sentence Final]', results['sentence_final'])
    print()

    print('=' * 72)
    print('Oracle Diagnostics')
    print('=' * 72)
    print_oracle_block('[gold_doc_in_candidates]', results['oracle']['gold_doc_in_candidates'])
    print()
    print_oracle_block('[gold_sent_in_candidates]', results['oracle']['gold_sent_in_candidates'])
    print()
    print_oracle_block('[gold_sent_in_rerank_topk]', results['oracle']['gold_sent_in_rerank_topk'])
    print()
    print_oracle_block('[gold_sent_in_final_selected]', results['oracle']['gold_sent_in_final_selected'])
    print()

    print('=' * 72)
    print('Grouped by num_hops')
    print('=' * 72)
    for hop, block in results['by_num_hops'].items():
        print(f'num_hops = {hop}')
        print_set_block('  [Document Final]', block['document_final'])
        print_set_block('  [Sentence Final]', block['sentence_final'])
        print_oracle_block('  [gold_doc_in_candidates]', block['oracle']['gold_doc_in_candidates'])
        print_oracle_block('  [gold_sent_in_candidates]', block['oracle']['gold_sent_in_candidates'])
        print_oracle_block('  [gold_sent_in_rerank_topk]', block['oracle']['gold_sent_in_rerank_topk'])
        print_oracle_block('  [gold_sent_in_final_selected]', block['oracle']['gold_sent_in_final_selected'])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='data/plan4.2/gold_evidence_dev.json')
    parser.add_argument('--pred_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000_pred_evidence.json')
    parser.add_argument('--save_path', type=str, default='data/[PLAN]/hover_eval_results.json')
    parser.add_argument('--plan', type=str, default='plan4.3')
    args = parser.parse_args()

    gold_data = load_json(args.gold_path.replace('[PLAN]', args.plan))
    pred_data = load_json(args.pred_path.replace('[PLAN]', args.plan))

    results = evaluate_retrieval(gold_data, pred_data)
    pretty_print(results)

    save_path = args.save_path.replace('[PLAN]', args.plan)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'Saved results to: {save_path}')


if __name__ == '__main__':
    main()
