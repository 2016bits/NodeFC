#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x: str) -> str:
    """
    统一字符串格式，减少由于 unicode 组合字符、空格等导致的误匹配。
    """
    if x is None:
        return ""
    x = unicodedata.normalize("NFKC", str(x))
    x = x.strip()
    return x


def safe_list(obj, key: str) -> List[str]:
    v = obj.get(key, [])
    if v is None:
        return []
    return v


def build_gold_map(gold_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in gold_data}


def compute_set_metrics(gold_items: List[str], pred_items: List[str]) -> Dict[str, float]:
    """
    对整条 claim 的证据集合计算 EM / Precision / Recall / F1
    """
    gold_set = set(normalize_text(x) for x in gold_items if normalize_text(x))
    pred_set = set(normalize_text(x) for x in pred_items if normalize_text(x))

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    em = 1.0 if gold_set == pred_set else 0.0

    return {
        "em": em,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_count": len(gold_set),
        "pred_count": len(pred_set),
    }


def compute_per_hop_metrics(gold_items: List[str], pred_items: List[str]) -> Dict[int, Dict[str, float]]:
    """
    按“第 i 跳（第 i 个证据位置）”统计。
    每一跳 gold 和 pred 都只有一个元素，因此：
      - EM: 是否完全一致
      - Precision/Recall/F1: 在单标签条件下等价于 EM
    """
    max_hops = max(len(gold_items), len(pred_items))
    results = {}

    for i in range(max_hops):
        g = normalize_text(gold_items[i]) if i < len(gold_items) else None
        p = normalize_text(pred_items[i]) if i < len(pred_items) else None

        tp = 1 if (g is not None and p is not None and g == p) else 0
        fp = 1 if (p is not None and (g is None or p != g)) else 0
        fn = 1 if (g is not None and (p is None or p != g)) else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        em = 1.0 if (g is not None and p is not None and g == p) else 0.0

        results[i + 1] = {
            "em": em,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold": g,
            "pred": p,
        }

    return results


def aggregate_metric_dict(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {
            "em": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "count": 0,
        }

    n = len(metric_list)
    return {
        "em": sum(x["em"] for x in metric_list) / n,
        "precision": sum(x["precision"] for x in metric_list) / n,
        "recall": sum(x["recall"] for x in metric_list) / n,
        "f1": sum(x["f1"] for x in metric_list) / n,
        "count": n,
    }


def evaluate(gold_data: List[Dict[str, Any]], pred_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold_map = build_gold_map(gold_data)

    overall_doc_metrics = []
    overall_sent_metrics = []

    hop_doc_metrics = defaultdict(list)   # key: hop_idx
    hop_sent_metrics = defaultdict(list)

    by_num_hops = defaultdict(lambda: {
        "doc_overall": [],
        "sent_overall": [],
        "hop_doc": defaultdict(list),
        "hop_sent": defaultdict(list),
    })

    missing_ids = []
    evaluated = 0

    for pred_item in pred_data:
        claim_id = pred_item.get("id")
        if claim_id not in gold_map:
            missing_ids.append(claim_id)
            continue

        gold_item = gold_map[claim_id]

        gold_docs = safe_list(gold_item, "gold_doc_titles")
        pred_docs = safe_list(pred_item, "pred_doc_titles")

        gold_sents = safe_list(gold_item, "gold_sent_keys")
        pred_sents = safe_list(pred_item, "pred_sent_keys")

        num_hops = gold_item.get("num_hops", len(gold_docs))

        # 总体集合级评估
        doc_overall = compute_set_metrics(gold_docs, pred_docs)
        sent_overall = compute_set_metrics(gold_sents, pred_sents)

        overall_doc_metrics.append(doc_overall)
        overall_sent_metrics.append(sent_overall)

        # 按位置（每一跳）评估
        doc_hops = compute_per_hop_metrics(gold_docs, pred_docs)
        sent_hops = compute_per_hop_metrics(gold_sents, pred_sents)

        for hop_idx, m in doc_hops.items():
            hop_doc_metrics[hop_idx].append(m)
            by_num_hops[num_hops]["hop_doc"][hop_idx].append(m)

        for hop_idx, m in sent_hops.items():
            hop_sent_metrics[hop_idx].append(m)
            by_num_hops[num_hops]["hop_sent"][hop_idx].append(m)

        by_num_hops[num_hops]["doc_overall"].append(doc_overall)
        by_num_hops[num_hops]["sent_overall"].append(sent_overall)

        evaluated += 1

    results = {
        "total_gold": len(gold_data),
        "total_pred": len(pred_data),
        "evaluated": evaluated,
        "missing_pred_ids_in_gold": missing_ids,
        "document_overall": aggregate_metric_dict(overall_doc_metrics),
        "sentence_overall": aggregate_metric_dict(overall_sent_metrics),
        "document_per_hop": {},
        "sentence_per_hop": {},
        "by_num_hops": {},
    }

    for hop_idx in sorted(hop_doc_metrics.keys()):
        results["document_per_hop"][hop_idx] = aggregate_metric_dict(hop_doc_metrics[hop_idx])

    for hop_idx in sorted(hop_sent_metrics.keys()):
        results["sentence_per_hop"][hop_idx] = aggregate_metric_dict(hop_sent_metrics[hop_idx])

    for num_hops, bucket in sorted(by_num_hops.items()):
        results["by_num_hops"][num_hops] = {
            "document_overall": aggregate_metric_dict(bucket["doc_overall"]),
            "sentence_overall": aggregate_metric_dict(bucket["sent_overall"]),
            "document_per_hop": {
                hop_idx: aggregate_metric_dict(metrics)
                for hop_idx, metrics in sorted(bucket["hop_doc"].items())
            },
            "sentence_per_hop": {
                hop_idx: aggregate_metric_dict(metrics)
                for hop_idx, metrics in sorted(bucket["hop_sent"].items())
            }
        }

    return results


def pretty_print_results(results: Dict[str, Any]):
    print("=" * 80)
    print("Evidence Retrieval Evaluation Report")
    print("=" * 80)
    print(f"Total gold examples: {results['total_gold']}")
    print(f"Total pred examples: {results['total_pred']}")
    print(f"Evaluated examples : {results['evaluated']}")
    print(f"Missing pred ids in gold: {len(results['missing_pred_ids_in_gold'])}")
    print()

    print("[Overall - Document]")
    for k in ["em", "precision", "recall", "f1", "count"]:
        print(f"  {k}: {results['document_overall'][k]:.4f}" if k != "count"
              else f"  {k}: {results['document_overall'][k]}")
    print()

    print("[Overall - Sentence]")
    for k in ["em", "precision", "recall", "f1", "count"]:
        print(f"  {k}: {results['sentence_overall'][k]:.4f}" if k != "count"
              else f"  {k}: {results['sentence_overall'][k]}")
    print()

    print("[Per-Hop - Document]")
    for hop_idx, metrics in results["document_per_hop"].items():
        print(
            f"  Hop {hop_idx}: "
            f"EM={metrics['em']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Count={metrics['count']}"
        )
    print()

    print("[Per-Hop - Sentence]")
    for hop_idx, metrics in results["sentence_per_hop"].items():
        print(
            f"  Hop {hop_idx}: "
            f"EM={metrics['em']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Count={metrics['count']}"
        )
    print()

    print("[Grouped by num_hops]")
    for num_hops, block in results["by_num_hops"].items():
        print(f"  num_hops = {num_hops}")
        d = block["document_overall"]
        s = block["sentence_overall"]
        print(f"    Document Overall: EM={d['em']:.4f}, P={d['precision']:.4f}, R={d['recall']:.4f}, F1={d['f1']:.4f}, Count={d['count']}")
        print(f"    Sentence Overall: EM={s['em']:.4f}, P={s['precision']:.4f}, R={s['recall']:.4f}, F1={s['f1']:.4f}, Count={s['count']}")

        print(f"    Document Per-Hop:")
        for hop_idx, metrics in block["document_per_hop"].items():
            print(
                f"      Hop {hop_idx}: "
                f"EM={metrics['em']:.4f}, "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Count={metrics['count']}"
            )

        print(f"    Sentence Per-Hop:")
        for hop_idx, metrics in block["sentence_per_hop"].items():
            print(
                f"      Hop {hop_idx}: "
                f"EM={metrics['em']:.4f}, "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Count={metrics['count']}"
            )
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=str, default='data/[PLAN]/gold_evidence_dev.json')
    parser.add_argument("--pred_path", type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000_pred_evidence.json')
    parser.add_argument("--save_path", type=str, default='data/[PLAN]/hover_eval_results.json')
    parser.add_argument("--plan", type=str, default='plan4.2')
    args = parser.parse_args()

    gold_data = load_json(args.gold_path.replace('[PLAN]', args.plan))
    pred_data = load_json(args.pred_path.replace('[PLAN]', args.plan))

    results = evaluate(gold_data, pred_data)
    pretty_print_results(results)

    save_path = args.save_path.replace('[PLAN]', args.plan)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {save_path}")


if __name__ == "__main__":
    main()