#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import unicodedata
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x: str) -> str:
    """
    统一 unicode 表示，避免如:
    'Tomo Zdelarić' vs 'Tomo Zdelarić'
    这种组合字符不同导致的不匹配。
    """
    if x is None:
        return ""
    x = str(x).strip()
    x = unicodedata.normalize("NFKC", x)
    return x


def normalize_list(items: List[str]) -> List[str]:
    return [normalize_text(x) for x in items if normalize_text(x) != ""]


def compute_em_f1(gold_items: List[str], pred_items: List[str]) -> Tuple[float, float]:
    """
    对单个样本计算集合级 EM / F1
    """
    gold_set = set(normalize_list(gold_items))
    pred_set = set(normalize_list(pred_items))

    em = 1.0 if gold_set == pred_set else 0.0

    if len(gold_set) == 0 and len(pred_set) == 0:
        return em, 1.0
    if len(gold_set) == 0 or len(pred_set) == 0:
        return em, 0.0

    overlap = len(gold_set & pred_set)
    precision = overlap / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = overlap / len(gold_set) if len(gold_set) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return em, f1


def build_gold_map(gold_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in gold_data}


def evaluate_retrieval(
    gold_data: List[Dict[str, Any]],
    pred_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_map = build_gold_map(gold_data)

    # 分组统计
    stats = {
        "doc": defaultdict(list),   # key: hop number or "overall"
        "sent": defaultdict(list),
    }

    missing_ids = []
    evaluated_ids = []

    for pred_item in pred_data:
        sample_id = pred_item.get("id")
        if sample_id not in gold_map:
            missing_ids.append(sample_id)
            continue

        gold_item = gold_map[sample_id]
        evaluated_ids.append(sample_id)

        num_hops = gold_item.get("num_hops", None)
        if num_hops is None:
            # 若 gold 里没有 num_hops，则退化为 gold 证据数
            num_hops = len(gold_item.get("gold_doc_titles", []))

        gold_docs = gold_item.get("gold_doc_titles", [])
        pred_docs = pred_item.get("pred_doc_titles", [])

        gold_sents = gold_item.get("gold_sent_keys", [])
        pred_sents = pred_item.get("pred_sent_keys", [])

        doc_em, doc_f1 = compute_em_f1(gold_docs, pred_docs)
        sent_em, sent_f1 = compute_em_f1(gold_sents, pred_sents)

        stats["doc"][num_hops].append({"em": doc_em, "f1": doc_f1})
        stats["doc"]["overall"].append({"em": doc_em, "f1": doc_f1})

        stats["sent"][num_hops].append({"em": sent_em, "f1": sent_f1})
        stats["sent"]["overall"].append({"em": sent_em, "f1": sent_f1})

    def summarize(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metric_list:
            return {"em": 0.0, "f1": 0.0, "count": 0}
        n = len(metric_list)
        avg_em = sum(x["em"] for x in metric_list) / n
        avg_f1 = sum(x["f1"] for x in metric_list) / n
        return {
            "em": avg_em * 100,
            "f1": avg_f1 * 100,
            "count": n,
        }

    results = {
        "meta": {
            "total_gold": len(gold_data),
            "total_pred": len(pred_data),
            "evaluated": len(evaluated_ids),
            "missing_pred_ids_in_gold": missing_ids,
        },
        "document_retrieval": {},
        "sentence_retrieval": {},
    }

    hop_keys = sorted([k for k in stats["doc"].keys() if k != "overall"])
    for hop in hop_keys:
        results["document_retrieval"][str(hop)] = summarize(stats["doc"][hop])
        results["sentence_retrieval"][str(hop)] = summarize(stats["sent"][hop])

    results["document_retrieval"]["overall"] = summarize(stats["doc"]["overall"])
    results["sentence_retrieval"]["overall"] = summarize(stats["sent"]["overall"])

    return results


def format_score(x: float) -> str:
    return f"{x:.1f}"


def print_table_like_paper(results: Dict[str, Any], retrieval_type: str):
    """
    retrieval_type:
      - "document_retrieval"
      - "sentence_retrieval"
    """
    block = results[retrieval_type]

    hop2 = block.get("2", {"em": 0.0, "f1": 0.0})
    hop3 = block.get("3", {"em": 0.0, "f1": 0.0})
    hop4 = block.get("4", {"em": 0.0, "f1": 0.0})
    overall = block.get("overall", {"em": 0.0, "f1": 0.0})

    print("=" * 72)
    print(retrieval_type.replace("_", " ").title())
    print("=" * 72)
    print(f"{'Models':<12}{'2':>12}{'3':>12}{'4':>12}{'Overall':>12}")
    print(
        f"{'YourModel':<12}"
        f"{format_score(hop2['em'])}/{format_score(hop2['f1']):>7}"
        f"{format_score(hop3['em'])}/{format_score(hop3['f1']):>7}"
        f"{format_score(hop4['em'])}/{format_score(hop4['f1']):>7}"
        f"{format_score(overall['em'])}/{format_score(overall['f1']):>7}"
    )
    print()

    print("More detailed stats:")
    for k in ["2", "3", "4", "overall"]:
        if k in block:
            print(
                f"  {k:>7}: "
                f"EM={block[k]['em']:.2f}, "
                f"F1={block[k]['f1']:.2f}, "
                f"count={block[k]['count']}"
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

    results = evaluate_retrieval(gold_data, pred_data)

    print("=" * 72)
    print("Meta")
    print("=" * 72)
    print(json.dumps(results["meta"], ensure_ascii=False, indent=2))
    print()

    print_table_like_paper(results, "document_retrieval")
    print_table_like_paper(results, "sentence_retrieval")

    save_path = args.save_path.replace('[PLAN]', args.plan)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to: {save_path}")


if __name__ == "__main__":
    main()