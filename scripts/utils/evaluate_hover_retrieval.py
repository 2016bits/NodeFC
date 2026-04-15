#!/usr/bin/env python3
import argparse
import json
import unicodedata
from collections import defaultdict
from statistics import mean


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = " ".join(s.strip().split())
    return s


def normalize_title(s: str) -> str:
    return normalize_text(s)


def normalize_key(s: str) -> str:
    s = normalize_text(s)
    if "@@" not in s:
        return s
    title, idx = s.rsplit("@@", 1)
    title = normalize_title(title)
    idx = str(idx).strip()
    return f"{title}@@{idx}"


def to_doc_set(item, prefix: str):
    titles = item.get(f"{prefix}_doc_titles", []) or []
    if titles:
        return {normalize_title(x) for x in titles if str(x).strip()}

    ev_list = item.get(f"{prefix}_evidence_list", []) or []
    out = set()
    for ev in ev_list:
        t = normalize_title(ev.get("title", ""))
        if t:
            out.add(t)
    return out


def to_sent_set(item, prefix: str):
    keys = item.get(f"{prefix}_sent_keys", []) or []
    if keys:
        return {normalize_key(x) for x in keys if str(x).strip()}

    ev_list = item.get(f"{prefix}_evidence_list", []) or []
    out = set()
    for ev in ev_list:
        title = normalize_title(ev.get("title", ""))
        idx = ev.get("index", None)
        if title and idx is not None:
            out.add(f"{title}@@{idx}")
    return out


def safe_div(a, b):
    return a / b if b else 0.0


def prf1(tp, pred_n, gold_n):
    p = safe_div(tp, pred_n)
    r = safe_div(tp, gold_n)
    f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_items(gold_items, pred_items):
    gold_by_id = {x["id"]: x for x in gold_items}
    pred_by_id = {x["id"]: x for x in pred_items}
    common_ids = [cid for cid in gold_by_id if cid in pred_by_id]

    per_claim = []
    agg = defaultdict(float)
    by_hop = defaultdict(lambda: defaultdict(float))

    for cid in common_ids:
        g = gold_by_id[cid]
        p = pred_by_id[cid]
        hop = g.get("num_hops", p.get("num_hops", "unknown"))

        g_docs = to_doc_set(g, "gold")
        p_docs = to_doc_set(p, "pred")
        g_sents = to_sent_set(g, "gold")
        p_sents = to_sent_set(p, "pred")

        doc_tp = len(g_docs & p_docs)
        sent_tp = len(g_sents & p_sents)

        doc_any = 1.0 if doc_tp > 0 else 0.0
        doc_all = 1.0 if g_docs.issubset(p_docs) else 0.0
        sent_any = 1.0 if sent_tp > 0 else 0.0
        sent_all = 1.0 if g_sents.issubset(p_sents) else 0.0
        doc_em = 1.0 if g_docs == p_docs else 0.0
        sent_em = 1.0 if g_sents == p_sents else 0.0

        doc_p, doc_r, doc_f1 = prf1(doc_tp, len(p_docs), len(g_docs))
        sent_p, sent_r, sent_f1 = prf1(sent_tp, len(p_sents), len(g_sents))

        row = {
            "id": cid,
            "claim": g.get("claim", p.get("claim", "")),
            "num_hops": hop,
            "gold_doc_count": len(g_docs),
            "pred_doc_count": len(p_docs),
            "gold_sent_count": len(g_sents),
            "pred_sent_count": len(p_sents),
            "doc_hit_count": doc_tp,
            "sent_hit_count": sent_tp,
            "doc_any_recall": doc_any,
            "doc_all_recall": doc_all,
            "sent_any_recall": sent_any,
            "sent_all_recall": sent_all,
            "doc_em": doc_em,
            "sent_em": sent_em,
            "doc_precision": doc_p,
            "doc_recall": doc_r,
            "doc_f1": doc_f1,
            "sent_precision": sent_p,
            "sent_recall": sent_r,
            "sent_f1": sent_f1,
            "gold_doc_titles": sorted(g_docs),
            "pred_doc_titles": sorted(p_docs),
            "gold_sent_keys": sorted(g_sents),
            "pred_sent_keys": sorted(p_sents),
            "missing_docs": sorted(g_docs - p_docs),
            "extra_docs": sorted(p_docs - g_docs),
            "missing_sents": sorted(g_sents - p_sents),
            "extra_sents": sorted(p_sents - g_sents),
        }
        per_claim.append(row)

        for bucket in (agg, by_hop[hop]):
            bucket["count"] += 1
            bucket["doc_any_recall"] += doc_any
            bucket["doc_all_recall"] += doc_all
            bucket["sent_any_recall"] += sent_any
            bucket["sent_all_recall"] += sent_all
            bucket["doc_em"] += doc_em
            bucket["sent_em"] += sent_em
            bucket["doc_precision"] += doc_p
            bucket["doc_recall"] += doc_r
            bucket["doc_f1"] += doc_f1
            bucket["sent_precision"] += sent_p
            bucket["sent_recall"] += sent_r
            bucket["sent_f1"] += sent_f1
            bucket["avg_doc_hit_count"] += doc_tp
            bucket["avg_sent_hit_count"] += sent_tp
            bucket["avg_pred_doc_count"] += len(p_docs)
            bucket["avg_pred_sent_count"] += len(p_sents)
            bucket["avg_gold_doc_count"] += len(g_docs)
            bucket["avg_gold_sent_count"] += len(g_sents)

    def finalize(bucket):
        n = bucket.get("count", 0)
        if not n:
            return {"count": 0}
        out = {"count": int(n)}
        for k, v in bucket.items():
            if k == "count":
                continue
            out[k] = v / n
        return out

    summary = {
        "total_gold": len(gold_items),
        "total_pred": len(pred_items),
        "total_common": len(common_ids),
        "overall": finalize(agg),
        "by_hop": {str(h): finalize(b) for h, b in sorted(by_hop.items(), key=lambda x: str(x[0]))},
    }
    return summary, per_claim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default='data/plan4.2/gold_evidence_dev.json', help="Path to gold evidence json/jsonl")
    ap.add_argument("--pred", type=str, default='data/plan4.2/nodefc_decomposition_aware_dev_0_4000_pred_evidence.json', help="Path to predicted evidence json/jsonl")
    ap.add_argument("--out_summary", default='data/plan4.2/hover_eval_summary.json', help="Where to save summary json")
    ap.add_argument("--out_per_claim", default='data/plan4.2/hover_eval_per_claim.jsonl', help="Where to save per-claim jsonl")
    args = ap.parse_args()

    def load_any(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        if text[0] == "[":
            return json.loads(text)
        if text[0] == "{":
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "id" in obj:
                return [obj]
            raise ValueError(f"Unsupported JSON object in {path}")
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

    gold_items = load_any(args.gold)
    pred_items = load_any(args.pred)
    summary, per_claim = evaluate_items(gold_items, pred_items)

    print("=" * 80)
    print("Evidence Evaluation Report")
    print("=" * 80)
    print(f"Total gold examples: {summary['total_gold']}")
    print(f"Total pred examples: {summary['total_pred']}")
    print(f"Total aligned examples: {summary['total_common']}")
    print()
    ov = summary["overall"]
    if ov.get("count", 0):
        print("[overall]")
        print(f"Doc Any Recall:  {ov['doc_any_recall']:.4f}")
        print(f"Doc All Recall:  {ov['doc_all_recall']:.4f}")
        print(f"Sent Any Recall: {ov['sent_any_recall']:.4f}")
        print(f"Sent All Recall: {ov['sent_all_recall']:.4f}")
        print(f"Doc EM:          {ov['doc_em']:.4f}")
        print(f"Sent EM:         {ov['sent_em']:.4f}")
        print(f"Doc Precision:   {ov['doc_precision']:.4f}")
        print(f"Doc Recall:      {ov['doc_recall']:.4f}")
        print(f"Doc F1:          {ov['doc_f1']:.4f}")
        print(f"Sent Precision:  {ov['sent_precision']:.4f}")
        print(f"Sent Recall:     {ov['sent_recall']:.4f}")
        print(f"Sent F1:         {ov['sent_f1']:.4f}")
        print(f"Avg Doc Hit:     {ov['avg_doc_hit_count']:.4f}")
        print(f"Avg Sent Hit:    {ov['avg_sent_hit_count']:.4f}")
        print(f"Avg Pred Docs:   {ov['avg_pred_doc_count']:.4f}")
        print(f"Avg Pred Sents:  {ov['avg_pred_sent_count']:.4f}")
        print()

    print("[by hop]")
    for hop, b in summary["by_hop"].items():
        print(
            f"Hop={hop} | count={b['count']} | "
            f"doc_any={b['doc_any_recall']:.4f} | doc_all={b['doc_all_recall']:.4f} | "
            f"sent_any={b['sent_any_recall']:.4f} | sent_all={b['sent_all_recall']:.4f} | "
            f"doc_f1={b['doc_f1']:.4f} | sent_f1={b['sent_f1']:.4f}"
        )

    if args.out_summary:
        with open(args.out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.out_per_claim:
        with open(args.out_per_claim, "w", encoding="utf-8") as f:
            for row in per_claim:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
