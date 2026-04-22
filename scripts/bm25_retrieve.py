import argparse
import json

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher


def normalize_ws(text):
    return " ".join(str(text if text is not None else "").split()).strip()


def flatten_constraint_value(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_ws(value)
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        text = normalize_ws(value)
        return [text] if text else []
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_constraint_value(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(flatten_constraint_value(item))
        return out
    return []


def render_constraint_text(constraint):
    if not isinstance(constraint, dict):
        return ""
    parts = []
    for key in ("time", "date", "year", "quantity", "number", "count", "negation"):
        values = flatten_constraint_value(constraint.get(key))
        if values:
            parts.append(" ".join(values))
    if not parts:
        parts = flatten_constraint_value(constraint)
    seen = set()
    ordered = []
    for part in parts:
        text = normalize_ws(part)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return " ".join(ordered)


def load_decomposition_map(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item.get("id")): item for item in data if isinstance(item, dict) and "id" in item}


def extract_atomic_fact_queries(data, decomposition_map, max_atomic_fact_queries):
    decomposition_item = decomposition_map.get(str(data.get("id"))) or {}
    atomic_facts = []

    if isinstance(data.get("decomposition"), dict):
        atomic_facts = (data.get("decomposition") or {}).get("atomic_facts") or []
    elif isinstance(data.get("atomic_facts"), list):
        atomic_facts = data.get("atomic_facts") or []
    elif isinstance(decomposition_item.get("decomposition"), dict):
        atomic_facts = (decomposition_item.get("decomposition") or {}).get("atomic_facts") or []
    elif isinstance(decomposition_item.get("atomic_facts"), list):
        atomic_facts = decomposition_item.get("atomic_facts") or []

    queries = []
    seen = set()
    ordered_facts = sorted(
        [fact for fact in atomic_facts if isinstance(fact, dict)],
        key=lambda fact: (0 if fact.get("critical") else 1, -len(normalize_ws(fact.get("text", "")))),
    )
    for idx, fact in enumerate(ordered_facts):
        fact_text = normalize_ws(fact.get("text"))
        constraint_text = render_constraint_text(fact.get("constraint") or {})
        query_text = normalize_ws(" ".join(part for part in (fact_text, constraint_text) if part))
        if not query_text or query_text in seen:
            continue
        seen.add(query_text)
        queries.append({
            "fact_id": fact.get("id") or f"atomic_{idx}",
            "text": fact_text or query_text,
            "query": query_text,
            "critical": bool(fact.get("critical")),
        })
        if max_atomic_fact_queries > 0 and len(queries) >= max_atomic_fact_queries:
            break
    return queries


def run_query(searcher, query_text, topk):
    hits = searcher.search(query_text, k=topk)
    results = []
    for rank, hit in enumerate(hits, start=1):
        try:
            doc = json.loads(hit.raw)
        except json.JSONDecodeError:
            doc = {}
        results.append({
            "docid": hit.docid,
            "raw_score": float(hit.score),
            "rank": rank,
            "text": doc.get("contents", ""),
        })
    return results


def resolve_claim_topk(args):
    return args.claim_topk if args.claim_topk > 0 else args.topk


def resolve_atomic_topk(args, claim_topk):
    if args.atomic_topk > 0:
        return args.atomic_topk
    return max(4, max(1, claim_topk // 2))


def resolve_union_topk(args, claim_topk, atomic_query_count, atomic_topk):
    if args.union_topk > 0:
        return args.union_topk
    if atomic_query_count <= 0:
        return claim_topk
    extra = max(atomic_topk, min(claim_topk, atomic_query_count * max(1, atomic_topk // 2)))
    return max(claim_topk, min(claim_topk * 2, claim_topk + extra))


def merge_dual_route_hits(claim_hits, atomic_results, union_topk, args):
    merged = {}

    def get_doc_record(hit):
        item = merged.get(hit["docid"])
        if item is None:
            item = {
                "docid": hit["docid"],
                "text": hit.get("text", ""),
                "score": 0.0,
                "claim_score": None,
                "claim_rank": None,
                "atomic_fact_score": None,
                "best_rank": 10**9,
                "retrieval_routes": set(),
                "matched_atomic_fact_ids": set(),
                "atomic_hit_count": 0,
            }
            merged[hit["docid"]] = item
        if not item["text"] and hit.get("text"):
            item["text"] = hit["text"]
        item["best_rank"] = min(item["best_rank"], int(hit["rank"]))
        return item

    for hit in claim_hits:
        item = get_doc_record(hit)
        item["score"] += float(args.claim_rrf_weight) / (float(args.rrf_k) + float(hit["rank"]))
        item["retrieval_routes"].add("claim")
        item["claim_score"] = float(hit["raw_score"]) if item["claim_score"] is None else max(item["claim_score"], float(hit["raw_score"]))
        item["claim_rank"] = int(hit["rank"]) if item["claim_rank"] is None else min(item["claim_rank"], int(hit["rank"]))

    for atomic_query in atomic_results:
        fact_id = atomic_query["fact_id"]
        for hit in atomic_query["hits"]:
            item = get_doc_record(hit)
            item["score"] += float(args.atomic_rrf_weight) / (float(args.rrf_k) + float(hit["rank"]))
            item["retrieval_routes"].add("atomic_fact")
            item["matched_atomic_fact_ids"].add(str(fact_id))
            item["atomic_hit_count"] += 1
            item["atomic_fact_score"] = float(hit["raw_score"]) if item["atomic_fact_score"] is None else max(item["atomic_fact_score"], float(hit["raw_score"]))

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -float(item["score"]),
            -len(item["retrieval_routes"]),
            -len(item["matched_atomic_fact_ids"]),
            item["claim_rank"] if item["claim_rank"] is not None else 10**9,
            int(item["best_rank"]),
            item["docid"],
        ),
    )

    results = []
    for item in ranked[:union_topk]:
        results.append({
            "docid": item["docid"],
            "score": float(item["score"]),
            "text": item["text"],
            "claim_score": None if item["claim_score"] is None else float(item["claim_score"]),
            "claim_rank": item["claim_rank"],
            "atomic_fact_score": None if item["atomic_fact_score"] is None else float(item["atomic_fact_score"]),
            "atomic_hit_count": int(item["atomic_hit_count"]),
            "matched_atomic_fact_ids": sorted(item["matched_atomic_fact_ids"]),
            "retrieval_routes": sorted(item["retrieval_routes"]),
        })
    return results


def main(args):
    searcher = LuceneSearcher(args.index_path)
    searcher.set_bm25(0.9, 0.4)

    in_path = args.in_path.replace("[SPLIT]", args.split)
    with open(in_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    decomposition_path = args.decomposition_path.replace("[SPLIT]", args.split).replace("[PLAN]", args.plan) if args.decomposition_path else ""
    decomposition_map = load_decomposition_map(decomposition_path) if decomposition_path else {}

    claim_topk = resolve_claim_topk(args)
    atomic_topk = resolve_atomic_topk(args, claim_topk)

    results = []
    for data in tqdm(dataset):
        claim = data["claim"]
        gold_evidence = data.get("gold_evidence_list", data.get("gold_evidence", []))
        label = data.get("label")
        num_hops = data.get("num_hops")

        claim_hits = run_query(searcher, claim, topk=claim_topk)
        atomic_queries = extract_atomic_fact_queries(data, decomposition_map, args.max_atomic_fact_queries)
        atomic_results = []
        for atomic_query in atomic_queries:
            atomic_results.append({
                "fact_id": atomic_query["fact_id"],
                "text": atomic_query["text"],
                "hits": run_query(searcher, atomic_query["query"], topk=atomic_topk),
            })

        union_topk = resolve_union_topk(args, claim_topk, len(atomic_queries), atomic_topk)
        retrieved_docs = merge_dual_route_hits(claim_hits, atomic_results, union_topk, args)

        result = {
            "id": data["id"],
            "claim": claim,
            "gold_evidence": gold_evidence,
            "label": label,
            "num_hops": num_hops,
            "retrieved_docs": retrieved_docs,
            "retrieval_summary": {
                "claim_topk": int(claim_topk),
                "atomic_topk": int(atomic_topk),
                "union_topk": int(union_topk),
                "atomic_query_count": int(len(atomic_queries)),
                "used_atomic_fact_bm25": bool(atomic_queries),
            },
        }
        results.append(result)

    out_path = args.out_path.replace("[SPLIT]", args.split).replace("[PLAN]", args.plan)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Retrieve {len(results)} samples from {in_path} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="/mnt/data/yangjun/data/HOVER/data/converted_data/[SPLIT]_full.json")
    parser.add_argument("--out_path", type=str, default="./data/[PLAN]/bm25_[SPLIT].json")
    parser.add_argument("--index_path", type=str, default="/mnt/data/yangjun/data/HOVER/corpus/index")
    parser.add_argument("--decomposition_path", type=str, default="data/[PLAN]/dev_2_decomposed_0_4000.json")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--claim_topk", type=int, default=0)
    parser.add_argument("--atomic_topk", type=int, default=0)
    parser.add_argument("--union_topk", type=int, default=0)
    parser.add_argument("--max_atomic_fact_queries", type=int, default=0)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--claim_rrf_weight", type=float, default=1.0)
    parser.add_argument("--atomic_rrf_weight", type=float, default=0.9)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--plan", type=str, default="plan5.1")

    args = parser.parse_args()
    main(args)
