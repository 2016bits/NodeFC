#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except ModuleNotFoundError:
    PunktSentenceTokenizer = None


MAPPED_DRIVE_HINT = (
    'If this file is on a mapped WebDAV drive such as Y:, Python may fail to open very large JSON files '
    'with OSError [Errno 22]. In that case, rerun retrieval with a local --out_path or copy the result '
    'to a local disk before running this script.'
)


# -----------------------------
# Normalization / sentence split
# -----------------------------

def normalize_ws(text: str) -> str:
    text = (text or '').replace('\u00a0', ' ')
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def sentence_split(text: str) -> List[str]:
    """
    Split retrieved doc text into natural sentence units.
    IMPORTANT: do NOT prepend title as a pseudo-sentence. Gold indices refer to doc sentences,
    and prepending the title shifts every downstream index by +1.
    """
    text = text or ''
    parts: List[str] = []

    # First, respect explicit line breaks from the raw BM25 dump.
    for line in text.splitlines():
        line = normalize_ws(line)
        if not line:
            continue
        if PunktSentenceTokenizer is not None:
            tokenizer = PunktSentenceTokenizer()
            line_parts = [normalize_ws(s) for s in tokenizer.tokenize(line) if normalize_ws(s)]
        else:
            line_parts = [normalize_ws(s) for s in re.split(r'(?<=[.!?])\s+', line) if normalize_ws(s)]
        if line_parts:
            parts.extend(line_parts)
        else:
            parts.append(line)
    return parts


def text_similarity(a: str, b: str) -> float:
    a_n = normalize_ws(a)
    b_n = normalize_ws(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


# -----------------------------
# Flexible loaders
# -----------------------------

def load_json(path: Path, label: str):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except OSError as exc:
        raise RuntimeError(f'Failed to open {label}: {path}. {exc}. {MAPPED_DRIVE_HINT}') from exc


def ensure_examples(obj: Any) -> List[dict]:
    """
    Accept either:
      - a plain list of examples
      - or a dict wrapper like {"some/file.json": {...single example...}} or {"some/file.json": [..examples..]}
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Common wrapper in pasted samples: {path: payload}
        if len(obj) == 1:
            only_val = next(iter(obj.values()))
            if isinstance(only_val, list):
                return only_val
            if isinstance(only_val, dict) and 'id' in only_val:
                return [only_val]
        # Already keyed by id? Keep values that look like examples.
        vals = [v for v in obj.values() if isinstance(v, dict) and 'id' in v]
        if vals:
            return vals
    raise ValueError('Unsupported JSON shape: expected a list of examples or a single-key wrapper.')


# -----------------------------
# Raw doc indexing / matching
# -----------------------------

def build_raw_doc_index(raw_item: dict) -> Dict[str, Dict[str, Any]]:
    """
    Build per-doc sentence index from raw retrieved_docs.
    Structure:
      docid -> {
        'sentences': [sent0, sent1, ...],
        'norm_to_indices': {norm_sent: [idx1, idx2, ...]}
      }
    """
    doc_index: Dict[str, Dict[str, Any]] = {}
    for doc in raw_item.get('retrieved_docs', []) or []:
        docid = normalize_ws(doc.get('docid', ''))
        if not docid:
            continue
        sents = sentence_split(doc.get('text', ''))
        norm_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, sent in enumerate(sents):
            norm_to_indices[normalize_ws(sent)].append(idx)
        doc_index[docid] = {
            'sentences': sents,
            'norm_to_indices': norm_to_indices,
        }
    return doc_index


def find_sentence_match(
    docid: str,
    text: str,
    doc_index: Dict[str, Dict[str, Any]],
    max_span: int = 4,
    fuzzy_threshold: float = 0.92,
) -> Tuple[Optional[int], Optional[str], str, float]:
    """
    Return (sent_idx, matched_sentence_text, match_type, score)
    match_type in {'exact', 'merged', 'fuzzy', 'unmatched'}
    """
    docid = normalize_ws(docid)
    text_n = normalize_ws(text)
    if not docid or not text_n or docid not in doc_index:
        return None, None, 'unmatched', 0.0

    doc_info = doc_index[docid]
    norm_to_indices = doc_info['norm_to_indices']
    sents = doc_info['sentences']

    # Exact sentence match.
    if text_n in norm_to_indices and norm_to_indices[text_n]:
        idx = norm_to_indices[text_n][0]
        return idx, sents[idx], 'exact', 1.0

    # Merged adjacent span match.
    n = len(sents)
    for start in range(n):
        merged = normalize_ws(sents[start])
        if merged == text_n:
            return start, merged, 'exact', 1.0
        for end in range(start + 1, min(n, start + max_span)):
            merged = normalize_ws(merged + ' ' + sents[end])
            if merged == text_n:
                return start, merged, 'merged', 1.0

    # Fuzzy best match.
    best_idx = None
    best_score = 0.0
    for idx, sent in enumerate(sents):
        score = text_similarity(text_n, sent)
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is not None and best_score >= fuzzy_threshold:
        return best_idx, sents[best_idx], 'fuzzy', best_score

    return None, None, 'unmatched', best_score


# -----------------------------
# Extraction helpers
# -----------------------------

def normalize_label(label):
    if isinstance(label, str):
        return label.upper()
    return label


def dedup_nodes(nodes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for node in nodes:
        key = (normalize_ws(node.get('title')), node.get('index'))
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def dedup_strs(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        x_n = normalize_ws(x)
        if not x_n or x_n in seen:
            continue
        seen.add(x_n)
        out.append(x_n)
    return out


def convert_candidates_to_nodes(
    candidates: List[dict],
    doc_index: Dict[str, Dict[str, Any]],
    max_span: int,
    fuzzy_threshold: float,
    topk: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if topk > 0:
        candidates = candidates[:topk]

    mapped: List[Dict[str, Any]] = []
    stats = {'exact': 0, 'merged': 0, 'fuzzy': 0, 'unmatched': 0}
    for cand in candidates:
        docid = normalize_ws(cand.get('docid', ''))
        text = normalize_ws(cand.get('text', ''))
        if not docid or not text:
            stats['unmatched'] += 1
            continue
        sent_idx, matched_text, match_type, score = find_sentence_match(
            docid=docid,
            text=text,
            doc_index=doc_index,
            max_span=max_span,
            fuzzy_threshold=fuzzy_threshold,
        )
        if sent_idx is None:
            stats['unmatched'] += 1
            continue
        stats[match_type] += 1
        mapped.append({
            'title': docid,
            'index': sent_idx,
            'sentences': matched_text or text,
            'source_text': text,
            'sid': cand.get('sid'),
            'match_type': match_type,
            'match_score': score,
            'support_type': cand.get('support_type'),
            'fact_id': cand.get('fact_id'),
            'fact_role': cand.get('fact_role'),
            'aggregate_score': cand.get('aggregate_score'),
        })
    return dedup_nodes(mapped), stats


def extract_fact_trace_candidates(item: dict) -> List[dict]:
    out: List[dict] = []
    for trace in item.get('fact_traces', []) or []:
        for cand in trace.get('top_candidates', []) or []:
            if isinstance(cand, dict):
                out.append(cand)
    return out


# -----------------------------
# Main extraction
# -----------------------------

def export_one(
    retrieval_item: dict,
    raw_item: dict,
    rerank_topk: int,
    max_span: int,
    fuzzy_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    doc_index = build_raw_doc_index(raw_item)

    # Final selected evidence: use top_evidences directly, not top_evidence_texts.
    final_nodes, final_stats = convert_candidates_to_nodes(
        candidates=list(retrieval_item.get('top_evidences', []) or []),
        doc_index=doc_index,
        max_span=max_span,
        fuzzy_threshold=fuzzy_threshold,
        topk=0,
    )

    reranked_all = list(retrieval_item.get('reranked_candidates', []) or [])
    rerank_nodes_all, rerank_all_stats = convert_candidates_to_nodes(
        candidates=reranked_all,
        doc_index=doc_index,
        max_span=max_span,
        fuzzy_threshold=fuzzy_threshold,
        topk=0,
    )
    rerank_nodes_topk, rerank_topk_stats = convert_candidates_to_nodes(
        candidates=reranked_all,
        doc_index=doc_index,
        max_span=max_span,
        fuzzy_threshold=fuzzy_threshold,
        topk=rerank_topk,
    )

    fact_top_candidates = extract_fact_trace_candidates(retrieval_item)
    fact_candidate_nodes, fact_candidate_stats = convert_candidates_to_nodes(
        candidates=fact_top_candidates,
        doc_index=doc_index,
        max_span=max_span,
        fuzzy_threshold=fuzzy_threshold,
        topk=0,
    )

    # Candidate pool for oracle: use union of global reranked candidates and per-fact top candidates.
    union_candidate_nodes = dedup_nodes(rerank_nodes_all + fact_candidate_nodes)

    pred_doc_titles = dedup_strs(node['title'] for node in final_nodes)
    pred_sent_keys = [f"{node['title']}@@{node['index']}" for node in final_nodes]

    candidate_doc_titles = dedup_strs(node['title'] for node in union_candidate_nodes)
    candidate_sent_keys = [f"{node['title']}@@{node['index']}" for node in union_candidate_nodes]

    rerank_topk_doc_titles = dedup_strs(node['title'] for node in rerank_nodes_topk)
    rerank_topk_sent_keys = [f"{node['title']}@@{node['index']}" for node in rerank_nodes_topk]

    result = {
        'id': retrieval_item['id'],
        'claim': retrieval_item.get('claim', ''),
        'label': normalize_label(raw_item.get('label', retrieval_item.get('label'))),
        'num_hops': raw_item.get('num_hops', retrieval_item.get('num_hops')),

        # Final evidence
        'pred_evidence_list': [
            {'title': n['title'], 'index': n['index'], 'sentences': n['sentences']}
            for n in final_nodes
        ],
        'pred_doc_titles': pred_doc_titles,
        'pred_sent_keys': pred_sent_keys,
        'pred_sent_texts': [n['sentences'] for n in final_nodes],

        # Oracle layers
        'candidate_doc_titles': candidate_doc_titles,
        'candidate_sent_keys': candidate_sent_keys,
        'rerank_topk_doc_titles': rerank_topk_doc_titles,
        'rerank_topk_sent_keys': rerank_topk_sent_keys,

        # Optional debugging info
        'debug_final_match_info': final_nodes,
        'debug_rerank_topk_match_info': rerank_nodes_topk,
    }

    stats = {
        'final': final_stats,
        'rerank_all': rerank_all_stats,
        'rerank_topk': rerank_topk_stats,
        'fact_candidates': fact_candidate_stats,
        'candidate_union_size': len(union_candidate_nodes),
        'final_size': len(final_nodes),
        'rerank_topk_size': len(rerank_nodes_topk),
    }
    return result, stats


def extract_predicted_evidence(
    retrieval_data: List[dict],
    raw_data: List[dict],
    rerank_topk: int,
    max_span: int,
    fuzzy_threshold: float,
) -> Tuple[List[dict], Dict[str, Any]]:
    raw_map = {item['id']: item for item in raw_data}

    results: List[dict] = []
    stats = {
        'examples': 0,
        'missing_raw_examples': 0,
        'final_exact': 0,
        'final_merged': 0,
        'final_fuzzy': 0,
        'final_unmatched': 0,
        'candidate_unmatched_total': 0,
        'rerank_topk_unmatched_total': 0,
        'empty_final_examples': 0,
    }

    for retrieval_item in retrieval_data:
        raw_item = raw_map.get(retrieval_item.get('id'))
        if raw_item is None:
            stats['missing_raw_examples'] += 1
            continue
        exported, one_stats = export_one(
            retrieval_item=retrieval_item,
            raw_item=raw_item,
            rerank_topk=rerank_topk,
            max_span=max_span,
            fuzzy_threshold=fuzzy_threshold,
        )
        results.append(exported)
        stats['examples'] += 1
        stats['final_exact'] += one_stats['final']['exact']
        stats['final_merged'] += one_stats['final']['merged']
        stats['final_fuzzy'] += one_stats['final']['fuzzy']
        stats['final_unmatched'] += one_stats['final']['unmatched']
        stats['candidate_unmatched_total'] += one_stats['fact_candidates']['unmatched'] + one_stats['rerank_all']['unmatched']
        stats['rerank_topk_unmatched_total'] += one_stats['rerank_topk']['unmatched']
        if not exported['pred_evidence_list']:
            stats['empty_final_examples'] += 1

    return results, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000.json')
    parser.add_argument('--raw_path', type=str, default='data/plan1/bm25_dev.json')
    parser.add_argument('--output_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000_pred_evidence.json')
    parser.add_argument('--stats_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000_pred_evidence_stats.json')
    parser.add_argument('--plan', type=str, default='plan4.3')
    parser.add_argument('--rerank_topk', type=int, default=10)
    parser.add_argument('--max_span', type=int, default=4)
    parser.add_argument('--fuzzy_threshold', type=float, default=0.92)
    args = parser.parse_args()

    retrieval_path = Path(args.retrieval_path.replace('[PLAN]', args.plan))
    raw_path = Path(args.raw_path.replace('[PLAN]', args.plan))
    output_path = Path(args.output_path.replace('[PLAN]', args.plan))
    stats_path = Path(args.stats_path.replace('[PLAN]', args.plan))

    retrieval_data = ensure_examples(load_json(retrieval_path, 'retrieval_path'))
    raw_data = ensure_examples(load_json(raw_path, 'raw_path'))

    results, stats = extract_predicted_evidence(
        retrieval_data=retrieval_data,
        raw_data=raw_data,
        rerank_topk=args.rerank_topk,
        max_span=args.max_span,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with stats_path.open('w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f'saved_path={output_path}')
    print(f'stats_path={stats_path}')


if __name__ == '__main__':
    main()
