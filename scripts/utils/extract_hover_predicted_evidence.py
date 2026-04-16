#!/usr/bin/env python3
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

try:
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except ModuleNotFoundError:
    PunktSentenceTokenizer = None


MAPPED_DRIVE_HINT = (
    'If this file is on a mapped WebDAV drive such as Y:, Python may fail to open very large JSON files '
    'with OSError [Errno 22]. In that case, rerun retrieval with a local --out_path or copy the result '
    'to a local disk before running this script.'
)


def normalize_ws(text: str) -> str:
    text = (text or '').replace('\u00a0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def sentence_split(text: str) -> List[str]:
    text = text or ''
    if PunktSentenceTokenizer is not None:
        tokenizer = PunktSentenceTokenizer()
        return [normalize_ws(sent) for sent in tokenizer.tokenize(text) if normalize_ws(sent)]
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [normalize_ws(sent) for sent in parts if normalize_ws(sent)]



def base_pieces(text: str) -> List[str]:
    pieces = []
    for sent in sentence_split(text):
        if len(sent) > 10:
            pieces.append(sent)
    return pieces



def choose_spans(pieces: List[str], selected_norms: set, max_span: int) -> Dict[int, Tuple[int, str]]:
    candidates = []
    exact_set = set(pieces)
    n = len(pieces)

    for start in range(n):
        merged = pieces[start]
        for end in range(start + 1, min(n, start + max_span)):
            merged = normalize_ws(merged + ' ' + pieces[end])
            if merged in selected_norms and merged not in exact_set:
                span_len = end - start + 1
                candidates.append((start, end, merged, span_len))

    candidates.sort(key=lambda item: (-item[3], item[0], item[1]))

    used = set()
    chosen = {}
    for start, end, merged, _span_len in candidates:
        if any(idx in used for idx in range(start, end + 1)):
            continue
        for idx in range(start, end + 1):
            used.add(idx)
        chosen[start] = (end, merged)

    return chosen



def build_doc_units(title: str, text: str, selected_norms: set, max_span: int) -> List[str]:
    pieces = base_pieces(text)
    spans = choose_spans(pieces, selected_norms=selected_norms, max_span=max_span)

    units = []
    if title:
        units.append(normalize_ws(title))

    idx = 0
    while idx < len(pieces):
        if idx in spans:
            end, merged = spans[idx]
            units.append(merged)
            idx = end + 1
            continue
        units.append(pieces[idx])
        idx += 1

    return units



def build_node_bucket(raw_item: dict, selected_norms: set, max_span: int):
    bucket = defaultdict(deque)

    for doc in raw_item.get('retrieved_docs', []):
        title = doc.get('docid', '')
        units = build_doc_units(
            title=title,
            text=doc.get('text', ''),
            selected_norms=selected_norms,
            max_span=max_span,
        )
        for sent_idx, sent in enumerate(units):
            bucket[normalize_ws(sent)].append({
                'title': title,
                'index': sent_idx,
                'sentences': sent,
            })

    return bucket



def export_one(pred_item: dict, raw_item: dict, max_span: int, drop_title_nodes: bool):
    selected_texts = [normalize_ws(text) for text in pred_item.get('top_evidence_texts', []) if normalize_ws(text)]
    selected_norms = set(selected_texts)
    bucket = build_node_bucket(raw_item, selected_norms=selected_norms, max_span=max_span)

    mapped = []
    unmatched = 0
    title_only = 0

    for text in selected_texts:
        if bucket[text]:
            node = bucket[text].popleft()
            if drop_title_nodes and normalize_ws(node['sentences']) == normalize_ws(node['title']):
                title_only += 1
                continue
            mapped.append(node)
        else:
            unmatched += 1

    pred_doc_titles = list(dict.fromkeys(item['title'] for item in mapped))
    result = {
        'id': pred_item['id'],
        'claim': pred_item['claim'],
        'label': raw_item.get('label'),
        'num_hops': raw_item.get('num_hops'),
        'pred_evidence_list': mapped,
        'pred_doc_titles': pred_doc_titles,
        'pred_sent_keys': [f"{item['title']}@@{item['index']}" for item in mapped],
        'pred_sent_texts': [item['sentences'] for item in mapped],
    }
    return result, len(selected_texts), unmatched, title_only



def load_json(path: Path, label: str):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except OSError as exc:
        raise RuntimeError(f'Failed to open {label}: {path}. {exc}. {MAPPED_DRIVE_HINT}') from exc



def normalize_label(label):
    if isinstance(label, str):
        return label.upper()
    return label



def build_text_only_pred_items(retrieval_data: List[dict], topk: int) -> List[dict]:
    pred_items = []
    for item in retrieval_data:
        texts = []
        for ev in item.get('top_evidences', []) or []:
            text = normalize_ws(ev.get('text', ''))
            if text:
                texts.append(text)
            if topk > 0 and len(texts) >= topk:
                break
        pred_items.append({
            'id': item['id'],
            'claim': item.get('claim', ''),
            'top_evidence_texts': texts,
            'num_hops': item.get('num_hops'),
            'label': item.get('label'),
        })
    return pred_items



def extract_predicted_evidence(
    retrieval_data: List[dict],
    raw_data: List[dict],
    topk: int,
    max_span: int,
    keep_title_nodes: bool,
) -> Tuple[List[dict], Dict[str, int]]:
    pred_items = build_text_only_pred_items(retrieval_data, topk=topk)
    id2raw = {item['id']: item for item in raw_data}

    results = []
    stats = {
        'examples': 0,
        'selected_texts': 0,
        'unmatched': 0,
        'title_only_dropped': 0,
        'empty_examples': 0,
        'missing_raw_examples': 0,
    }

    for pred_item in pred_items:
        raw_item = id2raw.get(pred_item['id'])
        if raw_item is None:
            stats['missing_raw_examples'] += 1
            continue

        exported, selected_count, unmatched_count, title_only_count = export_one(
            pred_item=pred_item,
            raw_item=raw_item,
            max_span=max_span,
            drop_title_nodes=not keep_title_nodes,
        )
        exported['label'] = normalize_label(exported.get('label', pred_item.get('label')))
        if exported.get('num_hops') is None:
            exported['num_hops'] = pred_item.get('num_hops')
        results.append(exported)

        stats['examples'] += 1
        stats['selected_texts'] += selected_count
        stats['unmatched'] += unmatched_count
        stats['title_only_dropped'] += title_only_count
        if not exported['pred_evidence_list']:
            stats['empty_examples'] += 1

    return results, stats



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000.json',
                        help='Raw retrieval output, e.g. data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json')
    parser.add_argument('--raw_path', type=str, default='data/plan1/bm25_dev.json',
                        help='Raw retrieved-doc file containing retrieved_docs and labels')
    parser.add_argument('--output_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000_pred_evidence.json',
                        help='Output path for extracted predicted evidence')
    parser.add_argument('--topk', type=int, default=0,
                        help='0 means keep all top_evidences from the raw retrieval file')
    parser.add_argument('--max_span', type=int, default=8,
                        help='Maximum merged span length when matching text back to retrieved docs')
    parser.add_argument('--keep_title_nodes', action='store_true',
                        help='Keep title-only nodes where the sentence equals the document title')
    parser.add_argument('--stats_path', type=str, default='data/[PLAN]/nodefc_decomposition_aware_dev_0_4000_pred_evidence_stats.json',
                        help='Optional path to save extraction stats as JSON')
    parser.add_argument('--plan', type=str, default='plan4.3',)
    args = parser.parse_args()

    retrieval_path = args.retrieval_path.replace('[PLAN]', args.plan)
    raw_path = args.raw_path.replace('[PLAN]', args.plan)
    output_path = args.output_path.replace('[PLAN]', args.plan)
    stats_path = args.stats_path.replace('[PLAN]', args.plan)

    with open(retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    results, stats = extract_predicted_evidence(
        retrieval_data=retrieval_data,
        raw_data=raw_data,
        topk=args.topk,
        max_span=args.max_span,
        keep_title_nodes=args.keep_title_nodes,
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f'saved_path={output_path}')

if __name__ == '__main__':
    main()
