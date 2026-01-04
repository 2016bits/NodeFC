import argparse
import json
import spacy
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import math
import torch

def norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_ent(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    return s

def extract_numbers(text: str):
    # capture years, integers, decimals, percentages
    nums = re.findall(r"\b\d{1,4}(?:\.\d+)?%?\b", text)
    return set(nums)

def extract_keywords_simple(text: str):
    # very simple keyword extraction: non-stopword alphabetic tokens length>=3
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    out = set()
    for t in toks:
        if t in EN_STOP_WORDS:
            continue
        if len(t) < 3:
            continue
        out.add(t)
    return out

def semantic_entry_s(biencoder, claim, sent_nodes, topk=20):
    sids, texts = [], []
    for s in sent_nodes:
        txt = s['sentences']
        if not txt:
            continue
        sids.append(s['sid'])
        texts.append(txt)
        
    if not texts:
        return {}
    
    c_vec = biencoder.encode([claim], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    s_vecs = biencoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores = np.dot(s_vecs, c_vec.T).flatten()
    topk_indices = np.argsort(scores)[::-1][:topk]
    return {sids[i]: scores[i] for i in topk_indices}

def entity_entry_n(nlp, claim, entity_nodes):
    norm2eid = {}
    for n in entity_nodes:
        norm = n['norm']
        if norm and norm not in norm2eid:
            norm2eid[norm] = n['eid']
    
    matched = {}
    doc = nlp(claim)
    for ent in doc.ents:
        key = norm_ent(ent.text)
        if key in norm2eid:
            matched[norm2eid[key]] = 1.0  # weight can be improved
    
    claim_norm = norm_ent(claim)
    for norm, eid in norm2eid.items():
        if not norm:
            continue
        if norm in claim_norm or claim_norm in norm:
            matched[eid] = max(matched.get(eid, 0.0), 0.7)
    return matched

def make_personalization(entry_s, entry_n, w_s=0.6, w_n=0.4):
    """
    Build personalization vector p over heterogeneous nodes.
    We use prefixed ids to avoid clashes: S::sid, N::eid, R::rid
    """
    p = defaultdict(float)
    for sid, sc in entry_s.items():
        p[f"S::{sid}"] += w_s * max(sc, 0.0)
    for eid, sc in entry_n.items():
        p[f"N::{eid}"] += w_n * max(sc, 0.0)

    # normalize
    total = sum(p.values())
    if total <= 0:
        return {}
    for k in list(p.keys()):
        p[k] /= total
    return dict(p)

def add_edge(graph, src, dst, weight):
    if weight <= 0:
        return
    graph[src].append((dst, weight))

def build_hetero_graph(sent_nodes, entity_nodes, relation_nodes,
                       sn_edges, sr_edges, nrn_edges,
                       semantic_edges=[],
                       w_sn=1.0, w_sr=0.6, w_nrn=1.0, w_ss=0.5,
                       min_sen_sim=0.25):
    graph = defaultdict(list)
    
    # S-N edges
    for edge in sn_edges:
        u = f"S::{edge['sid']}"
        v = f"N::{edge['eid']}"
        add_edge(graph, u, v, w_sn)
        add_edge(graph, v, u, w_sn)
    
    # S-R edges
    for edge in sr_edges:
        u = f"S::{edge['sid']}"
        v = f"R::{edge['rid']}"
        add_edge(graph, u, v, w_sr)
        add_edge(graph, v, u, w_sr)
    
    # N-R-N edges
    for edge in nrn_edges:
        h = f"N::{edge['head_eid']}"
        t = f"N::{edge['tail_eid']}"
        r = f"R::{edge['rid']}"
        add_edge(graph, h, r, w_nrn)
        add_edge(graph, r, h, w_nrn)
        add_edge(graph, t, r, w_nrn)
        add_edge(graph, r, t, w_nrn)
    
    # Semantic S-S edges
    if semantic_edges:
        for edge in semantic_edges:
            sim = float(edge.get('sim', 0.0))
            if sim < min_sen_sim:
                continue
            u = f"S::{edge['sid1']}"
            v = f"S::{edge['sid2']}"
            w = w_ss * sim
            add_edge(graph, u, v, w)
            add_edge(graph, v, u, w)
    
    return graph

def ppr(graph, p, alpha=0.5, max_iter=20):
    """
    score_{t+1} = alpha * p + (1-alpha) * W^T * score_t
    where W is row-normalized by outgoing weight sum.
    """
    if not p:
        return {}
    
    out_sum = {u: sum(w for _, w in nbrs) for u, nbrs in graph.items()}
    
    cur = defaultdict(float, p)
    for _ in range(max_iter):
        nxt = defaultdict(float)
        
        # restart
        for u, pv in p.items():
            nxt[u] += alpha * pv
        
        # propagation
        for u, score_u in cur.items():
            denom = out_sum.get(u, 1e-10)
            if denom <= 0:
                continue
            for v, w in graph[u]:
                nxt[v] += (1 - alpha) * score_u * (w / denom)
        
        cur = nxt
    
    return dict(cur)

def select_candidates(sent_nodes, scores, topk=5):
    sid2meta = {}
    for s in sent_nodes:
        txt = s['sentences']
        if not txt:
            continue
        sid2meta[s['sid']] = (txt, int(s.get('doc_rank', 10**9)))
        
    ranked = sorted(
        sid2meta.keys(),
        key=lambda sid: scores.get(f"S::{sid}", 0.0),
        reverse=True
    )[:topk]
    
    out = []
    for sid in ranked:
        txt, doc_rank = sid2meta[sid]
        out.append((
            sid,
            float(scores.get(f"S::{sid}", 0.0)),
            txt,
            doc_rank
        ))
    
    return out
    

def rerank_cross_encoder(crossencoder, claim, candidates, beta_graph=0.4):
    pairs = [(claim, c[2]) for c in candidates]
    ce_scores = crossencoder.predict(pairs)
    
    out = []
    for (sid, gscore, txt, doc_rank), ce_score in zip(candidates, ce_scores):
        final_score = beta_graph * gscore + (1 - beta_graph) * float(ce_score)
        out.append({
            "sid": sid,
            "graph_score": gscore,
            "ce_score": float(ce_score),
            "final_score": final_score,
            "doc_rank": doc_rank,
            "text": txt
        })
    
    out = sorted(out, key=lambda x: x['final_score'], reverse=True)
    return out

def _get_contra_entail_label_ids(nli_model):
    """
    兼容不同 MNLI 模型的 label 命名：
    - 'CONTRADICTION'/'ENTAILMENT'/'NEUTRAL'
    - 'contradiction'/'entailment'/'neutral'
    - 有的模型是 'LABEL_0'...（这种就按常见顺序猜）
    """
    id2label = getattr(nli_model.config, "id2label", None) or {}
    # normalize
    norm = {i: str(l).lower() for i, l in id2label.items()}

    contra_id = None
    entail_id = None

    for i, l in norm.items():
        if "contrad" in l:
            contra_id = i
        elif "entail" in l:
            entail_id = i

    # fallback: 对 RoBERTa-MNLI 等常见模型，通常是 [contradiction, neutral, entailment] => [0,1,2]
    if contra_id is None or entail_id is None:
        # best-effort guess
        contra_id = 0
        entail_id = 2

    return int(contra_id), int(entail_id)

@torch.no_grad()
def nli_score_pairs(nli_tok, nli_model, claim: str, candidates: list, device: str = "cuda", batch_size: int = 32):
    """
    candidates: list of dicts, each has ['text', ...]
    returns: list of (p_contra, p_entail)
    """
    contra_id, entail_id = _get_contra_entail_label_ids(nli_model)

    texts = [c["text"] for c in candidates]
    scores = []

    for st in range(0, len(texts), batch_size):
        ed = min(len(texts), st + batch_size)
        batch_texts = texts[st:ed]

        # premise = evidence sentence, hypothesis = claim
        enc = nli_tok(
            batch_texts,
            [claim] * len(batch_texts),
            padding=True,
            truncation=True,
            max_length=256,   # NLI 不用 512，够用且更快
            return_tensors="pt"
        ).to(device)

        out = nli_model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)

        p_contra = probs[:, contra_id].detach().cpu().numpy().tolist()
        p_entail = probs[:, entail_id].detach().cpu().numpy().tolist()
        scores.extend(list(zip(p_contra, p_entail)))

    return scores

def rerank_by_nli(claim: str, reranked: list, nli_tok, nli_model,
                 nli_topk: int = 20,
                 w_contra: float = 0.7,
                 w_entail: float = 0.3,
                 w_rel: float = 0.2,
                 device: str = "cuda"):
    """
    对 reranked 的 top-nli_topk 计算 NLI 分数，并重排：
    - 主排序：contradiction 概率（高优先）
    - 次排序：final_score / ce_score（避免不相关矛盾句占满）
    同时把 nli 分数写回每个 dict。
    """
    if not reranked:
        return reranked

    head = reranked[:nli_topk]
    tail = reranked[nli_topk:]

    nli_scores = nli_score_pairs(nli_tok, nli_model, claim, head, device=device)

    out_head = []
    for x, (p_contra, p_entail) in zip(head, nli_scores):
        x = dict(x)  # copy
        x["p_contra"] = float(p_contra)
        x["p_entail"] = float(p_entail)
        # 可选：做一个融合分数，既鼓励反证，也保留相关性
        rel = float(x.get("final_score", x.get("ce_score", 0.0)))
        x["nli_mix_score"] = w_contra * x["p_contra"] + w_entail * x["p_entail"] + w_rel * rel
        out_head.append(x)

    # 你说“contradict 分高的放靠前”：就用 p_contra 为主排序
    out_head.sort(key=lambda z: (z["p_contra"], z.get("final_score", 0.0)), reverse=True)

    return out_head + tail

def minmax_norm(x, xmin, xmax, eps=1e-12):
    if xmax - xmin < eps:
        return 0.0
    return (x - xmin) / (xmax - xmin)

def build_sid2entnorm(sn_edges, entity_nodes):
    eid2norm = {}
    for n in entity_nodes:
        eid2norm[n['eid']] = n.get('norm', '')
    sid2ents = defaultdict(set)
    for e in sn_edges:
        sid = e['sid']
        eid = e['eid']
        en = eid2norm.get(eid, '')
        if en:
            sid2ents[sid].add(en)
    return sid2ents

def build_semantic_sim_map(semantic_edges, min_sen_sim=0.0):
    # store undirected max similarity for quick lookup
    sim = defaultdict(float)
    if not semantic_edges:
        return sim
    for e in semantic_edges:
        s1 = e.get('sid1')
        s2 = e.get('sid2')
        v = float(e.get('sim', 0.0))
        if v < min_sen_sim:
            continue
        if s1 is None or s2 is None:
            continue
        a, b = (s1, s2) if s1 <= s2 else (s2, s1)
        if v > sim[(a, b)]:
            sim[(a, b)] = v
    return sim

def get_sim(sim_map, sid1, sid2):
    a, b = (sid1, sid2) if sid1 <= sid2 else (sid2, sid1)
    return float(sim_map.get((a, b), 0.0))

def compute_entity_weights_df(candidate_sids, sid2ents):
    # df-based IDF weights inside candidate pool
    N = max(1, len(candidate_sids))
    df = defaultdict(int)
    for sid in candidate_sids:
        for ent in sid2ents.get(sid, set()):
            df[ent] += 1
    w = {}
    for ent, d in df.items():
        # idf-like
        w[ent] = math.log((N + 1.0) / (d + 1.0)) + 1.0
    return w

def diversify_greedy_select(
    claim,
    reranked,               # list of dicts (sid,text,ce_score,final_score,...)
    sid2ents,               # sid -> set(norm_entity)
    semantic_sim_map,       # (sid1,sid2) -> sim
    max_evidence=8,
    alpha=1.0, beta=0.7, gamma=0.8, delta=0.6,
    eta=0.6,
    conn_sem_weight=0.2,    # b in Conn = a*entity + b*semantic
    conn_hard_theta=0.55,   # if no shared ent and max sim < theta -> skip (after first)
    stop_tau=0.05           # stop if best score < tau
):
    if not reranked:
        return []

    # claim targets
    claim_ents = set()
    # If your claim language is not English, this will be weaker; but you already have entity_entry_n anyway.
    # Here we use simple keyword/number coverage which is language-agnostic enough for digits.
    claim_nums = extract_numbers(claim)
    claim_keys = extract_keywords_simple(claim)

    # We will also use entities in candidate sentences as potential claim_ents proxy
    # (optional) If you have a better claim entity set already, pass it in and replace this.
    # For now, leave claim_ents empty -> coverage uses keywords+numbers strongly, entity coverage uses only when claim_ents non-empty.

    candidate_sids = [x["sid"] for x in reranked]
    ent_w = compute_entity_weights_df(candidate_sids, sid2ents)

    # normalize relevance using ce_score (better than final_score for “semantic relevance”)
    ce_vals = np.array([x["ce_score"] for x in reranked], dtype=np.float32)
    ce_min, ce_max = float(ce_vals.min()), float(ce_vals.max())

    # precompute sentence keywords/numbers for coverage
    sid2keys = {}
    sid2nums = {}
    for x in reranked:
        sid2keys[x["sid"]] = extract_keywords_simple(x["text"])
        sid2nums[x["sid"]] = extract_numbers(x["text"])

    selected = []
    selected_sids = set()
    covered_ents = set()
    covered_keys = set()
    covered_nums = set()

    # ---- init: choose best relevance start
    best0 = max(reranked, key=lambda x: (x.get("p_contra", 0.0), x["ce_score"]))
    selected.append(best0)
    selected_sids.add(best0["sid"])
    covered_ents |= set(sid2ents.get(best0["sid"], set()))
    covered_keys |= set(sid2keys.get(best0["sid"], set()))
    covered_nums |= set(sid2nums.get(best0["sid"], set()))

    # ---- greedy iterations
    for _ in range(1, max_evidence):
        best_item = None
        best_score = -1e9

        # union ents in selected for quick shared check
        union_sel_ents = set()
        for it in selected:
            union_sel_ents |= set(sid2ents.get(it["sid"], set()))

        for cand in reranked:
            sid = cand["sid"]
            if sid in selected_sids:
                continue

            # relevance
            R = minmax_norm(cand["ce_score"], ce_min, ce_max)

            # connectivity (entity + semantic)
            shared = set(sid2ents.get(sid, set())) & union_sel_ents
            conn_ent = sum(ent_w.get(e, 0.0) for e in shared)

            conn_sem = 0.0
            for it in selected:
                conn_sem = max(conn_sem, get_sim(semantic_sim_map, sid, it["sid"]))

            # normalize conn_ent roughly by log-scale (avoid huge)
            Conn = math.tanh(conn_ent) + conn_sem_weight * conn_sem
            # clip into [0,1+]
            Conn = max(0.0, float(Conn))

            # hard constraint after first: must connect by entity or semantic
            if len(selected) > 0 and (conn_ent <= 0.0 and conn_sem < conn_hard_theta):
                continue

            # new coverage
            # entity coverage only meaningful if you have claim_ents; otherwise skip entity part
            dEnt = set()
            if claim_ents:
                dEnt = (set(sid2ents.get(sid, set())) & claim_ents) - covered_ents
            dKey = (sid2keys.get(sid, set()) & claim_keys) - covered_keys
            dNum = (sid2nums.get(sid, set()) & claim_nums) - covered_nums

            cov_ent = sum(ent_w.get(e, 0.0) for e in dEnt) if dEnt else 0.0
            cov = cov_ent + 0.6 * len(dKey) + 1.2 * len(dNum)
            Cov = math.tanh(cov)  # normalize

            # redundancy: max semantic similarity to selected
            Red = 0.0
            for it in selected:
                Red = max(Red, get_sim(semantic_sim_map, sid, it["sid"]))

            contra = float(cand.get("p_contra", 0.0))
            score = alpha * R + beta * Conn + gamma * Cov - delta * Red + eta * contra

            if score > best_score:
                best_score = score
                best_item = cand

        if best_item is None or best_score < stop_tau:
            break

        # add best
        sid = best_item["sid"]
        selected.append(best_item)
        selected_sids.add(sid)
        covered_ents |= set(sid2ents.get(sid, set()))
        covered_keys |= set(sid2keys.get(sid, set()))
        covered_nums |= set(sid2nums.get(sid, set()))

    return selected

def main(args):
    in_path = args.in_path.replace('[SPLIT]', args.split)
    with open(in_path, 'r') as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} data samples.")
    
    nodes_path = args.nodes_path.replace('[SPLIT]', args.split)
    with open(nodes_path, 'r') as f:
        nodes = json.load(f)

    edges_path = args.edges_path.replace('[SPLIT]', args.split)
    with open(edges_path, 'r') as f:
        edges = json.load(f)
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges.")
    
    semantic_edges_path = args.semantic_edges_path.replace('[SPLIT]', args.split)
    with open(semantic_edges_path, 'r') as f:
        semantic_edges = json.load(f)
    print(f"Loaded {len(semantic_edges)} semantic edges.")
    
    id2data = {data['id']: data for data in data_list}
    id2nodes = {node['id']: node for node in nodes}
    id2edges = {edge['id']: edge for edge in edges}
    id2semantics = {snode['id']: snode['semantic_edges'] for snode in semantic_edges}
    
    # models
    biencoder = SentenceTransformer(args.embedding_model)
    crossencoder = CrossEncoder(args.cross_encoder_model)
    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model, use_fast=True)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)
    nli_model.eval()
    nli_model.to(args.device)
    nlp = spacy.load(args.spacy_model, disable=["tagger", "parser", "lemmatizer"])
    print("Initialized models.")
    
    results = []
    keys = list(id2nodes.keys())
    for ex_id in tqdm(keys):
        node = id2nodes[ex_id]
        edge = id2edges[ex_id]
        
        data = id2data[ex_id]
        claim = data['claim']
        
        # nodes and edges
        sent_nodes = node['sent_nodes']
        entity_nodes = node['entity_nodes']
        relation_nodes = node['relation_nodes']
        
        sn_edges = edge['sn_edges']
        sr_edges = edge['sr_edges']
        nrn_edges = edge['nrn_edges']
        
        semantic_edges = id2semantics.get(ex_id, [])
        
        # entry
        entry_s = semantic_entry_s(biencoder, claim, sent_nodes, topk=args.entry_k_s)
        entry_n = entity_entry_n(nlp, claim, entity_nodes)
        
        p = make_personalization(entry_s, entry_n, w_s=args.w_entry_s, w_n=args.w_entry_n)
        
        # graph
        graph = build_hetero_graph(sent_nodes, entity_nodes, relation_nodes,
                                   sn_edges, sr_edges, nrn_edges,
                                   semantic_edges=semantic_edges,
                                   w_sn=args.w_sn, w_sr=args.w_sr, w_nrn=args.w_nrn, w_ss=args.w_ss,
                                   min_sen_sim=args.min_sen_sim
                                   )
        
        # PPR
        scores = ppr(graph, p, alpha=args.ppr_alpha, max_iter=args.ppr_max_iter)
        
        # candidates
        cnadidates = select_candidates(sent_nodes, scores, topk=args.candidate_k)
        
        # rerank
        reranked = rerank_cross_encoder(crossencoder, claim, cnadidates, beta_graph=args.beta_graph)
        
        # NLI rerank top-20: contradict high first
        reranked = rerank_by_nli(
            claim=claim,
            reranked=reranked,
            nli_tok=nli_tokenizer,
            nli_model=nli_model,
            nli_topk=args.nli_topk,
            w_contra=args.nli_w_contra,
            w_entail=args.nli_w_entail,
            w_rel=args.nli_w_rel,
            device=args.device
        )
        
        # diversify / complementary selection
        sid2ents = build_sid2entnorm(sn_edges, entity_nodes)
        sim_map = build_semantic_sim_map(semantic_edges, min_sen_sim=args.min_sen_sim)
        
        diversified = diversify_greedy_select(
            claim=claim,
            reranked=reranked,
            sid2ents=sid2ents,
            semantic_sim_map=sim_map,
            max_evidence=args.max_evidence,
            alpha=args.alpha_rel,
            beta=args.beta_conn,
            gamma=args.gamma_cov,
            delta=args.delta_red,
            eta=args.nli_eta,
            conn_sem_weight=args.conn_sem_weight,
            conn_hard_theta=args.conn_hard_theta,
            stop_tau=args.stop_tau
        )
        
        results.append({
            "id": ex_id,
            "claim": claim,
            "entry_sids": list(entry_s.keys()),
            "entry_nids": list(entry_n.keys()),
            "top_evidences": diversified,
            "reranked_candidates": reranked
        })
    
    out_path = args.out_path.replace('[SPLIT]', args.split)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_path}.")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/bm25_[SPLIT].json')
    parser.add_argument('--nodes_path', type=str, default='./data/bm25_nodes_[SPLIT].json')
    parser.add_argument('--edges_path', type=str, default='./data/bm25_edges_[SPLIT].json')
    parser.add_argument('--semantic_edges_path', type=str, default='./data/bm25_semantic_edges_[SPLIT].json')
    parser.add_argument('--split', type=str, default='dev')
    
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--cross_encoder_model', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm')
    
    # entry
    parser.add_argument('--entry_k_s', type=int, default=30)
    parser.add_argument('--w_entry_s', type=float, default=0.6)
    parser.add_argument('--w_entry_n', type=float, default=0.4)
    
    # semantic edges
    parser.add_argument('--min_sen_sim', type=float, default=0.25)
    
    # edge weights
    parser.add_argument('--w_sn', type=float, default=1.0)
    parser.add_argument('--w_sr', type=float, default=0.6)
    parser.add_argument('--w_nrn', type=float, default=1.0)
    parser.add_argument('--w_ss', type=float, default=0.6)
    
    # PPR
    parser.add_argument('--ppr_alpha', type=float, default=0.7)
    parser.add_argument('--ppr_max_iter', type=int, default=25)
    
    # candidates & rerank
    parser.add_argument('--candidate_k', type=int, default=50)
    parser.add_argument('--beta_graph', type=float, default=0.2)
    
    parser.add_argument('--max_evidence', type=int, default=5)

    parser.add_argument('--alpha_rel', type=float, default=1.0)   # α
    parser.add_argument('--beta_conn', type=float, default=0.7)   # β
    parser.add_argument('--gamma_cov', type=float, default=0.8)   # γ
    parser.add_argument('--delta_red', type=float, default=0.6)   # δ
    
    # NLI rerank
    parser.add_argument('--nli_model', type=str, default='roberta-large-mnli')
    parser.add_argument('--device', type=str, default='cuda')  # 你之前脚本没 device，这里补上
    parser.add_argument('--nli_topk', type=int, default=20)
    parser.add_argument('--nli_eta', type=float, default=0.6)  # contradiction bonus in greedy select

    # mix score weights (可选)
    parser.add_argument('--nli_w_contra', type=float, default=0.7)
    parser.add_argument('--nli_w_entail', type=float, default=0.3)
    parser.add_argument('--nli_w_rel', type=float, default=0.2)

    parser.add_argument('--conn_sem_weight', type=float, default=0.2)   # semantic part weight in connectivity
    parser.add_argument('--conn_hard_theta', type=float, default=0.55)  # hard connectivity threshold
    parser.add_argument('--stop_tau', type=float, default=0.05)         # stop if best marginal gain is too small
    
    # candidates & rerank
    # parser.add_argument('--candidate_k', type=int, default=50)
    # parser.add_argument('--beta_graph', type=float, default=0.2)
    
    parser.add_argument('--out_path', type=str, default='./data/search_results_greedy_select_contra_rerank_[SPLIT].json')
    args = parser.parse_args()
    main(args)
