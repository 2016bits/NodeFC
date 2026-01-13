import json
import torch
from torch.utils.data import TensorDataset, Dataset

two_class_label_dict = {
    "SUPPORTS": 0,
    "REFUTES": 1,
}
def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    data_list = []
    for data in dataset:
        claim = data['claim']
        retrieved_evidences = ' '.join(data['retrieved_evidence'])
        ce_pair = f"[CLS] {claim} [SEP] {retrieved_evidences} [SEP]"
        
        gold_label = data['label']
        if not isinstance(gold_label, int):
            gold_label = two_class_label_dict[gold_label]
        
        data_list.append({
            'id': data['id'],
            'ce_pair': ce_pair,
            'gold_label': gold_label,
            'num_hops': data['num_hops'],
            'gold_evidence': data['gold_evidence'],
        })
    return data_list

def batch_ce_data(data_raw, tokenizer, max_len):
    ids = []
    msks = []
    labels = []
    hops = []
    
    for data in data_raw:
        encoded_dict = tokenizer.encode_plus(
            data['ce_pair'],
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        ids.append(encoded_dict['input_ids'])
        msks.append(encoded_dict['attention_mask'])
        labels.append(data['gold_label'])
        hops.append(int(data['num_hops']))
    
    ids = torch.cat(ids, dim=0)
    msks = torch.cat(msks, dim=0)
    labels = torch.tensor(labels)
    hops = torch.tensor(hops, dtype=torch.long)

    batched_dataset = TensorDataset(ids, msks, labels, hops)
    return batched_dataset

def looks_like_title(s: str) -> bool:
    """
    你 retrieved_evidence 里混了 doc title + sentences。
    这个启发式用于过滤标题项，避免把标题噪声喂给 verifier。
    """
    if s is None:
        return False
    t = s.strip()
    if len(t) == 0:
        return False
    # 标题通常短、且不含句号
    if len(t) <= 80 and "." not in t and "?" not in t and "!" not in t:
        return True
    return False

def extract_evidence_sents(ex, max_ev: int, keep_titles: bool):
    ev = ex.get("retrieved_evidence", [])
    if not isinstance(ev, list):
        return []
    out = []
    for x in ev:
        if not isinstance(x, str):
            continue
        if (not keep_titles) and looks_like_title(x):
            continue
        out.append(x)
        if len(out) >= max_ev:
            break
    return out


def mil_collate_fn(batch, tokenizer, max_len: int):
    """
    Return:
      input_ids: [B, M, L]
      attn_mask: [B, M, L]
      labels: [B]
      hops: [B]
      meta: list[dict] (ids, claim, etc.)
    Note: M can vary per batch item; we pad M to max_M in this batch.
    """
    labels = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
    hops = torch.tensor([b["hop"] for b in batch], dtype=torch.long)
    meta = [{"id": b["id"], "claim": b["claim"]} for b in batch]

    max_M = max(len(b["evids"]) for b in batch)

    # flatten (claim, evid) pairs
    flat_text_a, flat_text_b = [], []
    for b in batch:
        claim = b["claim"]
        evids = b["evids"]
        # pad evid list to max_M by repeating last (or empty)
        if len(evids) < max_M:
            evids = evids + [evids[-1]] * (max_M - len(evids))
        for e in evids:
            flat_text_a.append(claim)
            flat_text_b.append(e)

    enc = tokenizer(
        flat_text_a,
        flat_text_b,
        truncation=True,
        max_length=max_len,
        padding=True,          # pad to max in this batch
        return_tensors="pt"
    )
    input_ids = enc["input_ids"]          # [B*M, L]
    attn_mask = enc["attention_mask"]     # [B*M, L]

    B = len(batch)
    M = max_M
    L = input_ids.size(1)

    input_ids = input_ids.view(B, M, L)
    attn_mask = attn_mask.view(B, M, L)

    return input_ids, attn_mask, labels, hops, meta

ID2LABEL = {0: "SUPPORTS", 1: "REFUTES"}
LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1}
class MILVerifierDataset(Dataset):
    """
    Each item:
      - pairs: list[str] evidence sentences (len<=max_ev)
      - label_id: 0/1
      - hop: 2/3/4 (int)
      - id: str
      - claim: str
    Tokenization happens in collate_fn for dynamic padding.
    """
    def __init__(self, raw_list, max_ev: int, keep_titles: bool):
        self.items = []
        for ex in raw_list:
            claim = ex.get("claim", None)
            label = ex.get("label", None)
            hop = ex.get("num_hops", None)
            ex_id = ex.get("id", "")

            if claim is None or label is None or hop is None:
                continue
            if label not in LABEL2ID:
                continue
            try:
                hop = int(hop)
            except Exception:
                continue

            evid_sents = extract_evidence_sents(ex, max_ev=max_ev, keep_titles=keep_titles)
            if len(evid_sents) == 0:
                # 没证据就跳过（也可以保留，做空证据的对比）
                continue

            self.items.append({
                "id": ex_id,
                "claim": claim,
                "evids": evid_sents,
                "label_id": LABEL2ID[label],
                "hop": hop
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]
