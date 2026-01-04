import argparse
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_ent(s: str) -> str:
    s = norm_text(s).lower()
    s = s.strip(" '\"“”‘’")
    return s

def norm_rel(s: str) -> str:
    # relation 一般是短语，做轻量归一化
    s = norm_text(s).lower()
    s = s.strip(" '\"“”‘’")
    return s

BAD_SUBSTR = ["<triplet>", "<subj>", "<obj>", "<pad>"]

def is_bad_span(x: str) -> bool:
    if not x:
        return True
    xl = x.lower()
    if any(b in xl for b in BAD_SUBSTR):
        return True
    return False

def parse_rebel_output(text):
    text = text.replace("<s>", "").replace("</s>", "").strip()
    triples = []
    
    TRIPLE = "<triplet>"
    SUBJ = "<subj>"
    OBJ = "<obj>"
    
    parts = text.split(TRIPLE)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 期望格式：head <subj> relation <obj> tail
        if SUBJ in p and OBJ in p:
            head, rest = p.split(SUBJ, 1)
            relation, tail = rest.split(OBJ, 1)
            head = norm_text(head)
            relation = norm_text(relation)
            tail = norm_text(tail)
            if head and relation and tail:
                triples.append((head, relation, tail))
    return triples

def extract_triples_from_rebel(text):
    triplets = []
    subject, relation, object_ = "", "", ""
    current = "x"
    
    text = text.strip().replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    for token in text.split():
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            subject, relation, object_ = "", "", ""
            current = "head"
        elif token == "<subj>":
            current = "tail"
        elif token == "<obj>":
            current = "rel"
        else:
            if current == "head":
                subject += " " + token
            elif current == "tail":
                object_ += " " + token
            elif current == "rel":
                relation += " " + token
    
    if subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets

class RebelExtractor:
    def __init__(self, model_name, device, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.max_len = max_len
    
    @torch.no_grad()
    def extract_batch(self, sentences, num_beams, max_new_tokens):
        inputs = self.tokenizer(
            sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
        
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        # return [parse_rebel_output(output) for output in decoded_outputs]
        return [extract_triples_from_rebel(output) for output in decoded_outputs]
    
def main(args):
    in_path = args.in_path.replace('[SPLIT]', args.split)
    with open(in_path, 'r') as f:
        dataset = json.load(f)
    print("Loaded dataset with {} samples.".format(len(dataset)))
    
    extractor = RebelExtractor(args.model, device='cuda' if torch.cuda.is_available() else 'cpu', max_len=args.max_length)
    print("Initialized REBEL extractor.")
    
    nodes_list = []
    edges_list = []
    for data in tqdm(dataset):
        sent_nodes = data['sent_nodes']
        
        # 图节点容器
        ent_key_to_eid = {}
        rel_key_to_rid = {}
        entity_nodes = []
        relation_nodes = []
        # 边容器
        sn_edges = []
        sr_edges = []
        nrn_edges = []
        
        # 用于构造局部图，所以放在每条数据内
        def get_or_create_entity(name):
            key = norm_ent(name)
            if not key:
                return ""
            if key in ent_key_to_eid:
                return ent_key_to_eid[key]
            eid = f"e{len(entity_nodes)}"
            ent_key_to_eid[key] = eid
            entity_nodes.append({"eid": eid, "name": name, "norm": key})
            return eid
        
        def get_or_create_relation(name):
            key = norm_rel(name)
            if not key:
                return ""
            if key in rel_key_to_rid:
                return rel_key_to_rid[key]
            rid = f"r{len(relation_nodes)}"
            rel_key_to_rid[key] = rid
            relation_nodes.append({"rid": rid, "name": name, "norm": key})
            return rid
        
        # 批量抽三元组
        texts = []
        sids = []
        for s in sent_nodes:
            sid = s['sid']
            text = norm_text(s['sentences'])
            if not text:
                continue
            texts.append(text)
            sids.append(sid)
        
        triples_by_sent = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i+args.batch_size]
            triples_by_sent.extend(
                extractor.extract_batch(
                    batch,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens
                )
            )
        
        # 建图：从三元组构建N/R/边
        sn_set = set()
        sr_set = set()
        nrn_set = set()
        for sid, triples in zip(sids, triples_by_sent):
            for h, r, t in triples:
                if is_bad_span(h) or is_bad_span(r) or is_bad_span(t):
                    continue
                eid_h = get_or_create_entity(h)
                eid_t = get_or_create_entity(t)
                rid = get_or_create_relation(r)
                if not eid_h or not eid_t or not rid:
                    continue
                
                # S-N
                if (sid, eid_h) not in sn_set:
                    sn_edges.append({"sid": sid, "eid": eid_h})
                    sn_set.add((sid, eid_h))
                if (sid, eid_t) not in sn_set:
                    sn_edges.append({"sid": sid, "eid": eid_t})
                    sn_set.add((sid, eid_t))
                
                # S-R
                if (sid, rid) not in sr_set:
                    sr_edges.append({"sid": sid, "rid": rid})
                    sr_set.add((sid, rid))
                
                # N-R-N
                key = (eid_h, rid, eid_t)
                if key not in nrn_set:
                    nrn_edges.append({"head_eid": eid_h, "rid": rid, "tail_eid": eid_t})
                    nrn_set.add(key)
        
        nodes_list.append({
            'id': data['id'],
            'sent_nodes': sent_nodes,
            'entity_nodes': entity_nodes,
            'relation_nodes': relation_nodes,
        })
        edges_list.append({
            'id': data['id'],
            'sn_edges': sn_edges,
            'sr_edges': sr_edges,
            'nrn_edges': nrn_edges,
        })
    
    out_nodes_path = args.out_nodes_path.replace('[SPLIT]', args.split)
    with open(out_nodes_path, 'w') as f:
        json.dump(nodes_list, f, indent=4)

    out_edges_path = args.out_edges_path.replace('[SPLIT]', args.split)
    with open(out_edges_path, 'w') as f:
        json.dump(edges_list, f, indent=4)
    
    print("Finished building entity and relation graphs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/bm25_sentnodes_[SPLIT].json')
    parser.add_argument('--out_nodes_path', type=str, default='./data/bm25_nodes_[SPLIT].json')
    parser.add_argument('--out_edges_path', type=str, default='./data/bm25_edges_[SPLIT].json')
    parser.add_argument('--split', type=str, default='dev')

    parser.add_argument('--model', type=str, default='Babelscape/rebel-large')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()
    main(args)
