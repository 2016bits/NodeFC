import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from sentence_transformers import SentenceTransformer

from utils import get_label, l2norm

def _index_by_id(records: List[dict], key="id") -> Dict[str, dict]:
    return {str(r[key]): r for r in records}

def _load_semantic_map(semantic_nodes: List[dict]) -> Dict[str, list]:
    # file format you used: [{"id": ex_id, "semantic_edges": [...]}, ...]
    out = {}
    for r in semantic_nodes:
        out[str(r["id"])] = r.get("semantic_edges", [])
    return out

class ClaimGraphDataset(Dataset):
    """
    Build claim-centric hetero subgraphs using selected evidence sids.
    Node types: sentence, entity
    Edge types: (sentence)-sn-(entity), (entity)-ns-(sentence), (sentence)-ss-(sentence)
    """
    def __init__(
        self,
        data_path: str,
        nodes_path: str,
        edges_path: str,
        semantic_edges_path: str,
        evidence_path: str,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_sentences: int = 12,
        min_sen_sim: float = 0.25,
        cache_embeddings: bool = True,
        cache_dir: str = "./cache",
        device: str = "cpu",
    ):
        super().__init__()
        self.max_sentences = max_sentences
        self.min_sen_sim = min_sen_sim
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir

        with open(data_path, "r") as f:
            data_list = json.load(f)
        with open(nodes_path, "r") as f:
            nodes_list = json.load(f)
        with open(edges_path, "r") as f:
            edges_list = json.load(f)
        with open(semantic_edges_path, "r") as f:
            semantic_list = json.load(f)
        with open(evidence_path, "r") as f:
            evidence_list = json.load(f)

        self.id2data = _index_by_id(data_list)
        self.id2nodes = _index_by_id(nodes_list)
        self.id2edges = _index_by_id(edges_list)
        self.id2sem = _load_semantic_map(semantic_list)
        self.id2evidence = _index_by_id(evidence_list)

        # keep only ids present in all sources
        ids = set(self.id2data.keys()) & set(self.id2nodes.keys()) & set(self.id2edges.keys()) & set(self.id2evidence.keys())
        self.ids = sorted(list(ids))

        # encoder (frozen for feature init)
        self.encoder = SentenceTransformer(encoder_name, device=device)

        # embedding caches in-memory (fast). You can replace with disk cache later.
        self._claim_emb: Dict[str, torch.Tensor] = {}
        self._sent_emb: Dict[str, Dict[int, torch.Tensor]] = {}
        self._ent_emb: Dict[str, Dict[int, torch.Tensor]] = {}

        self._prepare_embeddings()

    def __len__(self):
        return len(self.ids)

    def _prepare_embeddings(self):
        """
        Precompute embeddings per example:
        - claim embedding
        - selected sentence embeddings
        - involved entity embeddings (encode entity norm string)
        """
        for ex_id in self.ids:
            data = self.id2data[ex_id]
            claim = data["claim"]

            # selected sentences: use evidence_path top_evidences
            ev = self.id2evidence[ex_id].get("top_evidences", [])
            # top_evidences are list[dict] with sid / text / scores
            sids = []
            sid2text = {}
            for item in ev[: self.max_sentences]:
                sid = str(item["sid"])
                sids.append(sid)
                sid2text[sid] = item.get("text", "")

            # fallback: if evidence list empty, skip this example by giving empty; but better to still keep
            if len(sids) == 0:
                # take from nodes sent_nodes first few (not ideal but avoids crash)
                sent_nodes = self.id2nodes[ex_id]["sent_nodes"]
                for s in sent_nodes[: self.max_sentences]:
                    sid = str(s["sid"])
                    sids.append(sid)
                    sid2text[sid] = s.get("sentences", "")

            # claim embedding
            c = self.encoder.encode([claim], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
            self._claim_emb[ex_id] = torch.from_numpy(c)

            # sentence embeddings
            texts = [sid2text[sid] for sid in sids]
            s_emb = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            self._sent_emb[ex_id] = {sid: torch.from_numpy(s_emb[i]) for i, sid in enumerate(sids)}

            # entity embeddings: entities connected to selected sids via sn_edges
            node = self.id2nodes[ex_id]
            edge = self.id2edges[ex_id]
            entity_nodes = node["entity_nodes"]
            sn_edges = edge["sn_edges"]

            eid2norm = {str(n["eid"]): (n.get("norm") or n.get("name") or "") for n in entity_nodes}

            used_eids = set()
            sid_set = set(sids)
            for e in sn_edges:
                if str(e["sid"]) in sid_set:
                    used_eids.add(str(e["eid"]))

            eids = sorted(list(used_eids))
            ent_texts = [(eid2norm.get(eid, "") or str(eid)) for eid in eids]
            e_emb = self.encoder.encode(ent_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            self._ent_emb[ex_id] = {eid: torch.from_numpy(e_emb[i]) for i, eid in enumerate(eids)}

    def _build_heterodata(self, ex_id: str) -> HeteroData:
        data = self.id2data[ex_id]
        node = self.id2nodes[ex_id]
        edge = self.id2edges[ex_id]
        sem_edges = self.id2sem.get(ex_id, [])

        claim = data["claim"]
        y = get_label(data)

        # selected sids
        sent_map = self._sent_emb[ex_id]
        sids = list(sent_map.keys())
        sid2i = {sid: i for i, sid in enumerate(sids)}

        ent_map = self._ent_emb[ex_id]
        eids = list(ent_map.keys())
        eid2i = {eid: i for i, eid in enumerate(eids)}

        hd = HeteroData()

        # node features
        if len(sids) == 0:
            # avoid crash; create dummy one node
            dim = self._claim_emb[ex_id].numel()
            hd["sentence"].x = torch.zeros((1, dim), dtype=torch.float)
            sid2i = {"__DUMMY__": 0}
        else:
            hd["sentence"].x = torch.stack([sent_map[sid] for sid in sids], dim=0)

        if len(eids) == 0:
            dim = hd["sentence"].x.size(-1)
            hd["entity"].x = torch.zeros((0, dim), dtype=torch.float)
        else:
            hd["entity"].x = torch.stack([ent_map[eid] for eid in eids], dim=0)

        # edges: S-N and reverse N-S
        sn_src, sn_dst = [], []
        for e in edge["sn_edges"]:
            sid = str(e["sid"])
            eid = str(e["eid"])
            if sid in sid2i and eid in eid2i:
                sn_src.append(sid2i[sid])
                sn_dst.append(eid2i[eid])

        if len(sn_src) == 0:
            hd[("sentence", "sn", "entity")].edge_index = torch.zeros((2, 0), dtype=torch.long)
            hd[("entity", "ns", "sentence")].edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            ei = torch.tensor([sn_src, sn_dst], dtype=torch.long)
            hd[("sentence", "sn", "entity")].edge_index = ei
            hd[("entity", "ns", "sentence")].edge_index = ei.flip(0)

        # edges: S-S (semantic)
        ss_src, ss_dst = [], []
        for se in sem_edges:
            sim = float(se.get("similarity", 0.0))
            if sim < self.min_sen_sim:
                continue
            a = str(se["sid1"])
            b = str(se["sid2"])
            if a in sid2i and b in sid2i:
                ss_src.append(sid2i[a]); ss_dst.append(sid2i[b])
                ss_src.append(sid2i[b]); ss_dst.append(sid2i[a])

        if len(ss_src) == 0:
            hd[("sentence", "ss", "sentence")].edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            hd[("sentence", "ss", "sentence")].edge_index = torch.tensor([ss_src, ss_dst], dtype=torch.long)

        # graph-level fields
        hd.y = torch.tensor([y], dtype=torch.long)
        hd.claim_emb = self._claim_emb[ex_id].unsqueeze(0)
        hd.ex_id = ex_id

        return hd

    def __getitem__(self, idx: int) -> HeteroData:
        ex_id = self.ids[idx]
        return self._build_heterodata(ex_id)
