import json
import argparse
import torch
from torch.utils.data import random_split
from dataset import ClaimGraphDataset
from utils import set_seed

def count_pos(ds):
    pos = 0
    for i in range(len(ds)):
        y = int(ds[i].y.item())
        pos += (y == 1)
    return pos, len(ds) - pos

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("Device:", device)

    # train, dev
    train_dev_ds = ClaimGraphDataset(
        data_path=args.data_path.replace("[TYPE]", "train"),
        nodes_path=args.nodes_path.replace("[TYPE]", "train"),
        edges_path=args.edges_path.replace("[TYPE]", "train"),
        semantic_edges_path=args.semantic_edges_path.replace("[TYPE]", "train"),
        evidence_path=args.evidence_path.replace("[TYPE]", "train"),
        encoder_name=args.encoder_name,
        max_sentences=args.max_sentences,
        min_sen_sim=args.min_sen_sim,
        device=device if args.encode_on_gpu else "cpu",
    )
    
    # test
    test_ds = ClaimGraphDataset(
        data_path=args.data_path.replace("[TYPE]", "dev"),
        nodes_path=args.nodes_path.replace("[TYPE]", "dev"),
        edges_path=args.edges_path.replace("[TYPE]", "dev"),
        semantic_edges_path=args.semantic_edges_path.replace("[TYPE]", "dev"),
        evidence_path=args.evidence_path.replace("[TYPE]", "dev"),
        encoder_name=args.encoder_name,
        max_sentences=args.max_sentences,
        min_sen_sim=args.min_sen_sim,
        device=device if args.encode_on_gpu else "cpu",
    )

    n_total = len(train_dev_ds)
    n_train = int(n_total * args.train_ratio)
    n_val = n_total - n_train
    train_set, val_set = random_split(train_dev_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    
    print("Train pos/neg:", count_pos(train_set))
    print("Val   pos/neg:", count_pos(val_set))
    print("Test  pos/neg:", count_pos(test_ds))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_path", type=str, default="../../data/bm25_[TYPE].json")
    ap.add_argument("--nodes_path", type=str, default="../../data/bm25_nodes_[TYPE].json")
    ap.add_argument("--edges_path", type=str, default="../../data/bm25_edges_[TYPE].json")
    ap.add_argument("--semantic_edges_path", type=str, default="../../data/bm25_semantic_edges_[TYPE].json")
    ap.add_argument("--evidence_path", type=str, default="../../data/search_results_greedy_select_[TYPE].json")
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="use CPU even if GPU is available")
    ap.add_argument("--encode_on_gpu", action="store_true", help="encode embeddings on GPU")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_sentences", type=int, default=12)
    ap.add_argument("--min_sen_sim", type=float, default=0.25)
    args = ap.parse_args()
    main(args)
    