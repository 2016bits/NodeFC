import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from dataset import ClaimGraphDataset
from model import ClaimHeteroGNN
from utils import set_seed

def evaluate(model, loader, device, threshold=None):
    model.eval()
    total, correct = 0, 0
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1)  # [B]
            pred = torch.softmax(logits, dim=-1)
            if threshold is None:
                pred = pred.argmax(dim=-1)
            else:
                pred = (pred[:, 1] >= threshold).long()
            
            total += y.size(0)
            correct += (pred == y).sum().item()

            # binary confusion: label 1 = SUPPORTS, 0 = REFUTES
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()

    acc = correct / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

ID2LABEL = {0: "REFUTES", 1: "SUPPORTS"}
def _ensure_list(x):
    # PyG Batch may store custom attrs as list or scalar
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def predict_and_save(model, loader, device, out_path):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            gold = batch.y.view(-1)
            
            ex_ids = _ensure_list(getattr(batch, "ex_id"))
            B = gold.size(0)
            if len(ex_ids) != B:
                # fallback: create dummy ids if mismatch (shouldn't happen normally)
                ex_ids = ex_ids[:B] + [f"__missing_{i}" for i in range(len(ex_ids), B)]

            for i in range(B):
                g = int(gold[i].item())
                p = int(pred[i].item())
                results.append({
                    "id": str(ex_ids[i]),
                    "gold": g,
                    "pred": p,
                    "gold_label": ID2LABEL.get(g, str(g)),
                    "pred_label": ID2LABEL.get(p, str(p)),
                    "prob_refutes": float(probs[i, 0].item()),
                    "prob_supports": float(probs[i, 1].item()),
                })
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

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
    print("Train+Dev examples:", len(train_dev_ds))
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
    print("Test examples:", len(test_ds))

    n_total = len(train_dev_ds)
    n_train = int(n_total * args.train_ratio)
    n_val = n_total - n_train
    train_set, val_set = random_split(train_dev_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # infer in_dim from one sample
    sample = test_ds[0]
    in_dim = sample["sentence"].x.size(-1)
    model = ClaimHeteroGNN(in_dim=in_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    pos, neg = 0, 0
    for i in range(len(train_set)):
        y = int(train_set[i].y.item())
        pos += (y == 1)
        neg += (y == 0)

    # label: 0=REFUTES, 1=SUPPORTS
    w0 = (pos + neg) / (2.0 * max(1, neg))  # REFUTES weight
    w1 = (pos + neg) / (2.0 * max(1, pos))  # SUPPORTS weight
    class_weight = torch.tensor([w0, w1], dtype=torch.float, device=device)
    print("Class weights:", class_weight.tolist(), "pos/neg:", pos, neg)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    best_val_f1 = -1.0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1)

            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / max(1, len(train_set))
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} "
              f"val_prec={val_metrics['prec']:.4f} val_rec={val_metrics['rec']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    # load best and test
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device)
    print("TEST:", test_metrics)
    
    # save predictions
    out_path = args.out_path.replace("[K]", str(args.topk))
    predict_and_save(model, test_loader, device, out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_path", type=str, default="../../data/bm25_[TYPE].json")
    ap.add_argument("--nodes_path", type=str, default="../../data/bm25_nodes_[TYPE].json")
    ap.add_argument("--edges_path", type=str, default="../../data/bm25_edges_[TYPE].json")
    ap.add_argument("--semantic_edges_path", type=str, default="../../data/bm25_semantic_edges_[TYPE].json")
    ap.add_argument("--evidence_path", type=str, default="../../data/search_results_greedy_select_contra_rerank_[TYPE].json")
    ap.add_argument("--out_path", type=str, default="../../data/top[K]/bm25_noderag_search_greedy_select_contra_rerank_gnn_predictions.json")
    
    ap.add_argument("--topk", type=int, default=10, help='BM25 top-K evidences to use')
    ap.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_sentences", type=int, default=12)
    ap.add_argument("--min_sen_sim", type=float, default=0.25)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--train_ratio", type=float, default=0.8)

    ap.add_argument("--ckpt_dir", type=str, default="./ckpt")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cpu", action="store_true", help="force cpu for training")
    ap.add_argument("--encode_on_gpu", action="store_true", help="compute embeddings on GPU during dataset init")

    args = ap.parse_args()
    main(args)
