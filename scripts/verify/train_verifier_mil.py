import argparse
import json
import numpy as np
import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from scripts.verify.dataset import MILVerifierDataset, mil_collate_fn

def read_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_report(y_true, y_pred, title=""):
    target_names = ["SUPPORTS", "REFUTES"]
    labels = [0, 1]
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, labels=labels))
    print(confusion_matrix(y_true, y_pred, labels=labels))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    print(f"macro_p={p:.4f} macro_r={r:.4f} macro_f1={f1:.4f}")
    return f1

def aggregate_logits(logits_bmlc: torch.Tensor, pooling: str = "max") -> torch.Tensor:
    """
    logits_bmlc: [B, M, C]
    return: [B, C]
    pooling:
      - max: 对每个类取 max evidence（最常用，适合“关键句决定标签”）
      - logsumexp: softer max，稍稳健
    """
    if pooling == "max":
        return logits_bmlc.max(dim=1).values
    elif pooling == "logsumexp":
        return torch.logsumexp(logits_bmlc, dim=1)
    else:
        raise ValueError(f"Unknown pooling={pooling}")


def train(args, model, train_loader, dev_loader, device):
    # DDP: model 可能是 DDP 包装过的
    raw_model = model.module if hasattr(model, "module") else model

    optimizer = AdamW(raw_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_loader) * args.epoch_num
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_macro_f1 = 0.0

    for epoch in range(args.epoch_num):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        raw_model.train()
        running_loss = 0.0

        for i, (ids, masks, labels, hops, _) in tqdm(
            enumerate(train_loader), ncols=100, total=len(train_loader),
            desc=f"Epoch {epoch+1}"
        ):
            ids = ids.to(device)       # [B, M, L]
            masks = masks.to(device)
            labels = labels.to(device)

            B, M, L = ids.shape
            ids_f = ids.view(B * M, L)
            masks_f = masks.view(B * M, L)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(input_ids=ids_f, attention_mask=masks_f)
            logits = outputs.logits                     # [B*M, C]
            logits = logits.view(B, M, -1)              # [B, M, C]

            agg = aggregate_logits(logits, pooling=args.pooling)   # [B, C]
            loss = F.cross_entropy(agg, labels)

            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        if is_main_process(args):
            print(f"\nEpoch {epoch+1}/{args.epoch_num} avg_train_loss={avg_train_loss:.4f}")

        # dev eval
        macro_f1 = evaluate(args, model, dev_loader, device, title=f"dev@epoch{epoch+1}")
        if is_main_process(args):
            print(f"Epoch {epoch+1} dev_macro_f1={macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            if is_main_process(args):
                os.makedirs(os.path.dirname(args.saved_model_path), exist_ok=True)
                state_dict = raw_model.state_dict()
                torch.save(state_dict, args.saved_model_path.replace('[T]', args.t))
                print(f"[SAVE] best_macro_f1={best_macro_f1:.4f} -> {args.saved_model_path.replace('[T]', args.t)}")



def ddp_concat_all_gather_1d(t: torch.Tensor) -> torch.Tensor:
    """
    Gather variable-length 1D tensors from all ranks, concat on rank0.
    On non-rank0, return empty tensor.
    """
    if not dist.is_available() or not dist.is_initialized():
        return t.detach().cpu()

    t = t.contiguous()
    local_n = torch.tensor([t.numel()], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_n)
    sizes = [int(x.item()) for x in sizes]
    max_n = max(sizes)

    if t.numel() < max_n:
        pad = torch.zeros(max_n - t.numel(), device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t

    gathered = [torch.zeros_like(t_pad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t_pad)

    if dist.get_rank() == 0:
        out = []
        for g, n in zip(gathered, sizes):
            out.append(g[:n].detach().cpu())
        return torch.cat(out, dim=0)
    else:
        return torch.empty(0, dtype=t.dtype)

def evaluate(args, model, loader, device, title="dev"):
    model.eval()
    pred_list = []
    true_list = []

    with torch.no_grad():
        it = loader
        # 非主进程少打日志避免刷屏
        if not is_main_process(args):
            it = loader
        for ids, masks, labels, hops, _ in tqdm(it, ncols=100, desc=f"Eval {title}"):
            ids = ids.to(device)     # [B, M, L]
            masks = masks.to(device)
            labels = labels.to(device)

            B, M, L = ids.shape
            ids_f = ids.view(B * M, L)
            masks_f = masks.view(B * M, L)

            outputs = model(input_ids=ids_f, attention_mask=masks_f)
            logits = outputs.logits.view(B, M, -1)          # [B, M, C]
            agg = aggregate_logits(logits, pooling=args.pooling)   # [B, C]
            pred = torch.argmax(agg, dim=-1)

            pred_list.append(pred.detach())
            true_list.append(labels.detach())

    pred_local = torch.cat(pred_list, dim=0) if len(pred_list) else torch.empty(0, device=device, dtype=torch.long)
    true_local = torch.cat(true_list, dim=0) if len(true_list) else torch.empty(0, device=device, dtype=torch.long)

    # gather on rank0
    pred_all = ddp_concat_all_gather_1d(pred_local)
    true_all = ddp_concat_all_gather_1d(true_local)

    if is_main_process(args):
        y_pred = pred_all.numpy()
        y_true = true_all.numpy()
        macro_f1 = _print_report(y_true, y_pred, title=title)
    else:
        macro_f1 = 0.0

    # broadcast macro_f1 to all ranks
    if args.distributed:
        macro_f1_t = torch.tensor([macro_f1], device=device, dtype=torch.float32)
        dist.broadcast(macro_f1_t, src=0)
        macro_f1 = float(macro_f1_t.item())

    return macro_f1


def test(args, model, test_loader, device):
    hop_preds = {2: [], 3: [], 4: []}
    hop_trues = {2: [], 3: [], 4: []}

    model.eval()
    with torch.no_grad():
        for ids, masks, labels, hops, _ in tqdm(test_loader, ncols=100, desc="Testing"):
            ids = ids.to(device)     # [B, M, L]
            masks = masks.to(device)
            labels = labels.to(device)
            hops = hops.to(device)

            B, M, L = ids.shape
            ids_f = ids.view(B * M, L)
            masks_f = masks.view(B * M, L)

            outputs = model(input_ids=ids_f, attention_mask=masks_f)
            logits = outputs.logits.view(B, M, -1)                 # [B, M, C]
            agg = aggregate_logits(logits, pooling=args.pooling)   # [B, C]

            scores = F.softmax(agg, dim=-1)
            pred = torch.argmax(scores, dim=-1)

            for i in range(B):
                hop = int(hops[i].item())
                if hop in hop_preds:
                    hop_preds[hop].append(int(pred[i].item()))
                    hop_trues[hop].append(int(labels[i].item()))

    for hop in [2, 3, 4]:
        if len(hop_trues[hop]) == 0:
            print(f"\n===== {hop}-hop =====\nNo samples.")
            continue
        y_true = np.array(hop_trues[hop], dtype=np.int64)
        y_pred = np.array(hop_preds[hop], dtype=np.int64)
        _print_report(y_true, y_pred, title=f"{hop}-hop")

def ddp_setup(args):
    # torchrun 会自动注入这些环境变量
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        dist.barrier()

def is_main_process(args):
    return (not getattr(args, "distributed", False)) or args.rank == 0

def ddp_cleanup(args):
    if getattr(args, "distributed", False):
        dist.barrier()
        dist.destroy_process_group()
        
def main(args):
    ddp_setup(args)
    device = torch.device(f"cuda:{args.local_rank}" if args.distributed else args.device)
    args.device = device

    train_path = args.data_path.replace("[TYPE]", "train").replace("[T]", args.t)
    test_path = args.data_path.replace("[TYPE]", "dev").replace("[T]", args.t)

    train_dev_raw = read_data(train_path)
    if args.num_sample > 0:
        train_dev_raw = train_dev_raw[:args.num_sample]

    # split train/dev
    n_train = int(len(train_dev_raw) * args.train_ratio)
    train_raw = train_dev_raw[:n_train]
    dev_raw = train_dev_raw[n_train:]

    test_raw = read_data(test_path)
    if args.num_sample > 0:
        test_raw = test_raw[:args.num_sample]

    if is_main_process(args):
        print("finished loading data")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=2)
    model.to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    if is_main_process(args):
        print("finished loading verifier model")

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"loaded checkpoint: {args.checkpoint}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_ds = MILVerifierDataset(train_raw, max_ev=args.max_ev, keep_titles=args.keep_titles)
    dev_ds = MILVerifierDataset(dev_raw, max_ev=args.max_ev, keep_titles=args.keep_titles)
    test_ds = MILVerifierDataset(test_raw, max_ev=args.max_ev, keep_titles=args.keep_titles)

    if is_main_process(args):
        print(f"dataset sizes: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    collate = lambda b: mil_collate_fn(b, tokenizer, args.max_len)

    if args.distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        dev_sampler = DistributedSampler(dev_ds, shuffle=False)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        test_sampler = SequentialSampler(test_ds)

    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    if is_main_process(args):
        print("finished creating data loaders")

    if args.do_train:
        train(args, model, train_loader, dev_loader, args.device)

    test(args, model, test_loader, args.device)

    ddp_cleanup(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data_path", type=str, default="./data/plan2/[T]_[TYPE]_verifying_data.json")
    parser.add_argument("--saved_model_path", type=str, default="./save_models/bert_mil/bert_mil_[T]_best.pth")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--t", type=str, default="")

    # model
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")

    # running mode
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--do_train", action="store_true")

    # data split / sampling
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_sample", type=int, default=-1)

    # MIL settings
    parser.add_argument("--max_ev", type=int, default=5, help="Use top-M evidence sentences per claim")
    parser.add_argument("--keep_titles", action="store_true", help="If set, do NOT filter title-like items")
    parser.add_argument("--pooling", type=str, default="max", choices=["max", "logsumexp"],
                        help="MIL pooling over evidence instances")

    # train hyperparams
    parser.add_argument("--batch_size", type=int, default=32,
                        help="MIL uses B*M pairs, so batch_size should be smaller than concat version")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_len", type=int, default=256, help="claim+sentence is shorter than concat(top5)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=4)
    
    # seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 1) 强制 transformers 不用 SDPA
    os.environ["TRANSFORMERS_NO_SDPA"] = "1"
    # 2) 强制 torch 不用 flash / mem-efficient 的 SDP
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    main(args)
