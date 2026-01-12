import argparse
import json
import numpy as np
import torch
import os
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from collections import Counter

def count_hops(raw):
    c = Counter()
    for ex in raw:
        h = ex.get("num_hops", None)
        try:
            h = int(h)
        except:
            continue
        if h in [2,3,4]:
            c[h] += 1
    return c

HOP2ID = {2: 0, 3: 1, 4: 2}
ID2HOP = {v: k for k, v in HOP2ID.items()}


def read_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got: {type(data)}")
    return data


class HopClsDataset(Dataset):
    """
    Returns:
      input_ids: LongTensor [L]
      attention_mask: LongTensor [L]
      label_id: LongTensor []
      hop_raw: LongTensor [] (2/3/4)
    """
    def __init__(self, raw_list, tokenizer, max_len: int):
        self.items = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for ex in raw_list:
            claim = ex.get("claim", None)
            hop = ex.get("num_hops", None)

            if claim is None or hop is None:
                continue
            try:
                hop = int(hop)
            except Exception:
                continue
            if hop not in HOP2ID:
                continue

            enc = tokenizer(
                claim,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0)         # [L]
            attention_mask = enc["attention_mask"].squeeze(0)
            label_id = torch.tensor(HOP2ID[hop], dtype=torch.long)
            hop_raw = torch.tensor(hop, dtype=torch.long)

            self.items.append((input_ids, attention_mask, label_id, hop_raw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def _print_report(y_true, y_pred, title=""):
    # 标签名按 hop2/hop3/hop4
    target_names = ["HOP2", "HOP3", "HOP4"]
    labels = [0, 1, 2]

    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, labels=labels))
    print(confusion_matrix(y_true, y_pred, labels=labels))

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=labels, zero_division=0
    )
    print(f"macro_p={p:.4f} macro_r={r:.4f} macro_f1={f1:.4f}")
    return f1


def evaluate(model, loader, device, title="dev"):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc=f"Eval {title}"):
            ids, masks, labels, _ = batch
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=ids, attention_mask=masks)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)

            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(labels.detach().cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    macro_f1 = _print_report(all_true, all_pred, title=title)
    return macro_f1


def train(args, tokenizer, model, train_loader, dev_loader, device):
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    total_steps = len(train_loader) * args.epoch_num

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_macro_f1 = 0.0

    for epoch in range(args.epoch_num):
        model.train()
        running_loss = 0.0

        for i, (ids, masks, labels, _) in tqdm(
            enumerate(train_loader),
            ncols=100,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}"
        ):
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=ids, attention_mask=masks, labels=labels)
            loss = outputs.loss

            loss.backward()

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch+1}/{args.epoch_num} avg_train_loss={avg_train_loss:.4f}")

        macro_f1 = evaluate(model, dev_loader, device, title=f"dev@epoch{epoch+1}")
        print(f"Epoch {epoch+1} dev_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            os.makedirs(os.path.dirname(args.saved_model_path), exist_ok=True)
            model.save_pretrained(args.saved_model_path)
            tokenizer.save_pretrained(args.saved_model_path)
            print(f"[SAVE] best_macro_f1={best_macro_f1:.4f} -> {args.saved_model_path}")


def test(model, test_loader, device):
    """
    额外输出：按 hop(2/3/4) 的准确率/样本数，帮助诊断路由器偏差
    """
    model.eval()

    hop_bucket_true = {2: [], 3: [], 4: []}
    hop_bucket_pred = {2: [], 3: [], 4: []}

    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, ncols=100, desc="Testing"):
            ids, masks, labels, hop_raw = batch
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=ids, attention_mask=masks)
            logits = outputs.logits
            scores = F.softmax(logits, dim=-1)
            pred = torch.argmax(scores, dim=-1)

            y_true = labels.detach().cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            hops = hop_raw.detach().cpu().numpy()

            all_true.append(y_true)
            all_pred.append(y_pred)

            for i in range(len(hops)):
                hop = int(hops[i])
                if hop in hop_bucket_true:
                    hop_bucket_true[hop].append(int(y_true[i]))
                    hop_bucket_pred[hop].append(int(y_pred[i]))

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    _print_report(all_true, all_pred, title="test_overall")

    for hop in [2, 3, 4]:
        if len(hop_bucket_true[hop]) == 0:
            print(f"\n===== hop={hop} =====\nNo samples.")
            continue
        y_true = np.array(hop_bucket_true[hop], dtype=np.int64)
        y_pred = np.array(hop_bucket_pred[hop], dtype=np.int64)
        acc = float((y_true == y_pred).mean())
        print(f"\n===== hop={hop} =====")
        print(f"samples={len(y_true)} acc={acc:.4f}")


def main(args):
    train_path = args.data_path.replace("[TYPE]", "train")
    test_path = args.data_path.replace("[TYPE]", "dev")

    train_dev_raw = read_data(train_path)
    
    labels_for_split = []
    filtered = []
    for ex in train_dev_raw:
        hop = int(ex['num_hops'])
        if hop in HOP2ID:
            filtered.append(ex)
            labels_for_split.append(HOP2ID[hop])
    
    train_raw, dev_raw = train_test_split(
        filtered,
        test_size=1-args.train_ratio,
        random_state=args.seed,
        stratify=labels_for_split
    )

    # split train/dev
    n_total = len(train_dev_raw)
    n_train = int(n_total * args.train_ratio)
    train_raw = train_dev_raw[:n_train]
    dev_raw = train_dev_raw[n_train:]

    test_raw = read_data(test_path)
    if args.num_sample > 0:
        test_raw = test_raw[:args.num_sample]

    print("train hop dist:", count_hops(train_raw))
    print("dev   hop dist:", count_hops(dev_raw))
    print("test  hop dist:", count_hops(test_raw))

    print(f"finished loading data: train_dev={n_total}, train={len(train_raw)}, dev={len(dev_raw)}, test={len(test_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=3)
    model.to(args.device)
    print("finished loading BERT model")

    # load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"loaded checkpoint: {args.checkpoint}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_ds = HopClsDataset(train_raw, tokenizer, args.max_len)
    dev_ds = HopClsDataset(dev_raw, tokenizer, args.max_len)
    test_ds = HopClsDataset(test_raw, tokenizer, args.max_len)

    print(f"finished batching data: train={len(train_ds)}, dev={len(dev_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=args.batch_size)
    dev_loader = DataLoader(dev_ds, sampler=SequentialSampler(dev_ds), batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=args.batch_size)
    print("finished creating data loaders")

    train(args, tokenizer, model, train_loader, dev_loader, args.device)

    test(model, test_loader, args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data_path", type=str, default="./data/plan1/bm25_[TYPE].json",
                        help="JSON list, each item must contain: claim, num_hops (2/3/4)")
    parser.add_argument("--saved_model_path", type=str, default="./save_models/hop_cls/")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional: load state_dict before train/test")

    # model
    parser.add_argument("--bert_model_name", type=str, default="microsoft/deberta-v3-base")

    # running mode
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--do_train", action="store_true", help="If set, run training then test. Otherwise only test.")

    # data split / sampling
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_sample", type=int, default=-1)

    # train hyperparams
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_len", type=int, default=128, help="Hop prediction usually only needs claim, 128 is enough")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

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
