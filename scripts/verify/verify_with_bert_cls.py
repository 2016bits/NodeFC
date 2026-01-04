import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

from scripts.verify.dataset import read_data, batch_ce_data

def train(args, model, train_loader, dev_loader, device):
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    total_steps = len(train_loader) * args.epoch_num
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    
    best_macro_f1 = 0.0
    
    for epoch in range(args.epoch_num):
        model.train()
        running_loss = 0.0

        for i, (ids, masks, labels, _) in tqdm(enumerate(train_loader), ncols=100, total=len(train_loader), desc=f"Epoch {epoch+1}"):
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=ids, attention_mask=masks, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            running_loss += float(loss.item())
        
        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch+1}/{args.epoch_num} avg_train_loss={avg_train_loss:.4f}")

        all_prediction = []
        all_target = []
        model.eval()
        for i, (ids, masks, labels, _) in tqdm(enumerate(dev_loader), ncols=100, total=len(dev_loader), desc=f"Epoch {epoch+1}, validation "):
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=ids, attention_mask=masks)
                logits = outputs.logits  # [B, num_labels]

                pred_label = torch.argmax(logits, dim=-1)

            all_prediction.append(pred_label.detach().cpu().numpy())
            all_target.append(labels.detach().cpu().numpy())
        
        all_prediction = np.concatenate(all_prediction, axis=0)
        all_target = np.concatenate(all_target, axis=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(all_target, all_prediction, average='macro')
        print(f"Epoch {epoch+1} dev_macro_f1={macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), args.saved_model_path.replace('[T]', args.t))

def _print_report(y_true, y_pred, title=""):
    target_names = ["SUPPORTS", "REFUTES"]   # 你当前 two_class_label_dict: SUPPORTS=0, REFUTES=1
    labels = [0, 1]
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, labels=labels))
    print(confusion_matrix(y_true, y_pred, labels=labels))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    print(f"macro_p={p:.4f} macro_r={r:.4f} macro_f1={f1:.4f}")
    return

def test(model, test_loader, device):
    hop_preds = {2: [], 3: [], 4: []}
    hop_trues = {2: [], 3: [], 4: []}
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, ncols=100, desc="Testing"):
            ids, msks, labels, hops = batch
            ids = ids.to(device)
            msks = msks.to(device)
            labels = labels.to(device)
            hops = hops.to(device)
            
            outputs = model(input_ids=ids, attention_mask=msks)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            scores = F.softmax(logits, dim=-1)
            pred = torch.argmax(scores, dim=-1)
            
            for i in range(labels.size(0)):
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
        

def main(args):
    train_path = args.data_path.replace("[TYPE]", "train").replace("[T]", args.t)
    test_path = args.data_path.replace("[TYPE]", "dev").replace("[T]", args.t)

    train_dev_raw = read_data(train_path)
    train_raw, dev_raw = np.split(np.array(train_dev_raw), [int(len(train_dev_raw)*args.train_ratio)])
    test_raw = read_data(test_path)
    print("finished loading data")
    
    # tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    bert_model = BertForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=2)
    bert_model.to(args.device)
    print("finished loading BERT model")
    
    # batch data
    train_batched = batch_ce_data(train_raw, tokenizer, args.max_len)
    dev_batched = batch_ce_data(dev_raw, tokenizer, args.max_len)
    test_batched = batch_ce_data(test_raw, tokenizer, args.max_len)
    print("finished batching data")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    train_loader = DataLoader(train_batched, sampler=RandomSampler(train_batched), batch_size=args.batch_size)
    dev_loader = DataLoader(dev_batched, sampler=SequentialSampler(dev_batched), batch_size=args.batch_size)
    test_loader = DataLoader(test_batched, sampler=SequentialSampler(test_batched), batch_size=args.batch_size)
    print("finished creating data loaders")
    
    train(args, bert_model, train_loader, dev_loader, args.device)
    test(bert_model, test_loader, args.device)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data_path", type=str, default="./data/verify/noderag_retrieved_[T][TYPE].json")
    parser.add_argument("--saved_model_path", type=str, default="./save_models/bert_cls/bert_cls_[T]_best.pth")
    parser.add_argument("--checkpoint", type=str, default="")  # test 模式时加载
    parser.add_argument("--t", type=str, default="")

    # model
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_classes", type=int, default=2)

    # running mode
    parser.add_argument("--device", type=str, default="cuda")  # None -> auto detect cuda/cpu

    # data split / sampling
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_sample", type=int, default=-1)  # >0 则截断数据做小实验

    # train hyperparams
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=512)
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
