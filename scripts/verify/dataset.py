import json
import torch
from torch.utils.data import TensorDataset

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
