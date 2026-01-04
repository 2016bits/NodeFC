import json
from sklearn.metrics import classification_report, confusion_matrix

def print_evaluation_results(predictions, gt_labels, num_of_classes=3):
    if num_of_classes == 3:
        target_names = ['REFUTES', 'SUPPORTS', 'NEI']
        label_map = {'REFUTES': 0, 'SUPPORTS': 1, 'NEI': 2}
        labels = [label_map[e] for e in gt_labels]
        predictions = [label_map[e] for e in predictions]
        print(classification_report(labels, predictions, target_names=target_names, digits=4))
        print(confusion_matrix(labels, predictions))
        print()
    elif num_of_classes == 2:
        target_names = ['REFUTES', 'SUPPORTS']
        label_map = {'REFUTES': 0, 'SUPPORTS': 1}
        labels = [label_map[e] for e in gt_labels]
        predictions = [label_map[e] for e in predictions]
        print(classification_report(labels, predictions, target_names=target_names, digits=4))
        print(confusion_matrix(labels, predictions))
        print()

def evaluate_hover_by_hops(in_data, results_data):
    id_num_hops_map = {sample['id']:sample['num_hops'] for sample in in_data}
    
    predictions = {'2_hop': [], '3_hop': [], '4_hop': []}
    gt_labels = {'2_hop': [], '3_hop': [], '4_hop': []}
    for sample in results_data:
        key = f"{id_num_hops_map[sample['id']]}_hop"
        gt_labels[key].append(sample['gold_label'].strip())
        predictions[key].append(sample['pred_label'].strip())

    for key in predictions:
        print(key)
        print_evaluation_results(predictions[key], gt_labels[key], num_of_classes=2)
        print()

in_path = 'data/bm25_dev.json'
results_path = 'data/bm25_noderag_search_greedy_select_gnn_predictions.json'

with open(in_path, 'r') as f:
    in_data = json.load(f)
with open(results_path, 'r') as f:
    results_data = json.load(f)
evaluate_hover_by_hops(in_data, results_data)