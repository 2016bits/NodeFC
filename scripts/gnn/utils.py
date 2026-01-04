import re
import numpy as np
import torch

LABEL2ID = {"REFUTES": 0, "SUPPORTS": 1}

def get_label(example: dict) -> int:
    # robust label fetching
    for k in ["label", "verdict", "gold_label", "classification"]:
        if k in example:
            v = str(example[k]).strip().upper()
            if v in LABEL2ID:
                return LABEL2ID[v]
    raise KeyError(f"Cannot find label in example keys: {list(example.keys())}")

def extract_numbers(text: str):
    return set(re.findall(r"\b\d{1,4}(?:\.\d+)?%?\b", text))

def l2norm(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x) + eps
    return x / n

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
