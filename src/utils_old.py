import torch
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return {'macro_f1': macro_f1, 'accuracy': acc}

def label2id(label):
    mapping = {'no': 0, 'intrinsic': 1, 'extrinsic': 2}
    return mapping[label]

def id2label(idx):
    mapping = {0: 'no', 1: 'intrinsic', 2: 'extrinsic'}
    return mapping[idx]
