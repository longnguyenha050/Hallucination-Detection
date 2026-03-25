import pandas as pd
from utils import compute_metrics

def evaluate(pred_file, label_col='label', pred_col='pred_label'):
    df = pd.read_csv(pred_file)
    y_true = df[label_col].tolist()
    y_pred = df[pred_col].tolist()
    metrics = compute_metrics(y_true, y_pred)
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics

if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1])
