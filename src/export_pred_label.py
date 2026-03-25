import pandas as pd
import sys

def export_pred_label(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if 'id' not in df.columns:
        df['id'] = df.index
    df[['id', 'pred_label']].to_csv(output_csv, index=False)
    print(f"Exported to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/export_pred_label.py <input_csv> <output_csv>")
    else:
        export_pred_label(sys.argv[1], sys.argv[2])
