import torch
from model import PhoBERTClassifier, get_tokenizer
from preprocess import load_data
from utils import id2label
import pandas as pd

def predict(data_path, model_path='phobert_hallucination.pt', model_name='vinai/phobert-base'):
    df = load_data(data_path)
    tokenizer = get_tokenizer(model_name)
    model = PhoBERTClassifier(model_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    results = []
    for _, row in df.iterrows():
        text = row['context'] + ' [SEP] ' + row['prompt'] + ' [SEP] ' + row['response']
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            pred = torch.argmax(logits, dim=1).item()
            results.append(id2label(pred))
    df['pred_label'] = results
    df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    import sys
    predict(sys.argv[1])
