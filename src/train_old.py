import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from model import PhoBERTClassifier
from preprocess import load_data
from utils import label2id, compute_metrics
import pandas as pd
import numpy as np

class HallucinationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['context'] + ' [SEP] ' + row['prompt'] + ' [SEP] ' + row['response']
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        label = label2id(row['label'])
        return { 'input_ids': inputs['input_ids'].squeeze(0), 'attention_mask': inputs['attention_mask'].squeeze(0), 'label': torch.tensor(label) }

def train_model(data_path, model_name='vinai/phobert-base', epochs=5, batch_size=4, lr=2e-5):
    df = load_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = HallucinationDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PhoBERTClassifier(model_name)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")
    torch.save(model.state_dict(), 'phobert_hallucination.pt')
    print("Model saved as phobert_hallucination.pt")

if __name__ == "__main__":
    import sys
    train_model(sys.argv[1])
