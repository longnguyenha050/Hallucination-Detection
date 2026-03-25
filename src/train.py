import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import PhoBERTClassifier
from preprocess import load_data
from utils import label2id, compute_metrics
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HallucinationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Tạo input text theo format chuẩn hơn
        text = f"context: {row['context']} prompt: {row['prompt']} response: {row['response']}"
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        label = label2id(row['label'])
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(data_path, model_name='vinai/phobert-base', epochs=5, batch_size=4, lr=2e-5, val_size=0.2):
    # Load và chia dữ liệu
    df = load_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Thêm token đặc biệt nếu cần
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    # Chia train/validation
    train_size = int(len(df) * (1 - val_size))
    val_size = len(df) - train_size
    train_df, val_df = random_split(df, [train_size, val_size])
    train_df = df.iloc[train_df.indices].reset_index(drop=True)
    val_df = df.iloc[val_df.indices].reset_index(drop=True)
    
    train_dataset = HallucinationDataset(train_df, tokenizer)
    val_dataset = HallucinationDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model và optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PhoBERTClassifier(model_name)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Tính metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        metrics = compute_metrics(val_labels, val_preds)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Macro-F1: {metrics['macro_f1']:.4f}, Val Accuracy: {metrics['accuracy']:.4f}")
        
        # Lưu model tốt nhất
        if metrics['macro_f1'] > best_val_f1:
            best_val_f1 = metrics['macro_f1']
            torch.save(model.state_dict(), 'best_phobert_hallucination.pt')
            print(f"Saved best model with Macro-F1: {best_val_f1:.4f}")
    
    print(f"\nTraining completed. Best Validation Macro-F1: {best_val_f1:.4f}")
    return model

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path> [model_name] [epochs] [batch_size] [lr]")
    else:
        data_path = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else 'vinai/phobert-base'
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        lr = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-5
        
        train_model(data_path, model_name, epochs, batch_size, lr)