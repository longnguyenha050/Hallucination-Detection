import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name='vinai/phobert-base', num_labels=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[1] if len(outputs) > 1 else outputs[0][:,0,:]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

def get_tokenizer(model_name='vinai/phobert-base'):
    return AutoTokenizer.from_pretrained(model_name)
