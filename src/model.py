import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name='vinai/phobert-base', num_labels=3, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Freeze một số layer đầu nếu muốn
        # for param in list(self.bert.parameters())[:-4]:  # Freeze tất cả trừ 4 layer cuối
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Sử dụng hidden state của last layer
        last_hidden_state = outputs.last_hidden_state
        
        # Lấy embedding của token [CLS]
        cls_embedding = last_hidden_state[:, 0, :]
        
        # Hoặc average pooling
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # cls_embedding = sum_embeddings / sum_mask
        
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits

def get_tokenizer(model_name='vinai/phobert-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    return tokenizer