import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=512):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = text.split()[:self.max_len]
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(token_ids)

class MaskedLanguageModelTask:
    def __init__(self, model, vocab, mask_token_id, pad_token_id, mlm_prob=0.15):
        self.model = model
        self.vocab = vocab
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mlm_prob = mlm_prob
        # 统一使用 -100 作为忽略索引
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.ignore_index = -100  # 添加这一行，定义明确的忽略索引

    def mask_tokens(self, inputs):
        labels = inputs.clone()
        masked_indices = torch.bernoulli(torch.full(labels.shape, self.mlm_prob)).bool()
        # 对非掩码位置使用忽略索引
        labels[~masked_indices] = self.ignore_index  

        # 80%的概率用[MASK]替换
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10%的概率用随机词替换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # 对填充位置也应用忽略索引
        padding_mask = (inputs == self.pad_token_id)
        labels[padding_mask] = self.ignore_index

        return inputs, labels

    def train_step(self, inputs, optimizer):
        inputs, labels = self.mask_tokens(inputs)
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1, len(self.vocab)), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

class SentimentClassificationTask:
    def __init__(self, model, num_classes, pad_token_id):
        self.model = model
        self.pooler = nn.Sequential(
            nn.Linear(model.d_model, model.d_model),
            nn.Tanh()
        )
        self.classifier = nn.Linear(model.d_model, num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.pad_token_id = pad_token_id

    def forward(self, inputs):
        attention_mask = (inputs != self.pad_token_id).float()
        # 使用get_hidden_states获取Transformer编码器的输出，而不是经过解码器的输出
        hidden_states = self.model.get_hidden_states(inputs)
        # 取第一个token的表示作为整个序列的表示
        cls_repr = hidden_states[:, 0, :]
        pooled_output = self.pooler(cls_repr)
        logits = self.classifier(pooled_output)
        return logits

    def train_step(self, inputs, labels, optimizer):
        logits = self.forward(inputs)
        loss = self.criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()