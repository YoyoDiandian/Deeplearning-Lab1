import torch  
from torch.utils.data import Dataset, DataLoader  
from transformers import BertTokenizer, BertForMaskedLM  
import pandas as pd  
import os  

# 1. 加载分词器  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2. 数据加载与预处理  
class TextDataset(Dataset):  
    def __init__(self, sentences, labels, tokenizer, max_len=128):  
        self.sentences = sentences  
        self.labels = labels  
        self.tokenizer = tokenizer  
        self.max_len = max_len  

    def __getitem__(self, idx):  
        encoding = self.tokenizer(  
            self.sentences[idx],  
            max_length=self.max_len,  
            padding='max_length',  
            truncation=True,  
            return_tensors='pt'  
        )  
        return {  
            'input_ids': encoding['input_ids'].squeeze(),  
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  
        }  

    def __len__(self):  
        return len(self.sentences)  

# 3. 加载数据  
df = pd.read_csv('sentences_and_labels.csv')  

sentences = df['Sentence'].tolist()  
labels = df['Label'].tolist()  

# 4. 初始化模型（使用预训练BERT）  
model = BertForMaskedLM.from_pretrained('bert-base-uncased')  

# 5. 微调分类任务（添加分类头）  
from transformers import BertForSequenceClassification  
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  

# 6. 训练循环（示例）  
dataset = TextDataset(sentences, labels, tokenizer)  
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  
for epoch in range(10):  
    total_loss = 0  
    for batch in dataloader:  
        optimizer.zero_grad()  
        outputs = model(  
            input_ids=batch['input_ids'].to(device),  
            labels=batch['labels'].to(device)  
        )  
        loss = outputs.loss  
        loss.backward()  
        optimizer.step()  
        total_loss += loss.item()  
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")  