import torch
from torch.nn.utils.rnn import pad_sequence
from model_architecture import CustomLanguageModel
from training_tasks import TextDataset, MaskedLanguageModelTask, SentimentClassificationTask
from torch.utils.data import DataLoader
import os
import pandas as pd

# 新增：自定义collate_fn用于处理变长序列
def collate_fn(batch, pad_id):
    """将不同长度的序列填充到相同长度"""
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)

# 示例配置
vocab = {'<pad>': 0, '<unk>': 1, '<mask>': 2, 'hello': 3, 'world': 4, 'good': 5, 'bad': 6}  # 简化示例
vocab_size = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 初始化模型
model = CustomLanguageModel(
    vocab_size=vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2,
    max_seq_len=128
).to(device)

# 2. 预训练阶段 (掩码语言模型)
def pretrain_model(model, train_data, vocab):
    dataset = TextDataset(train_data, vocab)
    # 使用自定义collate_fn处理变长序列
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, vocab['<pad>'])  # 指定填充ID
    )
    
    mlm_task = MaskedLanguageModelTask(
        model=model,
        vocab=vocab,
        mask_token_id=vocab['<mask>'],
        pad_token_id=vocab['<pad>']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(3):  # 简化示例，实际需要更多轮次
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = mlm_task.train_step(batch, optimizer)
            total_loss += loss
        print(f"Epoch {epoch+1}, MLM Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# 3. 微调阶段 (情感分类)
def finetune_model(model, train_data, labels, num_classes=2):
    dataset = TextDataset(train_data, vocab)
    # 同样使用自定义collate_fn
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, vocab['<pad>'])
    )
    label_tensor = torch.tensor(labels).to(device)
    
    clf_task = SentimentClassificationTask(
        model=model,
        num_classes=num_classes,
        pad_token_id=vocab['<pad>']
    )
    
    # 只训练分类器部分 (可选择)
    optimizer = torch.optim.Adam(
        list(clf_task.pooler.parameters()) + list(clf_task.classifier.parameters()), 
        lr=1e-4
    )
    
    for epoch in range(5):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            batch_labels = label_tensor[i*8 : (i+1)*8]
            loss = clf_task.train_step(batch, batch_labels, optimizer)
            total_loss += loss
        print(f"Epoch {epoch+1}, Classification Loss: {total_loss/len(dataloader):.4f}")
    
    return clf_task

# 示例数据
# 读取CSV文件
training_data_path = os.path.join(os.path.dirname(__file__), 'sentences_and_labels.csv')
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"Training data file not found: {training_data_path}")
df = pd.read_csv(training_data_path)

# 假设CSV文件有两列：'sentence'和'label'
pretrain_data = df['Sentence'].tolist()
sentiment_labels = df['Label'].tolist()

# 训练流程
pretrained_model = pretrain_model(model, pretrain_data, vocab)
clf_model = finetune_model(pretrained_model, pretrain_data, sentiment_labels)

# 保存模型 - 修改这一行
model_state = {
    'language_model': pretrained_model.state_dict(),
    'pooler': clf_model.pooler.state_dict(),
    'classifier': clf_model.classifier.state_dict()
}
torch.save(model_state, 'sentiment_model.pth')

# print(pretrain_data[:5])
# print(sentiment_labels[:5])