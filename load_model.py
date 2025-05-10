import torch
from model_architecture import CustomLanguageModel
from training_tasks import SentimentClassificationTask

# 1. 加载词汇表和配置
vocab = {'<pad>': 0, '<unk>': 1, '<mask>': 2, 'hello': 3, 'world': 4, 'good': 5, 'bad': 6}
vocab_size = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 初始化模型架构
language_model = CustomLanguageModel(
    vocab_size=vocab_size,
    d_model=128,
    nhead=4,
    num_layers=2,
    max_seq_len=128
).to(device)

# 3. 初始化分类任务
sentiment_model = SentimentClassificationTask(
    model=language_model,
    num_classes=2,
    pad_token_id=vocab['<pad>']
)

# 4. 加载保存的模型状态
model_state = torch.load('sentiment_model.pth')

# 5. 将状态加载到模型组件中
language_model.load_state_dict(model_state['language_model'])
sentiment_model.pooler.load_state_dict(model_state['pooler'])
sentiment_model.classifier.load_state_dict(model_state['classifier'])

# 6. 设置为评估模式
language_model.eval()

# 如果需要，可以单独设置 pooler 和 classifier 为评估模式
sentiment_model.pooler.eval()
sentiment_model.classifier.eval()

print("模型加载成功！")

# 7. 示例：使用模型进行预测
def predict_sentiment(text):
    # 文本转换为token ID
    tokens = [vocab.get(word, vocab['<unk>']) for word in text.split()]
    token_tensor = torch.tensor([tokens]).to(device)
    
    # 预测 - 修改这一行，直接调用forward方法
    with torch.no_grad():
        logits = sentiment_model.forward(token_tensor)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    sentiment = "正面" if prediction == 1 else "负面"
    return sentiment, probs[0].tolist()

# 测试一些样例
test_samples = [
    "hello world this is good",
    "this product is bad",
    "I like the service",
    "the experience was terrible",
    "good job on the project",
    "bad experience with the product",
]

for sample in test_samples:
    sentiment, probs = predict_sentiment(sample)
    print(f"文本: '{sample}'")
    print(f"情感预测: {sentiment}")
    print(f"概率分布: 负面={probs[0]:.4f}, 正面={probs[1]:.4f}\n")