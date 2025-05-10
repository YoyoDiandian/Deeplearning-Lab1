import random

# 扩展情感词汇库和句式模板
positive_words = [
    "joyful", "amazing", "wonderful", "successful", "love", 
    "bright", "perfect", "excited", "peaceful", "grateful",
    "thrilled", "delighted", "optimistic", "fantastic", "inspiring"
]
negative_words = [
    "awful", "terrible", "failed", "ruined", "broken", 
    "sad", "angry", "disappointing", "painful", "hopeless",
    "frustrating", "miserable", "disastrous", "unbearable", "dreadful"
]

# 多样化句式模板
positive_templates = [
    "The weather today is absolutely {word} for a picnic.",
    "I feel {word} about this new opportunity!",
    "My best friend surprised me with a {word} gift.",
    "This is one of the most {word} moments of my life.",
    "I'm so {word} to see the progress we've made."
]
negative_templates = [
    "The service at the restaurant was {word} and ruined my evening.",
    "I can't believe how {word} this situation turned out.",
    "This decision feels completely {word} to everyone involved.",
    "The movie was a {word} waste of time and money.",
    "I'm deeply {word} by the recent news."
]

# 生成1000个句子和标签
sentences = []
labels = []
for _ in range(1000):
    is_positive = random.choice([True, False])
    if is_positive:
        word = random.choice(positive_words)
        template = random.choice(positive_templates)
        label = 1
    else:
        word = random.choice(negative_words)
        template = random.choice(negative_templates)
        label = 0
    sentence = template.format(word=word)
    sentences.append(sentence)
    labels.append(label)

# 打印前10句示例及标签
print("=== 示例句子（前10句） ===")
for i in range(10):
    print(f"{i+1}. {sentences[i]} → Label: {labels[i]}")

print("\n=== 完整标签数组（前10个元素） ===")
print(labels[:10], "...")

# 可选：保存到本地文件（CSV格式）
import csv
with open("sentences_and_labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Sentence", "Label"])
    for s, l in zip(sentences, labels):
        writer.writerow([s, l])
print("\n数据已保存至 sentences_and_labels.csv")