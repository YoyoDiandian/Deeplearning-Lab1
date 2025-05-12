# BERT-based Emotion Recognition System

## 项目概述

本项目实现了一个基于BERT的六分类情感识别系统，能够将文本自动分类为六种不同的情感类别。系统采用了一系列优化策略，包括混合精度训练、动态批处理、类别不平衡处理等，使模型在测试集上达到了85.94%的准确率。

## 技术特点

- **增强型BERT架构**：设计了多层次分类头部结构，提升模型对情感细微差别的感知能力
- **参数选择性冻结**：仅微调BERT最后4层，保留预训练知识的同时提高训练效率
- **类别不平衡处理**：结合加权损失函数与加权采样技术，平衡各类别性能
- **混合精度训练**：使用PyTorch的AMP功能，显著减少内存占用，提高训练速度
- **可视化分析**：自动生成多种训练曲线图表，便于模型性能分析

## 环境依赖

```
torch>=1.9.0
transformers>=4.10.0
numpy>=1.19.5
matplotlib>=3.3.4
```

## 项目结构

```
.
├── main.ipynb        # 主要代码文件，包含训练和评估逻辑
├── model_output/     # 模型输出目录
│   ├── best_model.pt        # 验证损失最低的模型
│   ├── best_model_acc.pt    # 验证准确率最高的模型
│   ├── training_history.csv # 训练历史记录
│   └── plots/        # 训练曲线图表
│       ├── accuracy_curve.pdf
│       ├── all_metrics.pdf
│       ├── loss_curve.pdf
│       └── metrics_curve.pdf
├── README.md         # 项目说明文档
├── training.csv      # 训练数据集（需自行添加）
├── validation.csv    # 验证数据集（需自行添加）
└── test.csv          # 测试数据集（需自行添加）
```

## 使用方法

### 1. 数据准备

准备训练、验证和测试数据集，每个文件应为CSV格式，包含两列：文本和标签。标签为0-5之间的整数，代表不同情感类别。

### 2. 模型训练

运行main.ipynb中的训练代码：

```python
# 设置参数
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPSILON = 1e-8
EPOCHS = 15
MAX_LENGTH = 128
SAVE_PATH = 'model_output'
TRAIN_FILE = 'training.csv'
VAL_FILE = 'validation.csv'
USE_CLASS_WEIGHTS = True
USE_WEIGHTED_SAMPLER = True

# 运行主函数
main()
```

### 3. 模型评估与预测

训练完成后，可使用notebook中的第二个代码块评估模型性能并进行预测：

```python
# 设置参数
MODEL_PATH = 'model_output/best_model_acc.pt'
TEST_FILE = 'test.csv'
MAX_LENGTH = 128
BATCH_SIZE = 32

# 运行评估主函数
main()
```

## 模型性能

- **准确率**: 85.94%
- **精确率**: 79.57%
- **召回率**: 85.22%
- **F1分数**: 81.48%

## 模型架构

- **基础模型**: BERT-base-uncased（12层Transformer编码器，768维隐藏层，12个注意力头）
- **参数冻结**: 仅最后4层参数可训练
- **分类头**: 三层结构（768→768→384→6），每层配备LayerNorm、GELU激活和Dropout
- **损失函数**: 加权交叉熵损失，权重基于类别频率的倒数

## 许可证

[MIT License](https://opensource.org/licenses/MIT)

## 致谢

本项目使用了Hugging Face的Transformers库和PyTorch框架，感谢这些开源项目的贡献。