# 深度学习通用训练模板 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-持续更新-yellow.svg)](https://github.com/haibarazz/zz)

> 一个基于 PyTorch 和 Hydra 的深度学习通用训练框架，支持快速模型开发、训练和评估。

## 📋 项目简介

这是一个通用的深度学习训练模板项目，旨在提供一个标准化、模块化的训练流程框架。通过本项目，你可以：

- 🎯 **快速搭建模型**：提供标准化的模型开发模板
- ⚙️ **灵活配置管理**：使用 Hydra 进行超参数管理
- 📊 **完整训练流程**：包含训练、验证、测试全流程
- 🔄 **断点续训**：支持模型检查点保存和加载
- 📈 **实时监控**：训练过程可视化和早停机制
- 🛠️ **工具丰富**：提供数据预处理、评估指标等实用工具

## 🏗️ 项目结构

```
通用的模板/
├── config/                  # 配置文件目录
│   ├── config.yaml         # 主配置文件
│   └── models/             # 模型配置目录
│       ├── model1.yaml     # 模型1配置
│       └── model2.yaml     # 模型2配置
├── dataset/                # 数据集目录
│   └── data.csv           # 示例数据
├── models/                 # 模型定义目录
│   ├── model_utils.py     # 模型工具函数
│   ├── model1.py          # 模型1实现
│   └── model2.py          # 模型2实现
├── checkpoints/           # 模型检查点保存目录
├── Data_pre.py            # 数据预处理模块
├── engine.py              # 训练引擎核心代码
├── main.py                # 主程序入口
├── utils.py               # 通用工具函数
└── README.md              # 项目说明文档
```

## ✨ 主要特性

### 1. 基于 Hydra 的配置管理
- 支持多配置文件组合
- 命令行参数覆盖
- 配置版本管理

### 2. 模块化训练流程
- **数据预处理**：标准化的数据加载和预处理流程
- **模型训练**：支持分类和回归任务
- **模型评估**：丰富的评估指标（Accuracy, F1, AUC, MCC等）
- **早停机制**：防止过拟合
- **学习率调度**：自动调整学习率

### 3. 灵活的模型切换
只需修改配置文件即可切换不同模型，无需改动代码：
```yaml
defaults:
  - model: model1  # 切换为 model2 即可使用不同模型
```

### 4. 断点续训
支持从检查点恢复训练，节省时间和资源：
```python
python main.py checkpoint_path=checkpoints/model.pth
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- torch
- hydra-core
- pandas
- numpy
- scikit-learn
- tqdm

### 运行示例

1. **训练模型**
```bash
python main.py
```

2. **使用不同配置**
```bash
python main.py model=model2 training.epochs=200
```

3. **从检查点恢复**
```bash
python main.py checkpoint_path=checkpoints/model.pth
```

4. **修改超参数**
```bash
python main.py training.learning_rate=0.0001 data.batch_size=64
```

## 📝 配置说明

### 主配置文件 (config.yaml)

```yaml
# 数据配置
data:
  data_dir: "dataset/data.csv"
  batch_size: 128
  train_ratio: 0.8

# 训练配置
training:
  learning_rate: 0.001
  epochs: 150
  weight_decay: 5e-5
  early_stop_patience: 20
  task_type: classification  # classification 或 regression

# 设备配置
device: "cuda"  # cuda 或 cpu
```

### 模型配置示例 (models/model1.yaml)

```yaml
_target_: models.model1.Mymodel1
input_dim: 10
hidden_dim: 64
n_layers: 3
dropout: 0.3
```

## 🔧 自定义开发

### 添加新模型

1. 在 `models/` 目录下创建新的模型文件
2. 在 `config/models/` 下添加对应配置文件
3. 在主配置中切换模型

### 自定义数据处理

修改 `Data_pre.py` 中的 `load_and_preprocess_data` 函数以适应你的数据格式。

### 添加新的评估指标

在 `engine.py` 的 `BaseTrainer` 类中添加自定义评估函数。

## 📊 支持的评估指标

- **分类任务**：Accuracy, Precision, Recall, F1-Score, AUC, MCC, AUPRC
- **回归任务**：MSE, MAE, RMSE, R²
- **自定义指标**：可扩展

## 🛣️ 开发路线图

- [x] 基础训练框架
- [x] 多模型支持
- [x] 断点续训功能
- [x] 早停和学习率调度
- [ ] TensorBoard 集成
- [ ] 分布式训练支持
- [ ] 更多预训练模型
- [ ] 自动超参数调优
- [ ] 模型可解释性工具
- [ ] Docker 容器化部署

> **注意**：本项目将持续更新和维护，欢迎提出建议和贡献代码！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：
- Email: 2812156857@qq.com
⭐ 如果这个项目对你有帮助，欢迎给个 Star！

**最后更新时间**: 2025年11月7日
