# 深度学习训练模板（Hydra 进阶参数管理版）🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Hydra-1.x-0099cc.svg)](https://hydra.cc/)
[![Status](https://img.shields.io/badge/Status-持续更新-yellow.svg)](https://github.com/haibarazz/Deeplearning-Framework)

> 针对多模型、快速试验场景打造的训练脚手架，重点强化了 **参数管理与配置组合能力**，让「换模型/改流程」只需更新配置即可。

## ✨ 项目亮点

- �️ **进阶参数管理**：基于 Hydra 组合式配置，`configs/model/*.yaml` 与 `configs/training_loop/*.yaml` 可自由拼装，`main.py` 中会根据配置动态计算输入维度、实例化模型并注入任务类型。
- 🧠 **多任务友好**：同一个训练循环可在二分类、多分类、回归间切换，指标、损失函数、头部结构都会自动对齐。
- 🪄 **断点续训即插即用**：`load_training` 会从指定 checkpoint 恢复模型、优化器及调度器状态，无缝衔接训练历史。
- � **可追溯训练日志**：`result/` 下自动生成日志与 CSV 记录，便于后续分析与可视化。
- 🧱 **贴合实际目录结构**：所有脚本、配置、数据、模型、输出全部按模块归位，下方结构即当前仓库真实布局。

## 🗂️ 项目结构

```
Deeplearning-Framework/
├── configs/
│   ├── config.yaml                 # 顶层入口配置（defaults 组合）
│   ├── model/                      # 模型超参组（lstm、gru、transformer...）
│   └── training_loop/              # 训练/优化策略（learning rate、epoch 等）
├── dataset/
│   └── data.csv                    # 示例数据
├── models/
│   ├── rnn_models.py               # RNN/GRU/LSTM/Transformer 统一实现
│   ├── model1.py / model2.py       # 自定义样例模型
│   └── model_utils.py              # 额外工具
├── checkpoints/                    # 断点权重输出
├── outputs/                        # Hydra 默认输出（保留历史 log）
├── result/                         # 自定义训练日志与指标记录
├── Data_pre.py                     # 数据加载+特征工程
├── Dataset.py                      # Dataset 封装
├── Feature_selected.py             # 特征列定义
├── engine.py                       # 训练/验证流程 + 指标
├── main.py                         # 入口，负责组合配置与启动训练
├── utils.py                        # EarlyStopping、LR 调度等
├── requirements.txt
└── README.md
```

## 🚀 快速上手

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行默认实验**（默认 Transformer + classification）
   ```bash
   python main.py
   ```

3. **替换模型或训练策略**
   ```bash
   python main.py model=lstm training_loop=default
   python main.py model=gru training_loop=trainer/base
   ```

4. **调整单个超参**（所有字段均可在命令行覆盖）
   ```bash
   python main.py training_loop.learning_rate=3e-4 training_loop.batch_size=256
   ```

5. **断点续训**
   ```bash
   python main.py resume.checkpoint=checkpoints/best_lstm_model.pth
   ```

## 🎛️ 配置体系（参数管理升级点）

- `configs/config.yaml` 使用 Hydra `defaults` 声明组合：
  ```yaml
  defaults:
    - model: transformer
    - training_loop: default
    - trainer: base
    - _self_
  ```
- 模型组内只需关注自身超参（输入维度由 `main.py` 在运行时根据 `Feature_selected.py` 自动计算，不再写死）。
- 训练循环配置专注在 `batch_size / train_ratio / random_state / optimizer 超参`，与模型解耦。
- `best_model_path`、`log_path` 等公共路径统一放在主配置，方便所有组合共享。
- 通过 `OmegaConf` + `hydra.utils.instantiate`，新增模型/训练策略只需添加 YAML 文件并在 `defaults` 中引用，无需修改 Python 代码。

## 🧰 训练流程速览

1. `Data_pre.py`：解析配置路径 → 读取 CSV → split → 标准化/编码 → 构建 `BaseDataset`。
2. `models/rnn_models.py`：基于配置动态实例化 RNN/GRU/LSTM/Transformer，自动匹配分类/多分类/回归输出。
3. `engine.BaseTrainer`：
   - 统一的 `train/train_from_epoch`，支持断点恢复。
   - 自动调度损失函数、指标与 AUC/AUPRC/F1/MCC 等评估。
   - `ReduceLROnPlateau` + `EarlyStopping` 双保险。
   - 日志 & 指标 CSV 落地，方便可视化或追溯。

## � 扩展指引

- **新增模型**：在 `models/` 下实现 → `configs/model/xxx.yaml` 中声明 `_target_` + 超参 → 在命令行 `model=xxx` 即可使用。
- **新增训练策略**：在 `configs/training_loop/` 添加 YAML（如不同的 learning rate、patience、task_type），默认 `main.py` 会读取并自动适配。
- **新增数据流程**：
  - 常规结构直接在 `Data_pre.py` 扩展函数或新增模块。
  - 复杂场景可以通过即将上线的「数据流程配置组」将 time-series / tabular / 多标签 等策略标准化（见下方路线图）。

## 🧭 路线图（即将上线）

- [x] Hydra 组合式参数管理、日志持久化
- [x] RNN/GRU/LSTM/Transformer 模型族
- [ ] 各种图神经网络的模型 GCN，GAT，GraphSAGE 等
- [ ] 推荐系统模型库：Wide&Deep、DeepFM、TabNet、Temporal Fusion Transformer 等
- [ ] 一个benchmark文件夹，可以在你的任务上创建benchmark
- [ ] 超参搜索（Hydra Sweeper + Optuna）

欢迎在 Issue 中提交你希望优先支持的模型与流程 🙌

## 🤝 贡献指南

1. Fork 仓库并创建分支：`git checkout -b feature/awesome`
2. 根据需要新增配置文件或 Python 模块
3. 运行 `python main.py` 确认无误
4. 提交 PR，并说明修改的配置组合/训练场景

## 📮 联系方式

- Email: 2812156857@qq.com
- Issues: 欢迎在 GitHub Repo 中沟通

如果该模板对你有帮助，欢迎 star ⭐ 支持我们继续完善更多模型与数据流程！

**最后更新时间**：2025-11-20
