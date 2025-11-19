"""
生成示例序列数据，便于快速跑通 RNN/GRU/LSTM/Transformer。

输出: dataset/data.csv
特征: num_0..num_4 连续特征, cat_0..cat_2 离散特征, label 二分类标签。
"""

import numpy as np
import pandas as pd
from pathlib import Path


def make_sequences(num_samples: int = 2000, seq_len: int = 10, noise: float = 0.3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num_cols = [f"num_{i}" for i in range(5)]
    cat_cols = [f"cat_{i}" for i in range(3)]

    # 构造基础模式: 某些数值特征与标签正相关，某些负相关
    base_signal = rng.normal(0, 1, size=(num_samples, seq_len, len(num_cols)))
    weights = np.array([1.2, 0.8, -1.0, 0.5, -0.7])
    scores = (base_signal * weights).sum(axis=(-1, -2))  # 累计得分

    # 类别特征: 0/1/2 分布
    cats = rng.integers(0, 3, size=(num_samples, seq_len, len(cat_cols)))
    cat_bonus = (cats == 1).sum(axis=(-1, -2)) * 0.2

    logits = scores + cat_bonus + rng.normal(0, noise, size=num_samples)
    prob = 1 / (1 + np.exp(-logits))
    labels = (prob > 0.5).astype(int)

    # 展平为表格存储: 简化为对时间维求均值作为特征示例
    mean_nums = base_signal.mean(axis=1)
    mode_cats = cats.reshape(num_samples, seq_len * len(cat_cols))
    mode_cats = mode_cats[:, :len(cat_cols)]  # 取首个时间步代表类别

    data = pd.DataFrame(mean_nums, columns=num_cols)
    for i, col in enumerate(cat_cols):
        data[col] = mode_cats[:, i]
    data["label"] = labels
    return data


def main():
    out_path = Path("dataset/data.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = make_sequences()
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path.resolve()} with shape {df.shape}")


if __name__ == "__main__":
    main()