import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from Feature_selected import FEATURE
from Dataset import BaseDataset
"""
准备数据的一个流程
读取数据-数据集划分-特征归一化-dataset构建
"""


def load_and_preprocess_data(cfg):
    """数据加载和预处理"""
    print("开始加载和预处理数据...")
    df = pd.read_csv(cfg.data.data_dir)
    # df = df.iloc[:1000]  # 这个可以帮助我们快速测试自己的模型是否有bug
    # 这里可以用一些就是数据的一些map的操作
    return df


def data_process(train_data, test_data, cat_features, num_features):
    """
    数据预处理函数
    Args:
        train_data (DataFrame): 训练集数据
        test_data (DataFrame): 测试集数据
        cat_features (list): 分类特征列名列表
        num_features (list): 数值特征列名列表
    
    Returns:
        DataFrame: 处理后的训练集和测试集数据
        dict: 编码器字典（用于后续处理新数据）
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # 创建副本避免修改原数据
    train_processed = train_data.copy()
    test_processed = test_data.copy()
    
    # 创建字典存储编码器和标准化器
    encoders = {}
    scalers = {}
    
    # 对分类特征进行标签编码
    for col in cat_features:
        if col in train_processed.columns:
            train_processed[col] = train_processed[col].astype(str)
            test_processed[col] = test_processed[col].astype(str)
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col])
            test_processed[col] = le.transform(test_processed[col])
            encoders[col] = le
    
    # 对数值特征进行标准化
    for col in num_features:
        if col in train_processed.columns:
            # 处理缺失值（用中位数填充）
            train_processed[col] = train_processed[col].fillna(train_processed[col].median())
            test_processed[col] = test_processed[col].fillna(train_processed[col].median())
            scaler = StandardScaler()
            train_processed[col] = scaler.fit_transform(train_processed[[col]]).flatten()
            test_processed[col] = scaler.transform(test_processed[[col]]).flatten()
            scalers[col] = scaler
    
    return train_processed, test_processed, {'encoders': encoders, 'scalers': scalers}




def prepare_data(df,cfg):

    batch_size = cfg.data.batch_size
    train_ratio = cfg.data.train_ratio
    random_state = cfg.data.random_state


    # 划分训练测试集
    train_df, test_df = train_test_split(
        df, test_size=1-train_ratio, 
        random_state=random_state, stratify=df[FEATURE.label]
    )
    # 注意数据预处理一定要放在这个划分数据集之后做
    train_df, test_df,_ = data_process(train_df, test_df, FEATURE.cat_order_feat, FEATURE.num_order_feat) 

    # 创建数据集
    train_dataset = BaseDataset(train_df)
    test_dataset = BaseDataset(test_df)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader