import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
class FEATURE(object):
    num_fea = []
    cat_fea = []

# 数据读取


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



def order_data_embedding(df,cat_col,num_col):
    """
    order_data_embedding 这是我们对离散变量和这个连续变量进行这个embeding的时候常用的一个辅助函数

    :param df: 输入的数据
    :param cat_col: 分类特征列名列表
    :param num_col: 数值特征列名列表
    """
    def denseFeature(feat):
        return {'feat': feat}
    def sparsFeature(feat, feat_num):
        return {'feat': feat, 'feat_num': feat_num}
    feature_columns = [
        [denseFeature(feat) for feat in num_col], 
        [sparsFeature(feat, len(df[feat].unique())) for feat in cat_col]]
    return feature_columns


class BaseDataset(Dataset):

    def __init__(self, df):
        """
        这个里面也可以放一些数据处理的操作，数据增强之类的
        """
        self.df = df.reset_index(drop=True)

        basic_features = FEATURE.num_order_feat + FEATURE.cat_order_feat
        self.feature = torch.FloatTensor(df[basic_features].values)
        self.labels = torch.FloatTensor(df['label'].values)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            # 基础特征
            'basic_features': self.feature[idx],
            'label': self.labels[idx],
        }

def prepare_data(df,cfg):

    batch_size = cfg.data.batch_size
    train_ratio = cfg.data.train_ratio
    random_state = cfg.data.random_state


    # 划分训练测试集
    train_df, test_df = train_test_split(
        df, test_size=1-train_ratio, 
        random_state=random_state, stratify=df['d2c_label']
    )
    # 注意数据预处理一定要放在这个划分数据集之后做
    train_df, test_df,_ = data_process(train_df, test_df, FEATURE.cat_order_feat, FEATURE.num_order_feat+FEATURE.num_debe_feat) #FEATURE.num_true +FEATURE.num_debe_feat

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