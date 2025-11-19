# 这个里面是我们构造数据集的代码,在深度学习里面，构建数据集是十分重要的一环，
# 这里面可能回涉及到各种这个数据增强什么其他的手段
"""
Yang Zhou 
created on 2025-11-10
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Sequence
from Feature_selected import FEATURE



class BaseDataset(Dataset):
    """
    这只是一个最基础的数据处理
    """
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