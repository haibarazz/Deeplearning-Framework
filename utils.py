# 早停和学习速率的调度模块（余弦退火法进行学习率的调度）
# 这个里面主要是各种这个工具函数之类的
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from omegaconf import DictConfig, OmegaConf
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 


class LRScheduler():
    """当验证损失（validation loss）在一定的epoch内没有减少时，我们就将以一定的factor降低学习率。"""
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: 我们正在使用的优化器
        :param patience: 等待多少个epoch我们才更新学习速率
        :param min_lr: 在更新时的最小学习率值
        :param factor: 学习率应该更新的倍数
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, f1_score):
        self.lr_scheduler.step(f1_score)

class EarlyStopping():
    """
    这个我们可以设置什么指标在多少轮不提升就早停
    """
    def __init__(self, patience=7, verbose=False, delta=0, path=".../model.pth"):
        """
        Args:
            patience (int): 最多忍受几轮模型不在提升
                            Default: 7
            verbose (bool): 是否打印相关的信息
                            Default: False
            delta (float): 模型最小提升的阈值
                            Default: 0
            path (str): 早停时模型存储的地点
                            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, temp_score, model):

        score = temp_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'早停轮数计数: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(temp_score, model)
            self.counter = 0

    def save_checkpoint(self, temp_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'验证集上的指标的变化情况 ({self.score_min:.6f} --> {temp_score:.6f}). 保存模型.......')
        torch.save(model.state_dict(), self.path)
        self.score_min = temp_score


class EarlyStopping_loss():
    def __init__(self, patience=7, verbose=False, delta=0, path=".../model.pth"):
        """
        Args:
            patience (int): 最多忍受几轮模型不在提升
                            Default: 7
            verbose (bool): 是否打印相关的信息
                            Default: False
            delta (float): 模型最小提升的阈值
                            Default: 0
            path (str): 早停时模型存储的地点
                            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.score_min = np.Inf
        self.delta = delta
        self.path = path


    def __call__(self, temp_score, model):
        # temp_score 是当前损失值
        score = temp_score  # 损失值越小越好，直接使用

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(temp_score, model)
        elif score > self.best_score - self.delta:
            # 如果当前损失值没有显著下降（小于 best_score - delta），则增加计数器
            self.counter += 1
            print(f'早停轮数计数: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果当前损失值显著下降，则更新最佳损失值并保存模型
            self.best_score = score
            self.save_checkpoint(temp_score, model)
            self.counter = 0

    def save_checkpoint(self, temp_score, model):
        '''当验证集损失下降时保存模型'''
        if self.verbose:
            print(f'验证集损失变化情况 ({self.score_min:.6f} --> {temp_score:.6f}). 保存模型.......')
        torch.save(model.state_dict(), self.path)
        self.score_min = temp_score


def encode_multiclass_labels(train_data, test_data, target_columns):
    """
    对多分类标签进行编码
    """
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for target_col in target_columns:
        print(f"  处理标签列: {target_col}")
        
        # 检查原始标签分布
        print(f"    训练集标签分布: {train_data[target_col].value_counts().sort_index()}")
        print(f"    测试集标签分布: {test_data[target_col].value_counts().sort_index()}")
        # 创建标签编码器
        le = LabelEncoder()
        # 合并训练和测试集的标签，确保编码一致性
        all_labels = pd.concat([train_data[target_col], test_data[target_col]])
        le.fit(all_labels)
        # 应用编码
        train_data[target_col] = le.transform(train_data[target_col])
        test_data[target_col] = le.transform(test_data[target_col])
        # 保存编码器
        label_encoders[target_col] = le    
    return train_data, test_data, label_encoders

