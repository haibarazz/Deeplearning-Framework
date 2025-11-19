# 这个里面是我们定义我们自己特征的名称，很多时候，我们的特征都是各种各样的，所以我们要把这个特征明
# 最好给他列出来
"""
Yang Zhou 
created on 2025-11-10
"""
import pandas as pd
import numpy as np
import torch
class FEATURE_example(object):
    """
    特征类，一般情况下，我们会把这个离散特征和连续特征的列名分开，方便我们数据预处理，注意我们尽量这么写
    """
    num_fea = []
    cat_fea = []
    label = 'label'

class FEATURE(object):
    # 一个简单的例子
    # 默认的顺序特征与离散特征列名，便于直接跑通示例
    num_order_feat = [f"num_{i}" for i in range(5)]
    cat_order_feat = [f"cat_{i}" for i in range(3)]

    label = ['label']

