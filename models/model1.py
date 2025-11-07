import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from .model_utils import encoder, HypergraphConv, BaselineLSTM,encoder_order


class Mymodel1(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_layers,
                 dropout):
        super().__init__()
        
        # 定义 MLP 的层
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 使用 nn.Sequential 将层组合
        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        # ---- 编码原始输入 ----
        x = data  # 假设输入数据为 data
        x = self.mlp(x)
        # 这里之所以用一个字典，因为我们后续很多时候，要返回的不仅仅是我们的预测值
        # 多任务的时候要返回多个预测值，有时候还要返回一些中间变量供损失函数使用；或者是注意力权重等等
        return {
            'output': x
        }




