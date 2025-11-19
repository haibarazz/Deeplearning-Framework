import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNBase(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=64, 
                 n_layers=2, 
                 dropout=0.1, 
                 rnn_type="rnn", 
                 bidirectional=False, 
                 num_classes=1, 
                 task_type="classification"):
        super().__init__()
        self.name = rnn_type.upper()
        self.task_type = task_type
        rnn_cls = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }[rnn_type]
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, num_classes if task_type == "Multiclass" else 1)

    def forward(self, batch):
        x = batch["basic_features"]  # [B, T, D]
        # 支持输入为二维 [B, D] 时自动扩展时间维度
        if x.dim() == 2:
            x = x.unsqueeze(1)
        output, hidden = self.rnn(x)
        if isinstance(hidden, tuple):  # LSTM (h_n, c_n)
            hidden = hidden[0]
        # 取最后一层的隐藏状态
        last_hidden = hidden[-1]
        logits = self.head(self.dropout(last_hidden))
        return {"logits": logits.squeeze(-1)}


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, n_heads=4, dropout=0.1, num_classes=1, task_type="classification"):
        super().__init__()
        self.name = "Transformer"
        self.task_type = task_type
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, num_classes if task_type == "Multiclass" else 1)

    def forward(self, batch):
        x = batch["basic_features"]  # [B, T, D] 或 [B, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        enc = self.encoder(h)
        pooled = self.norm(enc.mean(dim=1))
        logits = self.cls_head(pooled)
        return {"logits": logits.squeeze(-1)}


def build_model(model_type: str, input_dim: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1, **kwargs):
    task_type = kwargs.get("task_type", "classification")
    num_classes = kwargs.get("num_classes", 1)
    bidirectional = kwargs.get("bidirectional", False)
    if model_type.lower() == "rnn":
        return RNNBase(input_dim, hidden_dim, n_layers, dropout, rnn_type="rnn", bidirectional=bidirectional, num_classes=num_classes, task_type=task_type)
    if model_type.lower() == "gru":
        return RNNBase(input_dim, hidden_dim, n_layers, dropout, rnn_type="gru", bidirectional=bidirectional, num_classes=num_classes, task_type=task_type)
    if model_type.lower() == "lstm":
        return RNNBase(input_dim, hidden_dim, n_layers, dropout, rnn_type="lstm", bidirectional=bidirectional, num_classes=num_classes, task_type=task_type)
    if model_type.lower() == "transformer":
        n_heads = kwargs.get("n_heads", 4)
        return TransformerModel(input_dim, hidden_dim, n_layers, n_heads, dropout, num_classes=num_classes, task_type=task_type)
    raise ValueError(f"Unsupported model_type: {model_type}")