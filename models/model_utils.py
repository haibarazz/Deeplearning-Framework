# 这个里面可以放一些就是我们模型通用的一些组件，比如MLP，门控之类的
import torch 
import torch.nn as nn
import torch_scatter
import numpy as np
import torch.nn.functional as F

class Embeding_layer(nn.Module):
    def __init__(self,feature_columns,embed_dim):
        # 这是一个通用的embeding层，可以同时对连续变量和离散变量进行embeding
        super().__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        self.len_dense = len(self.dense_feature_cols)       
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=embed_dim)
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        self.linear_layers = nn.ModuleList([nn.Linear(1,embed_dim ) for _ in enumerate(self.dense_feature_cols)])
        self.input_norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        dense_input, sparse_inputs = x[:, :self.len_dense], x[:, self.len_dense:]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_'+str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds, dim=1)
        dense_embeds = [self.linear_layers[i](dense_input[:, i].unsqueeze(-1)) for i in range(dense_input.shape[1])]
        dense_embeds = torch.stack(dense_embeds, dim=1)
        combined_embeds = torch.cat([dense_embeds, sparse_embeds], dim=1)    # [batch,num_nodes,embed]
        combined_embeds = self.input_norm(combined_embeds)

        return combined_embeds
    
class encoder(nn.Module):
    # 一个简单的encoder
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    


class BaselineLSTM(nn.Module):
    # 一个简单的lstm的实现
    def __init__(self,hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.no_history = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, seq, lengths):
        batch_size = seq.size(0)
        valid_mask = lengths > 0
        if valid_mask.sum() == 0:
            return self.no_history.unsqueeze(0).expand(batch_size, -1)
        valid_lengths = lengths[valid_mask]
        valid_seq = seq[valid_mask]
        packed = nn.utils.rnn.pack_padded_sequence(
            valid_seq, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_outputs = h_n[-1]
        valid_outputs = self.proj(last_outputs)
        full_outputs = self.no_history.unsqueeze(0).expand(batch_size, -1).clone()
        full_outputs[valid_mask] = valid_outputs
        return full_outputs
    
class Attention_layer(nn.Module):
    # 一个简单的注意力机制
    def __init__(self, dim):
        super().__init__()
        self.attention_weights = nn.Linear(dim, 1)
        
    def forward(self, vec1, vec2):
        vectors = torch.stack([vec1, vec2], dim=1)
        attention_scores = self.attention_weights(vectors)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_output = torch.sum(vectors * attention_weights, dim=1)
        return weighted_output, attention_weights.squeeze(-1) 
    

# 注意力机制
class AdditiveAttention(nn.Module):
    # 加性注意力机制
    def __init__(self, 
                 query_dim, 
                 key_dim, 
                 hidden_dim,
                 dropout=0.2):
        super(AdditiveAttention, self).__init__()
        self.Q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.K_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.att_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, query, keys, values=None):
        if values is None:
            values = keys
        batch_size, seq_len = keys.shape[:2]   #[batch, seq_len, hidden_dim]
        query = self.Q_net(query)  # [batch_size, hidden_dim]
        keys = self.K_net(keys)      # [batch_size, seq_len, hidden_dim]
        query = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]

        # 计算注意力分数: tanh(Wq*query + Wk*keys)
        combined = torch.tanh(query + keys)  # [batch_size, seq_len, hidden_dim]
        attention_scores = self.att_weight(combined).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        attention_weights = self.dropout(attention_weights)
        output = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch_size, value_dim]
        
        return output, attention_weights
    

class CrossAttentionLayer(nn.Module):
    # 交叉注意力机制QKV，并且还带有温度，可以调节注意力分布的平滑度
    def __init__(self, 
                 hidden_dim,
                 n_heads=4, 
                 dropout=0.2,
                 temp=1.5):
        super(CrossAttentionLayer, self).__init__()
        self.d_model = hidden_dim//2
        self.n_heads = n_heads
        self.d_k =self.d_model  // n_heads
        
        assert self.d_model  % n_heads == 0, "d_model must be divisible by n_heads"
        self.W_q = nn.Linear(hidden_dim, self.d_model , bias=False)
        self.W_k = nn.Linear(hidden_dim, self.d_model , bias=False)
        self.W_v = nn.Linear(hidden_dim, self.d_model , bias=False)
        self.W_o = nn.Linear(self.d_model , hidden_dim)
        self.temp = temp
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.d_model )
        self.output_dropout = nn.Dropout(dropout)
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores / self.temp
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, 1, num_order]
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)  # [batch_size, n_heads, 1, d_k]
        return output, attention_weights

    def forward(self, query, order, mask=None):

        batch_size = query.shape[0]
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        Q_original = query
        # 计算Q, K, V
        Q= self.W_q(query)  # [batch_size, 1, d_model]
        K = self.W_k(order)  # [batch_size, num_donor_features, d_model]
        V = self.W_v(order)  # [batch_size, num_donor_features, d_model]
        # 重塑为多头注意力格式
        Q = Q.view(batch_size, 1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, 1, d_k]
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, num_donor_features, d_k]
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, num_donor_features, d_k]
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        # [ batch_size, n_heads, 1, d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, 1, self.d_model)  # [batch_size, 1, d_model]
            # [ batch_size, 1,n_heads, d_model]
        output = self.W_o(attn_output)  # [batch_size, 1, d_model]
        output = self.layer_norm(self.output_dropout(output) + Q_original)
        output = output.squeeze(1)  # [batch_size, d_model]
        attention_weights = attention_weights.mean(dim=1).squeeze(1)  # [batch_size, num_donor_features]

        return output, attention_weights