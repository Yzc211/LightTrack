"""
(2021.06.27)
Transformer encoder class (Lite version)
-- Only use one layer of encoder
-- search region as queries, "concat" as keys and values
-- only pass the search region to the FFN
-- functions takes standard pytorch Tensor as input (for TensorRT)
"""
from typing import Optional
import torch.nn.functional as F
from torch import nn, Tensor
import torch



class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class MultiHeadSparseAttention_2(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_topk: float = 0.3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Define learnable topk parameter
        self.topk = nn.Parameter(torch.tensor(initial_topk), requires_grad=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        L, N, E = query.size()
        S = key.size(0)
        query = self.q_linear(query).view(L, N, self.num_heads, self.head_dim).transpose(0, 1)
        key = self.k_linear(key).view(S, N, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.v_linear(value).view(S, N, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = torch.einsum('nlhd,nshd->nlhs', query, key) / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply dynamic top-k sparsity
        k = int(attn_scores.size(-1) * torch.sigmoid(self.topk))
        topk_values, topk_indices = torch.topk(attn_scores, k, dim=-1)
        topk_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_indices, True)
        attn_scores = attn_scores.masked_fill(~topk_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('nlhs,nshd->nldh', attn_weights, value)

        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, E)
        attn_output = self.out_linear(attn_output)

        return attn_output

class MultiHeadSparseAttention_1(nn.Module):
    def __init__(self, embed_dim, num_heads, topk: float = 0.3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.topk = topk

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        A multi-head sparse attention mechanism.
        :param query: Query tensor of shape (L, N, E)
        :param key: Key tensor of shape (S, N, E)
        :param value: Value tensor of shape (S, N, E)
        :param key_padding_mask: Key padding mask of shape (N, S)
        :return: Output tensor of shape (L, N, E)
        """
        L, N, E = query.size()
        S = key.size(0)
        # Linear transformations
        query = self.q_linear(query).view(L, N, self.num_heads, self.head_dim).transpose(0, 1)  # Shape: (N, L, H, D)
        key = self.k_linear(key).view(S, N, self.num_heads, self.head_dim).transpose(0, 1)  # Shape: (N, S, H, D)
        value = self.v_linear(value).view(S, N, self.num_heads, self.head_dim).transpose(0, 1)  # Shape: (N, S, H, D)

        # Compute attention scores
        attn_scores = torch.einsum('nlhd,nshd->nlhs', query, key) / (self.head_dim ** 0.5)  # Shape: (N, L, H, S)

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Apply top-k sparsity
        k = int(attn_scores.size(-1) * self.topk)
        topk_values, topk_indices = torch.topk(attn_scores, k, dim=-1)
        topk_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_indices, True)
        attn_scores = attn_scores.masked_fill(~topk_mask, float('-inf'))

        # Compute softmax and attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('nlhs,nshd->nldh', attn_weights, value)  # Shape: (N, L, H, D)

        # Concatenate heads and apply linear transformation
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, E)  # Shape: (L, N, E)
        attn_output = self.out_linear(attn_output)

        return attn_output
'''---------------------------------------------------------------------------------------------------------'''
class TransformerEncoderLayerLite(nn.Module):
    """One lite encoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # d_model=128

       # plan3
        self.self_attn = MultiHeadSparseAttention_2(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = LowRankLinear(d_model, dim_feedforward, rank=64)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LowRankLinear(dim_feedforward, d_model, rank=64)

        #ori
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 128 8 0.1
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)  # 0.1
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None):
        """ q, k, v denote queries, keys, and values respectively """
        # s = time.time()

        src2 = self.self_attn(q,k,v,key_padding_mask=key_padding_mask) # SparseAttention plan3
        # src2 = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)[0] #ori

        src = q + self.dropout1(src2)
        src = self.norm1(src)
        # e1 = time.time()

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # e2 = time.time()
        # print("self-attention time: %.1f" % ((e1-s) * 1000))
        # print("MLP time: %.1f" % ((e2-e1) * 1000))
        return src


class TransformerEncoderLite(nn.Module):
    """search feature as queries, concatenated feature as keys and values"""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model # 128
        self.nhead = nhead  # 8
        self.d_feed = dim_feedforward # 1024
        self.encoder = TransformerEncoderLayerLite(d_model, nhead, dim_feedforward, dropout, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None):
        memory = self.encoder(q, k, v, key_padding_mask=key_padding_mask)
        return memory


def build_lite_encoder(cfg):
    print("Building lite transformer encoder...")
    encoder = TransformerEncoderLite(d_model=cfg.MODEL.HIDDEN_DIM, dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
                                     nhead=cfg.MODEL.TRANSFORMER.NHEADS,
                                     dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD)
    return encoder


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if activation == "glu":
        return F.glu
    if activation == "relu6":
        return F.relu6
    if activation == "swish":
        return lambda x: x * torch.sigmoid(x)
    raise RuntimeError(F"activation should be relu/gelu/glu/relu6/swish, not {activation}.")
