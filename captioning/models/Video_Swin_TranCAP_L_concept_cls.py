# 新增脚本


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/zhen_chen/code/surg_caption/SwinMLP_TranCAP-HK/captioning/models")
import utils
# from . import utils

import copy
import math
import numpy as np

# from .CaptionModel import CaptionModel
from AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel
# from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

from einops import rearrange, repeat # cz add


VIDEO_CLIP_LEN = 5
VOCAB_CONCEPT_NUM = 42+1
CAPTION_LEN = 30
# VOCAB_CONCEPT_NUM = 40
# CAPTION_LEN = 22

VISUAL_OUT_NUM = 196
# VISUAL_OUT_NUM = 98
TEXT_IN_NUM =29



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src):
        return self.encoder(self.src_embed(src)) # 先对src做embed，再做encoder
    
    def decode(self, memory, src_mask, tgt, tgt_mask): # # very first src_mask is used in decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# ================================================ Encoder (start) ========================================================================== #
# Encoder
# The encoder is composed of a stack of N=6 identical layers
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask): # 每层使用相同mask
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # 输出时套上layer norm

# We employ a residual connection around each of the two sub-layers, followed by layer normalization
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)).
# where Sublayer(x) is the function implemented by the sub-layer itself. We apply dropout (cite) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model=512.
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x))) # ====================== Norm, then Feed Forward, then Add 

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # original one
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # # for other token mixer: pooling
        # x = self.sublayer[0](x, lambda x: self.self_attn(x))

        return self.sublayer[1](x, self.feed_forward)
# ================================================ Encoder (End) ========================================================================== #


# ================================================ Decoder (Start) ======================================================================== #
# Decoder
# The decoder is also composed of a stack of N = 6  identical layers
"""
Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N_dec),
"""
# class Decoder(nn.Module):
#     "Generic N layer decoder with masking."
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N) # layer <- DecoderLayer
#         self.norm = LayerNorm(layer.size)
        
#     def forward(self, x, memory, src_mask, tgt_mask):
#         # print('src_mask in decoder', src_mask.shape) # src_mask in decoder torch.Size([5, 1, 256])
#         for layer in self.layers:
#             x = layer(x, memory, src_mask, tgt_mask)
#         # 
#         # print("Decoder output: ", x.shape) # torch.Size([9, 17, 512])
#         return self.norm(x)

class Decoder_Multimodal(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder_Multimodal, self).__init__()
        self.layers = clones(layer, N) # layer <- DecoderLayer
        self.norm = LayerNorm(layer.size)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, layer.size))
        
    def forward(self, 
        x, 
        proto_text,
        proto_visual,
        text2c_mask,
        visual2c_mask,
        multimodal_mask,
    ):
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = x.shape[0])
        # x = torch.cat((cls_tokens, x), dim=1)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x=x, 
                proto_text=proto_text,
                proto_visual=proto_visual,
                text2c_mask=text2c_mask,
                visual2c_mask=visual2c_mask,
                multimodal_mask=multimodal_mask
            )
        # print("* MM Decoder output: ", x.shape) # [12, 119, 512] <- torch.Size([9, 17, 512])
        return self.norm(x)


# In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. 
# Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization
"""
DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout)
                                 
attn = MultiHeadedAttention(h, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
"""

class DecoderLayer_Multimodal(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer_Multimodal, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        """
        self_attn和src_attn, 都是MultiHeadedAttention
        """
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4) # 跳接
 
    def forward(self, 
        x, 
        proto_text,
        proto_visual,
        text2c_mask,
        visual2c_mask,
        multimodal_mask,
    ):
        "Follow Figure 1 (right) for connections."
        # m = memory
        # print("* x:\t", x.shape)
        # print("* proto_text:\t", proto_text.shape)
        # print("* text2c_mask:\t", text2c_mask.shape)
        # print("---"*3)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, proto_text, proto_text, mask=text2c_mask)) # correct self-attention in decoder
        x = self.sublayer[1](x, lambda x: self.src_attn(x, proto_visual, proto_visual, mask=visual2c_mask)) # cross_attention, cross_mask
        x = self.sublayer[2](x, lambda x: self.src_attn(x, x, x, mask=multimodal_mask)) # cross_attention, cross_mask

        return self.sublayer[3](x, self.feed_forward)

# class DecoderLayer(nn.Module):
#     "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
#     def forward(self, x, memory, src_mask, tgt_mask):
#         "Follow Figure 1 (right) for connections."
#         m = memory
#         # print("decoder x:\t", x.shape) # [9, 17, 512]
#         # print("decoder m:\t", m.shape) # [9, 84, 512] with prototypes VOCAB_CONCEPT_NUM 40
#         # print("tgt_mask:\t", tgt_mask.shape) # [9, 17, 17]
#         # print("src_mask:\t", src_mask.shape) # [9, 1, 49]
#         #original one
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # correct self-attention in decoder
#         x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) # cross_attention, cross_mask

#         # # for other token_mixer: pooling
#         # x = self.sublayer[0](x, lambda x: self.self_attn(x))
#         # x = self.sublayer[1](x, lambda x: self.src_attn(x))


#         return self.sublayer[2](x, self.feed_forward)

# # We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. 
# # This masking, combined with fact that the output embeddings are offset by one position, 
# # ensures that the predictions for position i can depend only on the known outputs at positions less than i.
# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') #[1, 21, 21], 上三角矩阵 (不包含对角线)
    # print("* subsequent_mask:\n", subsequent_mask, subsequent_mask.shape) # (1, 21, 21)
    return torch.from_numpy(subsequent_mask) == 0 # 上三角变下三角

# ================================================ Decoder (End) ========================================================================= #


# ================================================ Multi-Head Self-Attention (MSA) (Start) ===================================================================== #
# core part of transformer model

def attention(query, key, value, mask=None, dropout=None): # Confrim shape of query, key, value, mask of this model and original transformer
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print('mask shape', mask.shape) # # mask shape torch.Size([5, 1, 1, 256])
        # print('scores', scores.shape) # # scores torch.Size([5, 8, 17, 3])

        # self-attention
        # mask shape torch.Size([5, 1, 17, 17])
        # scores torch.Size([5, 8, 17, 17])


        # Cross-attention in decoder
        # mask shape torch.Size([5, 1, 1, 256])  --> mask shape torch.Size([5, 1, 17, 3]) modify the mask
        # scores torch.Size([5, 8, 17, 3])  --> upsample it

        # RuntimeError: The size of tensor a (256) must match the size of tensor b (3) at non-singleton dimension 3

        # the example code to explain the mask: https://zhuanlan.zhihu.com/p/151783950
        mask = mask.to(scores.device) # cz add
        # print("* mask:\t", mask.shape) # [12, 1, 119, 40]
        # print("* score:\t", scores.shape) # [12, 8, 119, 40]
        scores = scores.masked_fill(mask == 0, float('-inf')) ################ mask is used here. We want to know in which positions, mask are 0 
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# How to call MultiHeadedAttention? Use self.self_attn(x, x, x, mask)
# we employ h=8 parallel attention layers, or heads.

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # 512 // 8 = 64
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # # src_mask in decoder torch.Size([5, 1, 256]) -->
            # print('mask in MultiHeadedAttention', mask.shape) # mask in MultiHeadedAttention torch.Size([5, 1, 1, 256]), mask in MultiHeadedAttention torch.Size([5, 1, 17, 17])
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # RuntimeError: shape '[3, -1, 8, 64]' is invalid for input of size 50176
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# ================================================ Multi-Head Self-Attention (MSA) (End) ===================================================================== #


# ===================================== Video based model ==================================== #
# Implementation is from https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_



from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x




def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Implementation is from https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        # print("* input Swin:\t", x.shape)
        """
        * input Swin:    torch.Size([12, 4, 56, 56, 128])
        * input Swin:    torch.Size([12, 4, 56, 56, 128])
        * input Swin:    torch.Size([12, 4, 28, 28, 256])
        * input Swin:    torch.Size([12, 4, 28, 28, 256])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 14, 14, 512])
        * input Swin:    torch.Size([12, 4, 7, 7, 1024])
        * input Swin:    torch.Size([12, 4, 7, 7, 1024])
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
# @lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x

# Here, the 3D patch is A cube 
class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size() # x0, input of the encoder torch.Size([1, 4, 3, 224, 224])
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:  # self.patch_size[0] is the about time axis, is sequence_length
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

class SwinTransformer3D_Encoder(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 d_model,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        print(' ==========    Check the version of the model ==========')
        print('embed_dim', embed_dim)
        print('depths', depths)
        print('num_heads', num_heads)

        self.d_model = d_model
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self.head = nn.Linear(self.num_features, self.d_model)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        print("* SwinTransformer3D_Encoder.inflate_weights => loaded successfully {}".format(self.pretrained)) # cz add
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            print("* SwinTransformer3D_Encoder.init_weights => load model from: {}".format(self.pretrained)) # cz add

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = rearrange(x, 'n d i h w -> n i d h w') # x0, input of the encoder torch.Size([3, 3, 4, 224, 224])
        # ====================================================================================== #
    
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        # print('x0 after 3D patch embedding', x.shape) # x0 after 3D patch embedding torch.Size([1, 96, 1, 56, 56])

        for layer in self.layers:
            x = layer(x.contiguous())
        
        # print('x1 inside swin3D model', x.shape) # x1 inside swin3D model torch.Size([1, 768, 1, 7, 7])

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        # print('x2 inside swin3D model', x.shape) # x2 inside swin3D model torch.Size([1, 1, 7, 7, 768]) # expect it will be converted into [1, 1*1*7, 768]
        # x2 inside swin3D model torch.Size([3, 1, 7, 7, 768]

        # x = rearrange(x, 'n d h w c -> n c d h w')
        # print('x3 inside swin3D model', x.shape) # x3 inside swin3D model torch.Size([1, 768, 1, 7, 7])
        
        x = rearrange(x, 'n d h w c -> n (d h w) c')
        # print('x4 output of encoder', x.shape) # x4 output of encoder torch.Size([1, 49, 768])
        x = self.head(x)
        # print('x5 output of encoder', x.shape) # x5 output of encoder torch.Size([1, 49, 512])
        # x5 output of encoder torch.Size([3, 49, 512])
        # =======================================================#


        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D_Encoder, self).train(mode)
        self._freeze_stages

# ============================================================================================================================== #

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Video_Swin_TranCAP_L_concept_cls(AttModel):

    def make_model(self, opt, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        self.opt = opt
        c = copy.deepcopy
        # original one
        attn = MultiHeadedAttention(h, d_model, dropout)

        # # replace attn with Pooling
        # attn = Pooling(pool_size=3)

        # # # replace attn with Swin Transformer Window based multi-head self attention
        # attn = WindowAttention(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        # # original Transformer model: def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        # model = EncoderDecoder(
        #     Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
        #     Decoder(DecoderLayer(d_model, c(attn), c(attn), 
        #                          c(ff), dropout), N_dec),
        #     lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        #     nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        #     Generator(d_model, tgt_vocab))


        # https://github.com/microsoft/Swin-Transformer/tree/3dc2a55301a02d71e956b382b50943a35c4abee9/configs
        # base configuration:EMBED_DIM: 128, DEPTHS: [ 2, 2, 18, 2 ], NUM_HEADS: [ 4, 8, 16, 32 ], WINDOW_SIZE: 7
        
        model_encoder = SwinTransformer3D_Encoder(
                   d_model=self.opt.d_model,
                   pretrained=None,
                   pretrained2d=True,
                   patch_size=(4,4,4),
                   in_chans=self.opt.in_channels,
                   embed_dim=self.opt.dim,
                   depths=[2, 2, 18, 2 ],
                   num_heads=[4, 8, 16, 32 ],
                   window_size=(2,7,7),
                   mlp_ratio=4.,
                   qkv_bias=True,
                   qk_scale=None,
                   drop_rate=0.,
                   attn_drop_rate=0.,
                   drop_path_rate=0.1,
                   norm_layer=nn.LayerNorm,
                   patch_norm=self.opt.patch_norm,
                   frozen_stages=-1,
                   use_checkpoint=False) # Default: patch_norm= False, patch_size=(4,4,4), window_size=(2,7,7)
        # model_decoder = Decoder(
        #     DecoderLayer(
        #         size= d_model, 
        #         self_attn=c(attn), 
        #         src_attn=c(attn), 
        #         feed_forward=c(ff), 
        #         dropout=dropout), 
        #     N_dec
        # )
        model_decoder = Decoder_Multimodal(
            DecoderLayer_Multimodal(
                size= d_model, 
                self_attn=c(attn), 
                src_attn=c(attn), 
                feed_forward=c(ff), 
                dropout=dropout), 
            N_dec
            ) # forward: x, memory, src_mask, tgt_mask -> x
        model_tgt_embed = nn.Sequential(
            Embeddings(
                d_model=d_model, 
                vocab=tgt_vocab), 
            c(position)) # foward: x -> x -> x
        model_generator = Generator(
            d_model=d_model, 
            vocab=tgt_vocab) # forward: x -> F.log_softmax(self.proj(x), dim=-1)
        
        ###########
        model_prototype_mapping = Encoder(
            EncoderLayer(
                size= d_model, 
                self_attn=c(attn), 
                feed_forward=c(ff), 
                dropout=dropout), 
            N_enc
            ) 
        model_prototype_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, VOCAB_CONCEPT_NUM)
        )

        model_text2visual_mapping = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 512)
        ) # 用于对齐text和visual两个空间
        
            


        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        print("model_encoder:\t", type(model_encoder))
        print("model_decoder:\t", type(model_decoder))
        print("model_tgt_embed:\t", type(model_tgt_embed))
        print("model_generator:\t", type(model_generator))
        print("model_prototype_mapping:\t", type(model_prototype_mapping))
        print("model_prototype_head:\t", type(model_prototype_head))
        print("model_text2visual_mapping:\t", type(model_text2visual_mapping))
        for model in [model_encoder, model_decoder, model_tgt_embed, model_generator,
                        model_prototype_mapping, model_prototype_head,
                        model_text2visual_mapping]:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        # return model
        return model_encoder, model_decoder, model_tgt_embed, model_generator, model_prototype_mapping, model_prototype_head, model_text2visual_mapping

    def __init__(self, opt):
        super(Video_Swin_TranCAP_L_concept_cls, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        # self.att_embed = nn.Sequential(*(
        #                             ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
        #                             (nn.Linear(self.att_feat_size, self.d_model),
        #                             nn.ReLU(),
        #                             nn.Dropout(self.drop_prob_lm))+
        #                             ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))


        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.att_feat_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1


        # self.model = self.make_model(0, tgt_vocab,
        #     N_enc=self.N_enc,
        #     N_dec=self.N_dec,
        #     d_model=self.d_model,
        #     d_ff=self.d_ff,
        #     h=self.h,
        #     dropout=self.dropout)

        self.model_encoder, self.model_decoder, self.model_tgt_embed, self.model_generator,\
            self.model_prototype_mapping, self.model_prototype_head, \
                self.model_text2visual_mapping = self.make_model(self.opt,
            0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout
        )

        self.text_cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        # self.text_classifier = nn.Linear(self.d_model, 35, bias=False)
        self.visual_classifier = nn.Linear(self.d_model, VOCAB_CONCEPT_NUM, bias=False)
        # add position embed for multimodal input
        self.multimodal_absolute_pos_embed = nn.Parameter(torch.zeros(1, TEXT_IN_NUM+VISUAL_OUT_NUM+VOCAB_CONCEPT_NUM, 512))
        trunc_normal_(self.multimodal_absolute_pos_embed, std=.02)
        
        self.multimodal_vision_layernorm = nn.LayerNorm(512)
        self.multimodal_text_layernorm = nn.LayerNorm(512)

    def logit(self, x): # unsafe way
        # return self.model.generator.proj(x)
        return self.model_generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        seq, seq_mask = self._prepare_feature_forward(att_feats) # Here, att_feats is (batch_size, 3, 224, 224). seq will be None as the default setting
        # memory = self.model.encode(att_feats)
        memory = self.model_encoder(att_feats) # 删去恒等映射，输入src
        
        # print('att_feats for beam search', att_feats.shape) # att_feats for beam search torch.Size([5, 3, 224, 224])
        # print('att_feats[...,:0]', att_feats[...,:0].shape) # torch.Size([5, 3, 224, 0])
        
        # att_feats[...,:0] convert (50, 196, 2048) (50, 196, 0)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, seq=None):
        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return seq, seq_mask


    def _forward(self, 
        prototypes_text, 
        prototypes_visual,
        fc_feats, # 无用
        att_feats, 
        seq, 
        att_masks=None # 无用，经过_prepare_feature_forward()更新
    ):
        """
        继承自 -> AttModel -> CaptionModel -> nn.Module
        model(**), 执行CaptionModel的forward(), 默认会调这个_forward
        若传入参数有'mode'则执行_'mode'。例如mode='sample', 执行AttModel的_sample(self, fc_feats, att_feats, att_masks=None, opt={})
        见于eval_utils的seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        """
        # self.model(/fc_feats, att_feats, labels[..., :-1], /att_masks), 

        # 废 fc_feats: torch.Size([9, 0]) <class 'torch.Tensor'>
        # 原始图像 att_feats: torch.Size([9, 3, 224, 224]) <class 'torch.Tensor'>
        # 标签int labels: torch.Size([9, 1, 18]) <class 'torch.Tensor'>
        # 好像表示标签seq长度+首尾 masks: torch.Size([9, 1, 18]) <class 'torch.Tensor'>
        # 废 att_masks: None
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        # att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(seq)
        seq, seq_mask = self._prepare_feature_forward(att_feats, seq)
        # print("* seq:\t", seq.shape) # [12, 31]
        # print("* seq_mask:\t", seq_mask.shape) # [12, 31, 31]

        # print("* att_feats:\t", att_feats.shape) # [12, 15, 3, 224, 224]
        out_encoder = self.model_encoder(att_feats) # out_encoder: [12, 196, 512] <- att_feats: [9, 3136, 128]
        # print("* out_encoder:\t", out_encoder.shape) # [12, 196, 512] <- [12, 98, 512]
        ######## prototype predict
        # predict_prototype = self.model_prototype_head(
        #     torch.mean(self.model_prototype_mapping(out_encoder, mask=None), dim=1, keepdim=False) # 视觉token整合起来，用于预测proto的多标签分类
        # ) # 用来对齐的visual feature来自model_encoder还是bypass head存疑，也许不干扰backbone再补充些额外的对齐token会比较好？
        ########

        # multimodal_mask = torch.ones(out_encoder.shape[0], TEXT_IN_NUM + VISUAL_OUT_NUM, TEXT_IN_NUM + VISUAL_OUT_NUM).to(out_encoder.device)
        multimodal_mask = torch.ones(out_encoder.shape[0], seq_mask.shape[1] + VISUAL_OUT_NUM, seq_mask.shape[2] + VISUAL_OUT_NUM).to(out_encoder.device)
        multimodal_mask[:, :, :seq_mask.shape[2]] = 0
        multimodal_mask[:, :seq_mask.shape[1], :seq_mask.shape[2]] = seq_mask # keep text temporal
        # print("* main multimodal_mask:\t", multimodal_mask.shape) # [12, 129, 129] # [12, 119, 119]
        #
        # text_cls_tokens = repeat(self.text_cls_token, '1 1 d -> b 1 d', b = out_encoder.shape[0])
        model_tgt_embed = self.model_tgt_embed(seq)
        # print("* model_tgt_embed:\t", model_tgt_embed.shape) # [12, 31, 512]
        # print("* model_tgt_embed:\t", torch.abs(model_tgt_embed).mean())
        # print("* out_encoder:\t", torch.abs(out_encoder).mean())
        # print("* prototypes:\t", torch.abs(prototypes).mean())
        multimodal_input = torch.cat(
            [
                # text_cls_tokens,
                # self.multimodal_text_layernorm(model_tgt_embed),
                # self.multimodal_vision_layernorm(out_encoder),
                model_tgt_embed,
                out_encoder,
            ],
            dim=1
        )
        # print("* multimodal_input:\t", multimodal_input.shape) # [12, 227, 512] # [12, 119, 512]
        # print("* self.multimodal_absolute_pos_embed:\t", self.multimodal_absolute_pos_embed.shape)
        # multimodal_input = multimodal_input + self.multimodal_absolute_pos_embed

        text2c_mask = torch.ones(out_encoder.shape[0], seq_mask.shape[1]+VISUAL_OUT_NUM, VOCAB_CONCEPT_NUM).to(out_encoder.device)
        visual2c_mask = torch.ones(out_encoder.shape[0], seq_mask.shape[1]+VISUAL_OUT_NUM, VOCAB_CONCEPT_NUM).to(out_encoder.device)
        # print("* text2c_mask:\t", text2c_mask.shape) # [12, 129, 42]
        # print("* visual2c_mask:\t", visual2c_mask.shape) # [12, 129, 42]

        # print("* prototypes_text.unsqueeze(0):\t", prototypes_text.unsqueeze(0).shape) # 
        # print("* prototypes_visual.unsqueeze(0):\t", prototypes_visual.unsqueeze(0).shape) # 

        out_decoder = self.model_decoder(
            x=multimodal_input, # # [12, 227, 512] for seq_15
            proto_text=repeat(prototypes_text.unsqueeze(0), '1 token d -> b token d', b = out_encoder.shape[0]),
            proto_visual=repeat(prototypes_visual.unsqueeze(0), '1 token d -> b token d', b = out_encoder.shape[0]),
            text2c_mask=text2c_mask, # [12, 129, 42]
            visual2c_mask=visual2c_mask, # [12, 129, 42]
            multimodal_mask=multimodal_mask, # [12, 129, 129], 129 shou be 227
        ) # multimodal [9, 102, 512]
        # print("* out_decoder:\t", out_decoder.shape) # 
        outputs = self.model_generator(out_decoder[:, :seq_mask.shape[1]]) # outputs: [9, 17, 40] <- out_decoder: [9, 17, 512]
        # print("* outputs:\t", outputs.shape) # 

        # out_decoder = self.model_decoder(
        #     x=self.model_tgt_embed(seq),
        #     # memory=out_encoder, 
        #     memory=torch.cat(
        #         [
        #             out_encoder, 
        #             repeat(prototypes.unsqueeze(0), '1 num_proto d -> bsz num_proto d', bsz=out_encoder.shape[0])
        #         ],
        #         dim=1
        #         ), 
        #     src_mask=att_masks,            
        #     tgt_mask=seq_mask
        # ) # [9, 17, 512]
        # # repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # # print("out_decoder:\t", out_decoder.shape) # [9, 17, 512]
         
        # outputs = self.model_generator(out_decoder)
        # return outputs
        # return outputs, (out_decoder, out_encoder, seq), predict_prototype, self.model_text2visual_mapping(out_decoder) 
        return outputs, (out_decoder[:, :seq_mask.shape[1]], out_encoder, seq), \
            self.visual_classifier(torch.mean(out_decoder[:, seq_mask.shape[1]:], dim=1, keepdim=False)), \
                self.model_text2visual_mapping(out_decoder[:, :seq_mask.shape[1]])



    def core(self, 
        prototypes_text, 
        prototypes_visual, 
        it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            # use this one, cz check
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        seq_mask = subsequent_mask(ys.size(1))
        # print("* core seq_mask:\t", seq_mask.shape) 

        multimodal_mask = torch.ones(memory.shape[0], seq_mask.shape[1]+VISUAL_OUT_NUM, seq_mask.shape[2]+VISUAL_OUT_NUM).to(memory.device)
        multimodal_mask[:, :, :seq_mask.shape[2]] = 0
        multimodal_mask[:, :seq_mask.shape[1], :seq_mask.shape[2]] = seq_mask # keep text temporal
        # print("* main multimodal_mask:\t", multimodal_mask.shape)
        #
        # text_cls_tokens = repeat(self.text_cls_token, '1 1 d -> b 1 d', b = memory.shape[0])
        model_tgt_embed = self.model_tgt_embed(ys)
        multimodal_input = torch.cat(
            [
                # text_cls_tokens,
                # self.multimodal_text_layernorm(model_tgt_embed),
                # self.multimodal_vision_layernorm(memory),
                model_tgt_embed,
                memory,
            ],
            dim=1
        )
        # multimodal_absolute_pos_embed = torch.cat(
        #     [   
        #         self.multimodal_absolute_pos_embed[:, :seq_mask.shape[1]],
        #         self.multimodal_absolute_pos_embed[:, 17:]
        #     ],
        #     dim=1
        # )
        # multimodal_input = multimodal_input + multimodal_absolute_pos_embed

        text2c_mask = torch.ones(memory.shape[0], seq_mask.shape[1]+VISUAL_OUT_NUM, VOCAB_CONCEPT_NUM).to(memory.device)
        visual2c_mask = torch.ones(memory.shape[0], seq_mask.shape[1]+VISUAL_OUT_NUM, VOCAB_CONCEPT_NUM).to(memory.device)

        
        out_decoder = self.model_decoder(
            x=multimodal_input,
            proto_text=repeat(prototypes_text.unsqueeze(0), '1 token d -> b token d', b = memory.shape[0]),
            proto_visual=repeat(prototypes_visual.unsqueeze(0), '1 token d -> b token d', b = memory.shape[0]),
            text2c_mask=text2c_mask,
            visual2c_mask=visual2c_mask,
            multimodal_mask=multimodal_mask,
        ) # multimodal [9, 102, 512]
        # print("* core out_decoder:\t", out_decoder.shape)

        # out = self.model_decoder(
        #     x=self.model_tgt_embed(ys), 
        #     memory=torch.cat([memory, repeat(prototypes.unsqueeze(0).to(memory.device), '1 num_proto d -> bsz num_proto d', bsz=memory.shape[0])], 
        #         dim=1), 
        #     src_mask=mask, 
        #     tgt_mask=subsequent_mask(ys.size(1))
        #     ).to(memory.device)
        return out_decoder[:, seq_mask.shape[1]-1], [ys.unsqueeze(0)]
    def get_logprobs_state(self, 
            # prototypes, # cz add
            prototypes_text, 
            prototypes_visual,
            it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        """
        cz add
        """
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(
            # prototypes=prototypes, 
            prototypes_text=prototypes_text, 
            prototypes_visual=prototypes_visual,
            it=xt, 
            fc_feats_ph=fc_feats, 
            att_feats_ph=att_feats, 
            memory=p_att_feats, 
            state=state, 
            mask=att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state
    
    def _sample(self, 
        # prototypes, # cz add
        prototypes_text, 
        prototypes_visual,
        fc_feats, att_feats, att_masks=None, opt={}):
        """
        cz add
        """
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            # 未使用
            print("need _sample_beam(), beam_size: {:d}, sample_method: {}".format(beam_size, sample_method))
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            # 未使用
            print("need _diverse_sample(), group_size: {:d}".format(group_size))
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size*sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        trigrams = [] # will be a list of batch_size dictionaries
        
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(
                # prototypes,
                prototypes_text, 
                prototypes_visual, 
                it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, output_logsoftmax=output_logsoftmax)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs


            
def init_prototypes(
    num_proto,
    dim=512,
    value_scale=0.1
):  
    # 确保完备
    prototype = value_scale * torch.rand(size=(num_proto, dim)).to('cuda')
    return prototype 


# ****************************************************************************************** #

if __name__ == "__main__":

    import os 
    import sys

    sys.path.append("/home/zhen_chen/code/surg_caption/SwinMLP_TranCAP-HK/captioning/data")
    from dataloader_video_vision_wales import DataLoader
    import captioning.utils.opts as opts
    
    opt = opts.parse_opt()
    
    
    
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.vocab = loader.get_vocab() ## vocab是什么？
    
    data = loader.get_batch('train')
    # data = loader.get_batch('val') 
    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
    tmp = [_ if _ is None else _.cuda() for _ in tmp]
    fc_feats, att_feats, labels, masks, att_masks = tmp

    prototypes_text = init_prototypes(num_proto=VOCAB_CONCEPT_NUM, dim=512, value_scale=0.1)
    prototypes_visual = init_prototypes(num_proto=VOCAB_CONCEPT_NUM, dim=512, value_scale=0.1)

    # model = models.setup(opt).cuda()
    model = Video_Swin_TranCAP_L_concept_cls(opt).cuda()
    
    model_output, (feature_decoder, feature_encoder, seq), visual_classifier_predict, feature_text2visual = model(
        prototypes_text=prototypes_text, 
        prototypes_visual=prototypes_visual,
        fc_feats=fc_feats, # 无用
        att_feats=att_feats, 
        seq=labels[..., :-1], 
        att_masks=None
    )

    print("* model_output:\t", model_output.shape)
    # print(model_output[0, 0])
    # print(model_output[0, 1])
    # print(model_output[0, -1])



    p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = model._prepare_feature(fc_feats, att_feats, att_masks)

    state = model.init_hidden(12) # batch_size*sample_n

    # trigrams = [] # will be a list of batch_size dictionaries
    
    print("model.seq_length:\t", model.seq_length)
    seq = fc_feats.new_full((12, model.seq_length), model.pad_idx, dtype=torch.long)
    seqLogprobs = fc_feats.new_zeros(12, model.seq_length, model.vocab_size + 1)
    for t in range(model.seq_length + 1):
        if t == 0: # input <bos>
            it = fc_feats.new_full([12], model.bos_idx, dtype=torch.long)

            xt = model.embed(it)

            core_output, state = model.core(
                    prototypes_text=prototypes_text, 
                    prototypes_visual=prototypes_visual,
                    it=xt, 
                    fc_feats_ph=fc_feats, 
                    att_feats_ph=att_feats, 
                    memory=p_att_feats, 
                    state=state, 
                    mask=att_masks
            )
            print("*t:\t", t)
            print("* core_output:\t", core_output.shape)
            print("* state:\t", state.shape)