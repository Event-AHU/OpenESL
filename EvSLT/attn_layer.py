import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Cross-Attention:
        - Queries come from dim_q (e.g., tokens)
        - Keys/Values come from dim_kv (e.g., context/features)
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, Nq, Cq = x_q.shape
        Nk = x_kv.shape[1]

        q = self.q_proj(x_q).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)  # B, H, Nq, Dh
        k = self.k_proj(x_kv).reshape(B, Nk, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)  # B, H, Nk, Dh
        v = self.v_proj(x_kv).reshape(B, Nk, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)  # B, H, Nk, Dh

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, Nq, Nk

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


        
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, sp_feat, g_feat, tp_feat, mask=None):
        g_t_feat = torch.cat((g_feat, tp_feat), dim=1)
        g_t_feat = g_t_feat + self.drop_path(self.attn(self.norm1(g_t_feat), mask))
        
        self_graph_feat = g_t_feat[:,:g_feat.shape[1]]
        self_temporal_feat = g_t_feat[:,g_feat.shape[1]:]
        
        cross_graph_feat = g_feat + self.drop_path(self.cross_attn(self.norm2(g_feat), self.norm2(sp_feat)))
        
        graph_feat = self_graph_feat * cross_graph_feat
        
        x = torch.cat((self_temporal_feat, graph_feat), dim=1)
        
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x


def init_weights(module):
    """Initialize network parameters like timm/ViT style."""
    if isinstance(module, nn.Linear):
        # truncated normal for weights, zero bias
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
            
class GSF_model(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        self.layers = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                drop_path=dpr[i],
                attn_drop=attn_dropout,
            ) for i in range(depth)
        ])

        self.layers.apply(init_weights)

    def forward(self, sp_feat: torch.Tensor, g_feat: torch.Tensor, tp_feat: torch.Tensor):
        """
        q: (B, Nq, C)
        context: (B, Nc, C)
        kwargs forwarded to block (masks)
        """
        for layer in self.layers:
            q = layer(sp_feat, g_feat, tp_feat)
        return q