import torch
import torch.nn as nn

from timm.models.layers import PatchEmbed, Mlp, DropPath
from timm.models.registry import register_model

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class EEGTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 4000
        self.decoder_depth = 1
        self.blocks = nn.Sequential(*[
            Block(
                dim= self.dim,
                drop= 0.2,
                attn_drop= 0.2,
            )
            for i in range(self.decoder_depth)])
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim)) # 1, 1, 8000

        # self.fc1 = nn.Linear(60, 512)
        self.fc1 = nn.Linear(self.dim, 512)

        self.fc2 = nn.Linear(512, 3)

        
        torch.nn.init.normal_(self.cls_token, std=.02)
        
    def forward(self, x):
        # x -> bz x 59 x 8000
        # print(x.shape)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1) # bz x 1 x 8000
        x = torch.cat((x, cls_token), 1) # bz x 60 x 8000
        x = self.blocks(x)

        # cls_token = x[:, -1, :].squeeze(1).unsqueeze(-1) # bz x 8000 x 1
        # attn_map = torch.bmm(x, cls_token).squeeze(-1) # bz x 60
        # attn_map = self.relu(self.fc1(attn_map))
        # attn_map = self.fc2(attn_map)
        # return attn_map

        cls_token = x[:, -1, :].squeeze(1)
        cls_token = self.relu(self.fc1(cls_token))
        cls_token = self.fc2(cls_token)
        return cls_token
    
@register_model
def eegt(**kwargs):
    model = EEGTransformer()
    return model

