import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from Dataset import load_data

class PosEmbedding(nn.Module):
    def __init__(self, in_dims : int = 32, in_size : int = 100, emb_size : int = 32):
        self.in_size = in_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dims, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn(in_size+1, emb_size))
        
    def forward(self, x:Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 32, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 
    
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
    
class EncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 32,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            Residual(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            Residual(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                ),
                nn.Dropout(drop_p)
            ))
        )
        
class Encoder(nn.Sequential):
    def __init__(self,depth: int=12, **kwargs):
        super().__init__(*[EncoderBlock(**kwargs) for _ in range(depth)])
        
class NLoc(nn.Module):
    def __init__(self,
                 in_dims: int = 32,
                 emb_size: int = 32,
                 in_size: int = 100,
                 depth: int = 12,
                 n_heads: int = 8,
                 **kwargs):
        super().__init__()
        self.embed = PosEmbedding(in_dims, in_size, emb_size)
        self.encode = Encoder(depth, emb_size=emb_size, num_heads=n_heads, **kwargs)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.encode(x)
        return x[:, 0, :]
        

if __name__ == '__main__':
    summary(NLoc(), (100,32), device='cpu')
    # data_loader = load_data("datasets/Aachen-Day-Night/images/images_upright",
    #                             "pairs/aachen/pairs-query-netvlad50.txt",
    #                             num_workers=0,
    #                             batch_size=1,
    #                             triplets=True)
    # for t1,t2,t3 in iter(data_loader):
    #     print(t1.shape)
    #     o1 = PosEmbedding()(t1)
    #     print(o1.shape)
    #     o2 = EncoderBlock(num_heads=8)(t1)
    #     print(o2.shape)
    #     o3 = NLoc()(t1)
    #     print(o3.shape)
    #     break
    
    data_loader = load_data("datasets/Aachen-Day-Night/images/images_upright",
                                "pairs/aachen/pairs-query-netvlad50.txt",
                                num_workers=12,
                                batch_size=32)
    for des, lbl in iter(data_loader):
        print(des.shape)
        print(lbl.shape)
        break