import torch
import torch.nn as nn

class AMDT(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C

        out, _ = self.attn(x, x, x)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out