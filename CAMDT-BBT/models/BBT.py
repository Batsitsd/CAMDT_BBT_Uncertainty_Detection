import torch
import torch.nn as nn

class BBT(nn.Module):
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C

        q = self.dropout(self.q(x))
        k = self.dropout(self.k(x))
        v = self.dropout(self.v(x))

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out