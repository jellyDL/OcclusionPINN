# FNO 几何编码器

import torch
import torch.nn as nn
from torch.fft import rfft, irfft

class FNOGeoEncoder(nn.Module):
    """
    输入: V_up (N,3), V_low (N,3)  已按弧长重排序
    输出: z_up, z_low  (256,)
    """
    def __init__(self, modes=32, width=64, out_channels=256):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(3, width)
        self.conv = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(4)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, V):               # V: (N,3)
        x = self.fc0(V).unsqueeze(0)    # (1,N,width)
        x = x.permute(0,2,1)            # (1,width,N)
        for layer in self.conv:
            x = layer(x)
        x = x.mean(dim=-1)              # (1,width)
        z = self.fc2(torch.tanh(self.fc1(x.squeeze(0))))
        return z

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (b,c,n)
        b, c, n = x.shape
        x_ft = rfft(x, n=self.modes, dim=-1)
        out_ft = torch.zeros(b, self.weights.shape[1], x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bcm,com->bom", x_ft[:, :, :self.modes], self.weights)
        x = irfft(out_ft, n=n, dim=-1)
        return x