# SIREN + 积分池化
import torch
import torch.nn as nn
from pytorch3d.transforms import so3_exponential_map

class SIREN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, omega=30.):
        super().__init__()
        self.net = []
        self.net.append(nn.Linear(in_dim, hidden))
        self.net.append(nn.SiLU())
        for i in range(3):
            self.net.append(nn.Linear(hidden, hidden))
            self.net.append(nn.SiLU())
        self.net.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*self.net)
        self.omega = omega
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -torch.sqrt(6./m.in_features)/self.omega,
                                     torch.sqrt(6./m.in_features)/self.omega)
    def forward(self, x):
        return self.net(self.omega * x)

class PhysicsCore(nn.Module):
    def __init__(self, z_dim=256, hidden=128):
        super().__init__()
        self.siren = SIREN(3 + z_dim, hidden, 6)   # 输出局部 6DoF
        self.z_dim = z_dim

    def forward(self, x_c, z_up, z_low):
        """
        x_c: (M,3)  采样点
        z_up, z_low: (256,)
        return: 全局 theta(3), t(3)
        """
        M = x_c.shape[0]
        z_up  = z_up.unsqueeze(0).expand(M, -1)
        z_low = z_low.unsqueeze(0).expand(M, -1)
        feat = torch.cat([x_c, z_up + z_low], dim=-1)  # 简单融合
        local_6dof = self.siren(feat)                  # (M,6)
        w = torch.exp(-self.sdf_up(x_c) / 0.5)         # 权重：靠近上颌接触面
        theta = torch.sum(w.unsqueeze(-1) * local_6dof[:, :3], dim=0) / (w.sum() + 1e-8)
        t     = torch.sum(w.unsqueeze(-1) * local_6dof[:, 3:], dim=0) / (w.sum() + 1e-8)
        return theta, t

    @torch.no_grad()
    def sdf_up(self, x):
        # 简易 SDF：到上颌顶点的最近距离
        return torch.cdist(x.unsqueeze(0), self.V_up.unsqueeze(0)).squeeze(0).min(dim=1)[0]
    def set_V_up(self, V_up):
        self.register_buffer('V_up', V_up)