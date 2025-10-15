import torch
import torch.nn as nn
import numpy as np

class PINNNet(nn.Module):
    """物理信息神经网络 - 用于预测位移场"""

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3, num_layers=6):
        super(PINNNet, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入坐标 (N, 3)
        Returns:
            u: 位移场 (N, 3)
        """
        return self.net(x)

    def compute_derivatives(self, x):
        """计算位移场的导数，用于PDE约束"""
        x.requires_grad_(True)
        u = self.forward(x)

        # 计算应变张量
        grad_outputs = torch.ones_like(u)
        gradients = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]

        return u, gradients