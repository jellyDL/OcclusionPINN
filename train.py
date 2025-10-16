# 两阶段训练
# # python train.py --epoch 3000 --save_step 500
import torch, os, trimesh
import numpy as np
from models.geo_encoder import FNOGeoEncoder
from models.physics_core import PhysicsCore
from models.pinn_net import PINNNet
from models.losses import compute_loss, compute_pde_loss, compute_boundary_loss
from pytorch3d.transforms import so3_exponential_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 读网格
upper = trimesh.load('data/upper.ply')
lower = trimesh.load('data/lower.ply')
V_up  = torch.tensor(upper.vertices, dtype=torch.float32, device=device)
F_up  = torch.tensor(upper.faces,    dtype=torch.long,   device=device)
V_low = torch.tensor(lower.vertices, dtype=torch.float32, device=device)

# 2. 初始化网络 (添加PINN网络)
enc_up  = FNOGeoEncoder().to(device)
enc_low = FNOGeoEncoder().to(device)
core    = PhysicsCore().to(device)
pinn_net = PINNNet(input_dim=3, hidden_dim=256, output_dim=3).to(device)  # 新增PINN网络
core.set_V_up(V_up)

# 为PINN网络设置更高学习率(5e-3)，其他网络保持1e-3
params = [
    {'params': enc_up.parameters(), 'lr': 1e-3},
    {'params': enc_low.parameters(), 'lr': 1e-3},
    {'params': core.parameters(), 'lr': 1e-3},
    {'params': pinn_net.parameters(), 'lr': 5e-3}
]
opt = torch.optim.AdamW(params, weight_decay=1e-4)

# PINN训练参数
# lambda_pde = 0.05      # 降低PDE损失权重
# lambda_bc = 20.0      # 边界条件损失权重
# lambda_geo = 100.0     # 几何损失权重

lambda_pde = 0.0001      # 降低PDE损失权重
lambda_bc = 0.0001      # 边界条件损失权重
lambda_geo = 1.0     # 几何损失权重

# 3. PINN训练
for step in range(5001):
    opt.zero_grad()

    # 几何编码
    z_up  = enc_up(V_up)
    z_low = enc_low(V_low)

    # 采样配点（用于PDE约束）
    collo = V_up[torch.randperm(V_up.shape[0])[:15000]]
    collo.requires_grad_(True)

    # 物理核心预测
    theta, t = core(collo, z_up, z_low)

    # PINN预测位移场 - 分别为配点和下颌顶点计算
    with torch.cuda.amp.autocast():
        u_pred_collo = pinn_net(collo)  # 配点位移，用于PDE损失
        u_pred_low = pinn_net(V_low)    # 下颌顶点位移，用于变形

    # 变形下颌
    R = so3_exponential_map(theta.unsqueeze(0)).squeeze(0)
    mandible_v = V_low @ R.T + t + u_pred_low  # 使用正确维度的位移

    # 计算各类损失
    # 1. 几何损失
    geo_loss, logs = compute_loss(V_up, F_up, V_low, mandible_v, theta, t,
                                  hinge_dir_L=torch.tensor([0.,0.,1.], device=device),
                                  hinge_dir_R=torch.tensor([0.,0.,1.], device=device))

    # 2. PDE损失 (弹性力学方程) - 使用配点位移
    pde_loss = compute_pde_loss(collo, u_pred_collo, pinn_net)

    # 3. 边界条件损失
    bc_loss = compute_boundary_loss(V_up, V_low, pinn_net, theta, t)

    print("geo_loss:{} pde_loss:{} bc_loss:{}".format(geo_loss.item(), pde_loss.item(), bc_loss.item()))
    # 总损失
    total_loss = lambda_geo * geo_loss + lambda_pde * pde_loss + lambda_bc * bc_loss

    total_loss.backward()
    opt.step()

    # 每10次迭代衰减pinn_net学习率(第4个参数组)
    if step > 0 and step % 10 == 0:
        opt.param_groups[3]['lr'] *= 0.8

    # if step % 10 == 0:
    if 1:
        print(f'step {step}: total_loss={total_loss.item():.4f}, '
              f'geo_loss={geo_loss.item():.4f}, '
              f'pde_loss={pde_loss.item():.4f}, '
              f'bc_loss={bc_loss.item():.4f}', logs)

        # 导出
        T = torch.eye(4, device=device)
        T[:3,:3] = R; T[:3,3] = t
        u_final = pinn_net(V_low)  # 获取最终位移
        final_v = ((torch.cat([V_low + u_final, torch.ones((V_low.shape[0],1), device=device)], dim=1) @ T.T))[:,:3]
        trimesh.Trimesh(vertices=final_v.detach().cpu().numpy(), faces=lower.faces).export(f'data/lower_bite_{step}.ply')