# 两阶段训练
# # python train.py --epoch 3000 --save_step 500
import torch, os, trimesh
from models.geo_encoder import FNOGeoEncoder
from models.physics_core import PhysicsCore
from models.losses import compute_loss
from pytorch3d.transforms import so3_exponential_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 读网格
upper = trimesh.load('data/upper.ply')
lower = trimesh.load('data/lower.ply')
V_up  = torch.tensor(upper.vertices, dtype=torch.float32, device=device)
F_up  = torch.tensor(upper.faces,    dtype=torch.long,   device=device)
V_low = torch.tensor(lower.vertices, dtype=torch.float32, device=device)

# 2. 初始化网络
enc_up  = FNOGeoEncoder().to(device)
enc_low = FNOGeoEncoder().to(device)
core    = PhysicsCore().to(device)
core.set_V_up(V_up)

opt = torch.optim.AdamW(list(enc_up.parameters())+list(enc_low.parameters())+list(core.parameters()), lr=1e-3, weight_decay=1e-4)

# 3. 训练
for step in range(3001):
    # if (step+1) % 10 == 0:
    #     print("### step: ", step + 1)
    #     torch.cuda.empty_cache()


    opt.zero_grad()
    # 几何编码
    z_up  = enc_up(V_up)
    z_low = enc_low(V_low)
    # 采样接触点
    collo = V_up[torch.randperm(V_up.shape[0])[:8000]]
    theta, t = core(collo, z_up, z_low)
    # 变形下颌
    R = so3_exponential_map(theta.unsqueeze(0)).squeeze(0)
    mandible_v = V_low @ R.T + t
    # 损失
    loss, logs = compute_loss(V_up, F_up, V_low, mandible_v, theta, t,
                              hinge_dir_L=torch.tensor([0.,0.,1.], device=device),
                              hinge_dir_R=torch.tensor([0.,0.,1.], device=device))
    loss.backward()
    opt.step()
    if step % 10==0:
        print(f'step {step}: loss={loss.item():.4f}', logs)
        # 导出
        T = torch.eye(4, device=device)
        T[:3,:3] = R; T[:3,3] = t
        final_v = (torch.cat([V_low, torch.ones((V_low.shape[0],1), device=device)], dim=1) @ T.T)[:,:3]
        trimesh.Trimesh(vertices=final_v.detach().cpu().numpy(), faces=lower.faces).export(f'data/lower_bite_{step}.ply')