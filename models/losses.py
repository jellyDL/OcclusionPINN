# 4 项守恒损失
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exponential_map

def compute_loss(V_up, F_up, V_low, mandible_v, theta, t, hinge_dir_L, hinge_dir_R,
                 lambda_pen=1.0, lambda_cont=0.3, lambda_axis=0.1, lambda_orth=0.05):
    # 1. 穿透损失
    sdf_up = kaolin_mesh_sdf(mandible_v, V_up, F_up)          # (N_low,)
    pen = F.relu(-sdf_up).mean()

    # 2. 接触均匀
    contact = sample_contact_points(V_up, mandible_v, delta=0.05)  # (K,3)
    if contact.shape[0] > 1:
        mu = contact.mean(0)
        cont = torch.var(contact - mu, unbiased=False).sum()
    else:
        cont = torch.tensor(0., device=V_up.device)

    # 3. 铰链轴守恒
    B_L = V_up[(V_up[:,0]<-40) & (V_up[:,1]>20)]   # 简易髁突区域
    B_R = V_up[(V_up[:,0]> 40) & (V_up[:,1]>20)]
    R = so3_exponential_map(theta.unsqueeze(0)).squeeze(0)
    T = torch.eye(4, device=V_up.device)
    T[:3,:3] = R; T[:3,3] = t
    def axis_err(B, e):
        if B.shape[0]==0: return torch.tensor(0., device=V_up.device)
        mu = B.mean(0)
        mu_T = (T @ torch.cat([mu, torch.ones(1, device=mu.device)]) )[:3]
        d = (mu_T - mu) / (torch.norm(mu_T - mu) + 1e-8)
        return torch.norm(torch.cross(d, e))**2
    axis = axis_err(B_L, hinge_dir_L) + axis_err(B_R, hinge_dir_R)

    # 4. 隐空间正交
    # z_up, z_low 由调用方传入
    # cos = F.cosine_similarity(z_up, z_low, dim=0); orth = cos**2

    loss = lambda_pen*pen + lambda_cont*cont + lambda_axis*axis
    return loss, {'pen':pen, 'cont':cont, 'axis':axis}

# 工具：kaolin 快速 SDF
@torch.jit.script
def kaolin_mesh_sdf(pts, verts, faces):
    # pts: (N,3)  返回 (N,)  带符号距离
    from kaolin.metrics.triangle import point_to_mesh_distance
    dist, _, _ = point_to_mesh_distance(pts.unsqueeze(0), verts.unsqueeze(0), faces.long())
    return dist.squeeze(0)

def sample_contact_points(V_up, V_low, delta=0.05):
    # 简单采样：上颌表面点中与下颌距离<delta 者
    d = torch.cdist(V_up, V_low).min(dim=1)[0]
    return V_up[d < delta]