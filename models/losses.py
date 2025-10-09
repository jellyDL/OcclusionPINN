# 4 项守恒损失
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance

def mesh_sdf_pytorch3d(pts, verts, faces):
    """
    近似 SDF：点到上颌网格的最近距离（正值在外，负值在内）
    pts: (N,3)
    verts/faces: 网格
    返回 (N,) 距离
    """
    mesh  = Meshes(verts=[verts], faces=[faces.long()])
    pcls  = Pointclouds(points=[pts])
    # dist 返回 (N,)  正值=外部距离，负值=内部穿透
    sq_dist = point_mesh_face_distance(mesh, pcls)
    return sq_dist.sqrt()


def compute_loss(V_up, F_up, V_low, mandible_v, theta, t, hinge_dir_L, hinge_dir_R,
                 lambda_pen=1.0, lambda_cont=0.3, lambda_axis=0.1, lambda_orth=0.05):
    # 1. 穿透损失
    sdf_up = mesh_sdf_pytorch3d(mandible_v, V_up, F_up)          # (N_low,)
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

def sample_contact_points(V_up, V_low, delta=0.05, max_pts=10000, chunk=3000):
    """
    降采样 + 分块 cdist，内存 O(chunk×M)
    返回 (K,3)  K≤max_pts
    """
    # 1. 先随机降采样到 max_pts
    if V_up.shape[0] > max_pts:
        idx_up = torch.randperm(V_up.shape[0], device=V_up.device)[:max_pts]
        V_up_s = V_up[idx_up]
    else:
        V_up_s = V_up

    # 2. 分块求最近距离
    n_low = V_low.shape[0]
    dist = torch.empty(V_up_s.shape[0], device=V_up.device)
    for i in range(0, V_up_s.shape[0], chunk):
        end = min(i + chunk, V_up_s.shape[0])
        dist[i:end] = torch.cdist(V_up_s[i:end], V_low).min(dim=1)[0]

    # 3. 筛选接触点
    mask = dist < delta
    contact = V_up_s[mask]
    return contact