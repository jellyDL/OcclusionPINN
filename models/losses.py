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
                 lambda_pen=30.0, lambda_cont=8.0, lambda_axis=0.3, lambda_orth=0.05):
    # 1. 穿透损失 (严格约束穿透)
    sdf_up = mesh_sdf_pytorch3d(mandible_v, V_up, F_up)          # (N_low,)
    pen = torch.abs(sdf_up).mean()

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

    print("#### pen:{} cont:{} axis:{} ".format(pen, cont, axis))

    loss = lambda_pen*pen + lambda_cont*cont + lambda_axis*axis
    return loss, {'pen':pen, 'cont':cont, 'axis':axis}

def sample_contact_points(V_up, V_low, delta=0.01, max_pts=15000, chunk=3000):
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

def compute_pde_loss(x, u_pred, pinn_net, E=1e4, nu=0.3, rho=1000.0):
    """
    计算弹性力学PDE损失 (线性弹性方程)
    Args:
        x: 配点坐标
        u_pred: 预测位移
        pinn_net: PINN网络
        E: 弹性模量
        nu: 泊松比
        rho: 密度
    """
    device = x.device

    # 计算拉梅常数
    mu = E / (2 * (1 + nu))
    lambda_lame = E * nu / ((1 + nu) * (1 - 2 * nu))

    # 计算位移梯度
    u, du_dx = pinn_net.compute_derivatives(x)

    # 计算二阶导数 (拉普拉斯算子)
    laplacian_u = torch.zeros_like(u)
    for i in range(3):  # 对每个位移分量
        grad_u_i = torch.autograd.grad(
            outputs=u[:, i:i+1], inputs=x,
            grad_outputs=torch.ones_like(u[:, i:i+1]),
            create_graph=True, retain_graph=True
        )[0]

        for j in range(3):  # 对每个坐标分量
            laplacian_u[:, i] += torch.autograd.grad(
                outputs=grad_u_i[:, j:j+1], inputs=x,
                grad_outputs=torch.ones_like(grad_u_i[:, j:j+1]),
                create_graph=True, retain_graph=True
            )[0][:, j]

    # 计算散度
    div_u = du_dx[:, 0] + du_dx[:, 1] + du_dx[:, 2]  # ∇·u

    # 线性弹性方程: μ∇²u + (λ+μ)∇(∇·u) = 0 (准静态)
    pde_residual = mu * laplacian_u + (lambda_lame + mu) * torch.autograd.grad(
        outputs=div_u, inputs=x,
        grad_outputs=torch.ones_like(div_u),
        create_graph=True, retain_graph=True
    )[0]

    pde_loss = torch.mean(pde_residual**2)
    return pde_loss

def compute_boundary_loss(V_up, V_low, pinn_net, theta, t):
    """
    计算边界条件损失 - 优化内存使用
    Args:
        V_up: 上颌顶点
        V_low: 下颌顶点
        pinn_net: PINN网络
        theta: 旋转参数
        t: 平移参数
    """
    # 内存友好的接触检测：分批处理
    contact_threshold = 0.5
    batch_size = 100
    contact_points = []

    # 分批计算距离，避免内存溢出
    for i in range(0, len(V_up), batch_size):
        batch_up = V_up[i:i+batch_size]

        # 对每个批次找最近的下颌点
        batch_distances = []
        for j in range(0, len(V_low), batch_size):
            batch_low = V_low[j:j+batch_size]
            dist_batch = torch.cdist(batch_up, batch_low)
            batch_distances.append(torch.min(dist_batch, dim=1)[0])

        # 找到整个下颌的最小距离
        min_distances = torch.min(torch.stack(batch_distances), dim=0)[0]

        # 选择接触点
        contact_mask = min_distances < contact_threshold
        if contact_mask.any():
            contact_points.append(batch_up[contact_mask])

        all_contact_points = torch.cat(contact_points, dim=0)

        # 限制接触点数量以节省内存
        if len(all_contact_points) > 2000:
            indices = torch.randperm(len(all_contact_points))[:2000]
            all_contact_points = all_contact_points[indices]

        # 预测接触点的位移
        u_contact = pinn_net(all_contact_points)

        # 边界条件：接触点的位移应该很小
        bc_loss = torch.mean(u_contact**2)
    else:
        bc_loss = torch.tensor(0.0, device=V_up.device)

    return bc_loss