# 4 项守恒损失
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.ops import sample_points_from_meshes, knn_points  # 新增
import numpy as np
import open3d as o3d

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

def approx_signed_distance_to_mesh(pts, verts, faces, nsamples=16000):
    """
    基于网格表面采样 + 法向的带符号距离近似：
    - 幅值：最近采样点欧氏距离
    - 符号：与最近采样点法向的点乘符号（法向外指）
    """
    device = pts.device
    mesh = Meshes(verts=[verts], faces=[faces.long()])

    nsamples = int(nsamples)
    # 采样表面点与法向 (1, S, 3)
    surf_pts, surf_normals = sample_points_from_meshes(mesh, nsamples, return_normals=True)
    surf_pts = surf_pts[0]                      # (S, 3)
    surf_normals = F.normalize(surf_normals[0], dim=-1)  # (S, 3)

    # KNN 查找每个查询点最近采样点
    p = pts.unsqueeze(0)                        # (1, N, 3)
    d2, idx, nn = knn_points(p, surf_pts.unsqueeze(0), K=1, return_nn=True)
    nn_pts = nn[0, :, 0, :]                     # (N, 3)
    nn_normals = surf_normals[idx[0, :, 0]]     # (N, 3)

    # 用法向点乘决定符号（外法向：正=外部，负=内部）
    vec = pts - nn_pts                          # (N, 3)
    signed_dir = (vec * nn_normals).sum(dim=-1) # (N,)
    dist = torch.sqrt(d2[0, :, 0] + 1e-12)      # (N,)
    sdf = dist * torch.sign(signed_dir + 1e-12) # (N,)
    return sdf

def mirror_points_across_plane(points: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    # 将点关于平面 (n, c) 镜像：p' = p - 2 * ((p-c)·n) n
    v = points - c
    d = v @ n
    return points - 2.0 * d[:, None] * n[None, :]

def symmetry_error(points: np.ndarray, n: np.ndarray, c: np.ndarray, kdtree: o3d.geometry.KDTreeFlann) -> float:
    """最近邻RMS距离作为对称误差（复用KDTree以便多次评估）"""
    mirrored = mirror_points_across_plane(points, n, c)
    dists2 = []
    for q in mirrored:
        _, _, dist2 = kdtree.search_knn_vector_3d(q, 1)
        dists2.append(dist2[0])
    return float(np.sqrt(np.mean(dists2)))


def optimize_offset_along_normal(points: np.ndarray, n: np.ndarray, c0: np.ndarray, kdtree: o3d.geometry.KDTreeFlann):
    """沿法向对平面位置做1D黄金分割搜索，最小化对称误差"""
    n = n / (np.linalg.norm(n) + 1e-12)
    proj = points @ n
    tmin = float(proj.min() - c0 @ n)
    tmax = float(proj.max() - c0 @ n)
    # 黄金分割
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = tmin, tmax
    c1 = b - phi * (b - a)
    c2 = a + phi * (b - a)
    f1 = symmetry_error(points, n, c0 + c1 * n, kdtree)
    f2 = symmetry_error(points, n, c0 + c2 * n, kdtree)
    for _ in range(40):
        if f1 > f2:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + phi * (b - a)
            f2 = symmetry_error(points, n, c0 + c2 * n, kdtree)
        else:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - phi * (b - a)
            f1 = symmetry_error(points, n, c0 + c1 * n, kdtree)
    t_best = 0.5 * (a + b)
    c_best = c0 + t_best * n
    err_best = symmetry_error(points, n, c_best, kdtree)
    return c_best, err_best

# 新增：在初始法向附近做小角度扰动，联动优化偏移，选最优
def refine_normal(points: np.ndarray, n0: np.ndarray, c0: np.ndarray, kdtree: o3d.geometry.KDTreeFlann):
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    # 选取与 n0 不共线的一条轴作为切向
    world = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(world, n0)) > 0.9:
        world = np.array([0.0, 1.0, 0.0], dtype=float)
    u = world - n0 * np.dot(world, n0)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n0, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    def rotate_vec(n, axis, ang):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        c, s = np.cos(ang), np.sin(ang)
        return n * c + np.cross(axis, n) * s + axis * np.dot(axis, n) * (1.0 - c)

    angles = np.deg2rad(np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float))
    best_n, best_c, best_err = n0, c0, symmetry_error(points, n0, c0, kdtree)
    for ax in (u, v):
        for a in angles:
            n_cand = rotate_vec(n0, ax, a)
            n_cand = n_cand / (np.linalg.norm(n_cand) + 1e-12)
            c_opt, err = optimize_offset_along_normal(points, n_cand, c0, kdtree)
            if err < best_err:
                best_n, best_c, best_err = n_cand, c_opt, err
    return best_n, best_c, best_err


# 新增：在初始法向附近做小角度扰动，联动优化偏移，选最优
def refine_normal_opt(points: np.ndarray, n0: np.ndarray, c0: np.ndarray, kdtree: o3d.geometry.KDTreeFlann):
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    # 选取与 n0 不共线的一条轴作为切向
    world = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(world, n0)) > 0.9:
        world = np.array([0.0, 1.0, 0.0], dtype=float)
    u = world - n0 * np.dot(world, n0)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n0, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    def rotate_vec(n, axis, ang):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        c, s = np.cos(ang), np.sin(ang)
        return n * c + np.cross(axis, n) * s + axis * np.dot(axis, n) * (1.0 - c)

    angles = np.deg2rad(np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float))
    best_n, best_c, best_err = n0, c0, symmetry_error(points, n0, c0, kdtree)
    for ax in (u, v):
        for a in angles:
            n_cand = rotate_vec(n0, ax, a)
            n_cand = n_cand / (np.linalg.norm(n_cand) + 1e-12)
            c_opt, err = optimize_offset_along_normal(points, n_cand, c0, kdtree)
            if err < best_err:
                best_n, best_c, best_err = n_cand, c_opt, err
    return best_n, best_c, best_err

def estimate_mid_plane(points: np.ndarray):  # 估计单颌中切平面
    """估计切分单颌左右的中切平面：先选主轴，再优化偏移与小角度法向"""
    # 兼容 torch.Tensor（含 GPU），统一转为 CPU numpy
    if isinstance(points, torch.Tensor):
        pts_np = points.detach().cpu().numpy()
    else:
        pts_np = np.asarray(points)

    center = pts_np.mean(axis=0)
    pts0 = pts_np - center
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 列为特征向量（升序）
    candidates = [eigvecs[:, i] / (np.linalg.norm(eigvecs[:, i]) + 1e-12) for i in range(3)]

    # 预构建KDTree（稳定写法）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 先在三个主轴上各自优化偏移，挑最优
    best_n, best_c, best_err = None, None, 1e18
    for n in candidates:
        c_opt, err = optimize_offset_along_normal(pts_np, n, center, kdtree)
        if err < best_err:
            best_n, best_c, best_err = n, c_opt, err

    # 在最优法向附近做小角度离散搜索并联动优化偏移
    best_n, best_c, best_err = refine_normal(pts_np, best_n, best_c, kdtree)

    # 方向统一（可选）
    if best_n[0] < 0:
        best_n = -best_n
    return best_n, best_c, best_err

def compute_loss(V_up, F_up, V_low, mandible_v, theta, t, hinge_dir_L, hinge_dir_R,
                 lambda_pen=10.0, lambda_cont=0.1, lambda_axis=0.3, lambda_orth=0.05):
    # 1. 穿透损失 (严格约束穿透：只惩罚负SDF)
    # sdf_up < 0 表示下颌点位于上颌网格内
    sdf_up = approx_signed_distance_to_mesh(
        mandible_v, V_up, F_up,
        nsamples=min(20000, max(10000, V_up.shape[0] * 4))
    )          # (N_low,)
    neg_depth = F.relu(-sdf_up)                 # 仅穿透深度
    pen = (neg_depth * neg_depth).mean()        # 深穿透二次惩罚

    # 2. 接触均匀（左右分区分别均匀，避免偏侧）
    contact = sample_contact_points(V_up, mandible_v, delta=0.1)  # (K,3)
    
    # 估计中切平面并分左右
    n, c, err = estimate_mid_plane(V_up)
    print(f"  中切面 点 c = {c.tolist()}")
    print(f"  中切面 法向 n = {n.tolist()}")
    print(f"  中切面 对称误差 = {err:.4f} (均方根最近邻距离)")
    
    if contact.shape[0] > 1:
        # 使用 estimate_mid_plane 返回的平面 (n, c) 区分左右
        n_t = torch.as_tensor(n, device=V_up.device, dtype=V_up.dtype)
        c_t = torch.as_tensor(c, device=V_up.device, dtype=V_up.dtype)
        n_t = F.normalize(n_t, dim=0)

        signed = (contact - c_t) @ n_t  # (p - c)·n
        left_mask = signed < 0
        right_mask = ~left_mask

        zero = torch.tensor(0.0, device=V_up.device)
        # print("contact[left_mask]: ", int(left_mask.sum().item()))
        # print("contact[right_mask]: ", int(right_mask.sum().item()))
        if 1:  # 保存接触点
            if not hasattr(compute_loss, "_iter"):
                compute_loss._iter = 0
            compute_loss._iter += 1
            print("Contact Point Num:{} ,  Iter: {} ".format(len(contact), compute_loss._iter))
            contact_left_path = "/home/jelly/Projects/OcclusionPINN_2025_10_16/data/contact_points_left" + str(compute_loss._iter) + ".txt"
            with open(contact_left_path, "a") as f:
                for i in range(len(contact[left_mask])):
                    f.write(f"{contact[left_mask][i,0].item()} {contact[left_mask][i,1].item()} {contact[left_mask][i,2].item()}\n")

            contact_right_path = "/home/jelly/Projects/OcclusionPINN_2025_10_16/data/contact_points_right" + str(compute_loss._iter) + ".txt"
            with open(contact_right_path, "a") as f:
                for i in range(len(contact[right_mask])):
                    f.write(f"{contact[right_mask][i,0].item()} {contact[right_mask][i,1].item()} {contact[right_mask][i,2].item()}\n")

        cont_left = torch.var(contact[left_mask], dim=0, unbiased=False).sum() if left_mask.any() else zero
        cont_right = torch.var(contact[right_mask], dim=0, unbiased=False).sum() if right_mask.any() else zero
        print("cont_left:", cont_left.item(), "cont_right:", cont_right.item())
        cont = cont_left + cont_right
        if left_mask.sum() < 5 or right_mask.sum() < 5:
            print("Warning: 左右接触点数量不足，可能导致损失波动大。")
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

def sample_contact_points(V_up, V_low, delta=0.1, max_pts=15000, chunk=3000):
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
    if 1: # 保存接触点 
        if not hasattr(sample_contact_points, "_iter"):
            sample_contact_points._iter = 0
        iter = sample_contact_points._iter
        sample_contact_points._iter += 1
        print("contact ",len(contact))
        contact_path = "/home/jelly/Projects/OcclusionPINN_2025_10_16/data/contact_points_"+str(sample_contact_points._iter )+".txt"
        for i in range(len(contact)):
            with open(contact_path, "a") as f:
                f.write(f"{contact[i,0].item()} {contact[i,1].item()} {contact[i,2].item()}\n")
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