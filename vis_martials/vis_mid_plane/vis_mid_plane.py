import os
import sys
import trimesh
import numpy as np
import open3d as o3d

# 将项目根目录加入 sys.path 以便导入 models
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

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

def estimate_mid_plane(points: np.ndarray):  # 估计单颌中切平面
    """估计切分单颌左右的中切平面：先选主轴，再优化偏移与小角度法向"""
    # 兼容 torch.Tensor（含 GPU），统一转为 CPU numpy
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


if __name__ == "__main__":

    ply_path = os.path.join(project_root, 'data', 'upper.ply')
    print("Loading upper jaw mesh from:", ply_path)

    upper = trimesh.load(ply_path, force='mesh')
    V_up  = np.asarray(upper.vertices)
    print("Upper Vertices:", V_up.shape)
    
    sample_count = 10000
    idx = np.random.choice(V_up.shape[0], size=min(sample_count, V_up.shape[0]), replace=False)
    sampled_points = V_up[idx]
    V_up = sampled_points
    print("After Sample, Upper Vertices:", V_up.shape)
    
    n, c, err = estimate_mid_plane(V_up)
    print(f"Normal: {n}")
    print(f"Center: {c}")
    
    #可视化上颌及中切平面
    # 1. 转换上颌网格为 Open3D 格式
    upper_o3d = o3d.geometry.TriangleMesh()
    upper_o3d.vertices = o3d.utility.Vector3dVector(upper.vertices)
    upper_o3d.triangles = o3d.utility.Vector3iVector(upper.faces)
    upper_o3d.compute_vertex_normals()
    
    # 尝试加载顶点颜色以呈现纹理
    if hasattr(upper.visual, 'vertex_colors') and len(upper.visual.vertex_colors) > 0:
        # trimesh 颜色通常是 (N, 4) uint8 RGBA，Open3D 需要 float RGB [0,1]
        colors = np.asarray(upper.visual.vertex_colors)[:, :3] / 255.0
        upper_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        upper_o3d.paint_uniform_color([0.75, 0.75, 0.75])

    # 2. 创建平面几何体
    # 计算包围盒对角线长度作为参考尺寸
    bbox_min = np.min(upper.vertices, axis=0)
    bbox_max = np.max(upper.vertices, axis=0)
    scale = np.linalg.norm(bbox_max - bbox_min)
    plane_size = scale * 0.3

    # 构建平面基向量
    n_unit = n / np.linalg.norm(n)
    # 找一个不共线的辅助向量
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n_unit, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    
    u = np.cross(n_unit, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(n_unit, u)
    
    # 平面四个顶点
    p1 = c - plane_size * u - plane_size * v
    p2 = c + plane_size * u - plane_size * v
    p3 = c + plane_size * u + plane_size * v
    p4 = c - plane_size * u + plane_size * v
    
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector([p1, p2, p3, p4])
    plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    plane_mesh.compute_vertex_normals()
    # plane_mesh.paint_uniform_color([0.2, 0.6, 1.0])  # 蓝色平面 (移至显示部分设置)

    # 3. 创建法向量箭头
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=scale*0.01, 
        cone_radius=scale*0.02, 
        cylinder_height=scale*0.15, 
        cone_height=scale*0.05
    )
    arrow.paint_uniform_color([1.0, 0.0, 0.0]) # 红色箭头
    
    # 旋转箭头使其指向法向量 n (默认指向Z轴)
    z_axis = np.array([0, 0, 1])
    if not np.allclose(n_unit, z_axis) and not np.allclose(n_unit, -z_axis):
        axis = np.cross(z_axis, n_unit)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, n_unit))
        R = arrow.get_rotation_matrix_from_axis_angle(axis * angle)
        arrow.rotate(R, center=[0,0,0])
    elif np.allclose(n_unit, -z_axis):
        R = arrow.get_rotation_matrix_from_axis_angle(np.array([1,0,0]) * np.pi)
        arrow.rotate(R, center=[0,0,0])
    arrow.translate(c)

    # 4. 显示
    print("Visualizing...")
    
    plane_alpha = 0.6  # 平面透明度 (0.0 - 1.0)

    # 尝试使用支持透明度的新版可视化 API (Open3D 0.13+)
    if hasattr(o3d.visualization, "draw"):
        import open3d.visualization.rendering as rendering
        
        # 平面材质：透明
        mat_plane = rendering.MaterialRecord()
        mat_plane.shader = "defaultLitTransparency"
        mat_plane.base_color = [0.2, 0.6, 1.0, plane_alpha]
        
        # 默认材质：不透明 (使用几何体颜色)
        mat_default = rendering.MaterialRecord()
        mat_default.shader = "defaultLit"
        
        o3d.visualization.draw([
            {'name': 'upper', 'geometry': upper_o3d, 'material': mat_default},
            {'name': 'plane', 'geometry': plane_mesh, 'material': mat_plane}
        ], title="Upper Jaw & Mid-Sagittal Plane")
    else:
        # 旧版 Visualizer 不支持网格透明度，回退到不透明
        print("Open3D version does not support 'draw' API for transparency. Showing opaque plane.")
        plane_mesh.paint_uniform_color([0.2, 0.6, 1.0])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Upper Jaw & Mid-Sagittal Plane", width=1024, height=768)
        vis.add_geometry(upper_o3d)
        vis.add_geometry(plane_mesh)
        vis.add_geometry(arrow)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.background_color = np.asarray([1, 1, 1]) # 白底
        opt.light_on = True
        
        vis.run()
        vis.destroy_window()
