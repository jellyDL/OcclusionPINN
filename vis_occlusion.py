import os
from pathlib import Path
import numpy as np
import open3d as o3d


def _estimate_plane_normal(points: np.ndarray) -> np.ndarray:
    """PCA 拟合平面法向（最小特征值对应特征向量）"""
    pts = points - points.mean(axis=0, keepdims=True)
    # 3x3 协方差
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, np.argmin(eigvals)]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n


def _safe_up_from_normal(normal: np.ndarray) -> np.ndarray:
    """根据法向生成稳定的 up 向量（世界Z轴在平面内的投影）"""
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(world_up, normal)) > 0.9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    up = world_up - normal * np.dot(world_up, normal)
    up = up / (np.linalg.norm(up) + 1e-12)
    return up


def _load_points_from_txt(path: Path) -> np.ndarray:
    """读取txt为Nx3点集，支持逗号/空白分隔，忽略注释与空行"""
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = s.split()
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            for i in range(0, len(nums) - 2, 3):
                pts.append([nums[i], nums[i + 1], nums[i + 2]])
    if not pts:
        raise ValueError(f"未在 {path} 解析到任何 (x,y,z) 点")
    return np.asarray(pts, dtype=float)


def main():
    root = Path(__file__).resolve().parent
    upper_path = root / "data" / "upper.ply"
    lower_path = root / "data" / "lower.ply"
    cpl_path = root / "data" / "contact_points_left2.txt"
    cpr_path = root / "data" / "contact_points_right2.txt"
    out_path = root / "occlusal_view.png"

    #如果输入参数大于3
    if len(os.sys.argv) > 2:
        iter = os.sys.argv[1]
        lower_path = "data/lower_bite_" + iter + ".ply"
        cpl_path = "data/contact_points_left" + iter + ".txt"
        cpr_path = "data/contact_points_right" + iter + ".txt"
    else:
        print("Usage: python vis_occlusion.py <lower path> <iter>")
        print("  Exp: python vis_occlusion.py lower_bite_10.ply 10")
        
    print("\n################################################")
    print("Upper:", upper_path)
    print("Lower:", lower_path)
    print("CPL:", cpl_path)
    print("CPR:", cpr_path)
    print("Output:", out_path)
    print("################################################\n")

    if not upper_path.exists() or not lower_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {upper_path} 或 {lower_path}")

    upper = o3d.io.read_triangle_mesh(str(upper_path))
    lower = o3d.io.read_triangle_mesh(str(lower_path))

    if upper.is_empty() or lower.is_empty():
        raise ValueError("读取到的网格为空，请检查PLY文件内容。")

    # 加载接触点
    if not cpl_path.exists() or not cpr_path.exists():
        raise FileNotFoundError(f"未找到接触点文件: {cpl_path, cpr_path}")
    cpl_points = _load_points_from_txt(cpl_path)
    pcdl = o3d.geometry.PointCloud()
    pcdl.points = o3d.utility.Vector3dVector(cpl_points)
    pcdl.paint_uniform_color([1.0, 0.85, 0.0])  # 黄色
    cpr_points = _load_points_from_txt(cpr_path)
    pcdr = o3d.geometry.PointCloud()
    pcdr.points = o3d.utility.Vector3dVector(cpr_points)
    pcdr.paint_uniform_color([0, 0.85, 0.0])  # 绿色

    # 基本处理与着色
    for m, color in [(upper, [0.9, 0.4, 0.4]), (lower, [0.4, 0.6, 0.9])]:
        if not m.has_vertex_normals():
            m.compute_vertex_normals()
        m.paint_uniform_color(color)

    # 收集顶点，PCA 拟合法向
    v_upper = np.asarray(upper.vertices)
    v_lower = np.asarray(lower.vertices)
    all_pts = np.vstack([v_upper, v_lower])

    normal = _estimate_plane_normal(all_pts)
    up = _safe_up_from_normal(normal)

    # 使用几何中心作为 lookat，eye 沿 -normal 拉开与场景尺度相关的距离
    center = all_pts.mean(axis=0)
    diag = all_pts.max(axis=0) - all_pts.min(axis=0)
    diag_len = float(np.linalg.norm(diag))
    dist = max(1e-6, 0.5 * diag_len)
    eye = center - normal * dist

    # 优先使用 OffscreenRenderer 进行半透明渲染；失败则回退 Visualizer
    width, height = 1280, 1080
    try:
        from open3d.visualization import rendering

        renderer = rendering.OffscreenRenderer(width, height)
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])

        # 半透明材质（重叠时可透视）
        mat_upper = rendering.MaterialRecord()
        mat_upper.shader = "defaultLitTransparency"
        mat_upper.base_color = [0.9, 0.4, 0.4, 0.8]  # RGBA，调节最后一个 alpha 达到更强/弱透视

        mat_lower = rendering.MaterialRecord()
        mat_lower.shader = "defaultLitTransparency"
        mat_lower.base_color = [0.4, 0.6, 0.9, 0.55]

        # 点云材质（不受光）
        mat_pts = rendering.MaterialRecord()
        mat_pts.shader = "defaultUnlit"
        mat_pts.point_size = 12.0

        scene.add_geometry("upper", upper, mat_upper)
        scene.add_geometry("lower", lower, mat_lower)
        scene.add_geometry("cpl", pcdl, mat_pts)
        scene.add_geometry("cpr", pcdr, mat_pts)

        # 相机
        scene.camera.look_at(center, eye, up)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = renderer.render_to_image()
        o3d.io.write_image(str(out_path), img)

        # 兼容不同 Open3D 版本的释放接口
        if hasattr(renderer, "release_resources"):
            renderer.release_resources()
        elif hasattr(renderer, "release"):
            renderer.release()
        renderer = None
    except Exception:
        # 回退：使用 legacy Visualizer（透明度支持有限）
        vis = o3d.visualization.Visualizer()
        created = vis.create_window(window_name="Occlusal View", width=width, height=height, visible=False)

        vis.add_geometry(upper)
        vis.add_geometry(lower)
        vis.add_geometry(pcdl)
        vis.add_geometry(pcdr)

        # 尽量提升重叠时可见性
        opt = vis.get_render_option()
        if opt is not None:
            try:
                opt.mesh_show_back_face = True
                opt.mesh_show_wireframe = True
            except Exception:
                pass

        vis.poll_events()
        vis.update_renderer()

        ctr = vis.get_view_control() if created else None
        if ctr is not None:
            ctr.set_front(normal.tolist())
            ctr.set_up(up.tolist())
            ctr.set_lookat(center.tolist())
            ctr.set_zoom(0.6)

        vis.poll_events()
        vis.update_renderer()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(out_path), do_render=True)
        vis.destroy_window()

    print(f"截图已保存: {out_path}")


if __name__ == "__main__":
    main()
