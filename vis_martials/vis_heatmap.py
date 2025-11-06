import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import argparse

def load_mesh(file_path):
    """加载PLY网格文件"""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"无法加载网格文件: {file_path}")
    return mesh

def compute_mesh_to_mesh_distance(source_mesh, target_mesh):
    """
    计算源网格到目标网格的点对点距离
    """
    # 将网格转换为点云
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = source_mesh.vertices
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = target_mesh.vertices
    
    # 为目标点云构建KD树以加速最近邻搜索
    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    
    # 计算每个源点到目标网格的最近距离
    distances = []
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    for point in source_points:
        [_, idx, _] = target_tree.search_knn_vector_3d(point, 1)
        nearest_point = target_points[idx[0]]
        dist = np.linalg.norm(point - nearest_point)
        distances.append(dist)
    
    return np.array(distances)

def apply_colormap_to_mesh(mesh, distances, colormap='jet', vmin=None, vmax=None, threshold=0.5):
    """
    根据距离值为网格应用颜色映射
    只显示距离小于threshold的区域，距离近的为红色，远的为蓝色
    """
    # 设置距离范围
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = threshold
    
    # 创建掩码：只处理距离小于阈值的顶点
    mask = distances <= threshold
    
    # 归一化距离值 (反转：近的值大，远的值小，这样红色表示近，蓝色表示远)
    normalized_distances = np.zeros_like(distances)
    if np.sum(mask) > 0:
        valid_distances = distances[mask]
        # 反转归一化：0mm->1.0(红色), 0.5mm->0.0(蓝色)
        normalized_distances[mask] = 1.0 - (valid_distances - vmin) / (vmax - vmin)
    
    # 应用colormap (jet: 0->蓝色, 1->红色)
    cmap = cm.get_cmap(colormap)
    colors = cmap(normalized_distances)[:, :3]
    
    # 对于超过阈值的顶点，设置为灰色（不显著）
    colors[~mask] = [0.7, 0.7, 0.7]  # 灰色
    
    # 设置网格颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh, vmin, vmax

def create_colorbar_mesh(vmin, vmax, colormap='jet', height=100, width=20):
    """
    创建颜色条网格用于显示
    """
    # 创建颜色条纹理
    values = np.linspace(1, 0, height)  # 从上到下
    colorbar_array = np.tile(values.reshape(-1, 1), (1, width))
    
    cmap = cm.get_cmap(colormap)
    colors = cmap(colorbar_array.flatten())[:, :3]
    
    # 创建简单的平面网格作为颜色条
    vertices = []
    triangles = []
    vertex_colors = []
    
    for i in range(height):
        for j in range(width):
            vertices.append([j * 0.1, (height - i) * 0.1, 0])
            vertex_colors.append(colors[i * width + j])
    
    # 创建三角形
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            triangles.append([idx, idx + width, idx + 1])
            triangles.append([idx + 1, idx + width, idx + width + 1])
    
    colorbar_mesh = o3d.geometry.TriangleMesh()
    colorbar_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    colorbar_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    colorbar_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return colorbar_mesh

def visualize_occlusion_heatmap(upper_file, lower_file, colormap='jet', threshold=0.5):
    """
    主函数：可视化牙颌咬合热力图
    threshold: 只显示小于此距离的咬合区域 (单位: mm)
    """
    print("加载上颌网格...")
    upper_mesh = load_mesh(upper_file)
    
    print("加载下颌网格...")
    lower_mesh = load_mesh(lower_file)
    
    # 确保网格有法向量
    if not upper_mesh.has_triangle_normals():
        upper_mesh.compute_triangle_normals()
    if not lower_mesh.has_triangle_normals():
        lower_mesh.compute_triangle_normals()
    
    print("计算上颌到下颌的距离...")
    upper_distances = compute_mesh_to_mesh_distance(upper_mesh, lower_mesh)
    
    print("计算下颌到上颌的距离...")
    lower_distances = compute_mesh_to_mesh_distance(lower_mesh, upper_mesh)
    
    # 统计信息
    upper_close = np.sum(upper_distances <= threshold)
    lower_close = np.sum(lower_distances <= threshold)
    print(f"\n距离统计:")
    print(f"上颌总顶点数: {len(upper_distances)}, 咬合区域(<{threshold}mm): {upper_close} ({100*upper_close/len(upper_distances):.1f}%)")
    print(f"下颌总顶点数: {len(lower_distances)}, 咬合区域(<{threshold}mm): {lower_close} ({100*lower_close/len(lower_distances):.1f}%)")
    print(f"上颌咬合区域距离范围: {np.min(upper_distances[upper_distances<=threshold]):.3f} - {threshold:.3f} mm")
    print(f"下颌咬合区域距离范围: {np.min(lower_distances[lower_distances<=threshold]):.3f} - {threshold:.3f} mm")
    
    # 应用颜色映射（只显示threshold范围内）
    print(f"\n应用颜色映射（阈值: {threshold}mm）...")
    upper_mesh_colored, vmin, vmax = apply_colormap_to_mesh(
        upper_mesh, upper_distances, colormap, vmin=0.0, vmax=threshold, threshold=threshold
    )
    lower_mesh_colored, _, _ = apply_colormap_to_mesh(
        lower_mesh, lower_distances, colormap, vmin=0.0, vmax=threshold, threshold=threshold
    )
    
    # 可视化
    print("准备可视化...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='牙颌咬合热力图', width=1200, height=800)
    
    vis.add_geometry(upper_mesh_colored)
    vis.add_geometry(lower_mesh_colored)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    print("\n=== 操作说明 ===")
    print("- 鼠标左键拖动: 旋转视图")
    print("- 鼠标滚轮: 缩放")
    print("- 鼠标右键拖动: 平移视图")
    print("- 按 Q 或关闭窗口: 退出")
    print(f"\n颜色说明 (咬合距离 0-{threshold}mm):")
    print(f"  红色: 紧密接触 (~0.0 mm)")
    print(f"  黄色/绿色: 中等距离")
    print(f"  蓝色: 咬合边缘 (~{threshold} mm)")
    print(f"  灰色: 非咬合区域 (>{threshold} mm)")
    
    vis.run()
    vis.destroy_window()
    
    return upper_mesh_colored, lower_mesh_colored, (vmin, vmax)

def capture_top_to_bottom(meshes, out_dir, steps=10, img_size=(800,800), prefix="vertical"):
    """
    沿 Z 轴从上到下截取多张图并保存。
    meshes: list of open3d TriangleMesh
    out_dir: 保存目录
    steps: 帧数
    img_size: (width, height)
    """
    os.makedirs(out_dir, exist_ok=True)
    # 计算总中心与尺度
    all_bounds = np.vstack([np.asarray(m.get_axis_aligned_bounding_box().get_min_bound())[None,:] for m in meshes])
    # 更稳妥地使用 min/max across meshes
    mins = np.min([np.asarray(m.get_axis_aligned_bounding_box().get_min_bound()) for m in meshes], axis=0)
    maxs = np.max([np.asarray(m.get_axis_aligned_bounding_box().get_max_bound()) for m in meshes], axis=0)
    center = (mins + maxs) / 2.0
    extent = np.max(maxs - mins)
    distance = max(0.1, extent * 2.2)

    # 准备可视化窗口（离屏）
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=False)
    for geom in meshes:
        vis.add_geometry(geom)
    vc = vis.get_view_control()
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True

    # 从上到下：z 从 center+distance 到 center-distance
    # for i in range(steps):
    for i in [0,5]:
        t = 0.0 if steps == 1 else i / (steps - 1)
        z_offset = distance * (1.0 - 2.0 * t)  # 从 +distance 到 -distance
        eye = center + np.array([0.0, 0.0, z_offset])
        front = (center - eye)
        front = front / np.linalg.norm(front)
        # 设置相机
        vc.set_lookat(center.tolist())
        vc.set_front(front.tolist())
        # 保持顶向量一致（可根据需要调整）
        vc.set_up([0.0, 1.0, 0.0])
        vc.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()
        fname = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        vis.capture_screen_image(fname, do_render=True)
        print(f"Saved: {fname}")

    vis.destroy_window()

# 使用示例
if __name__ == "__main__":
    # 替换原示例入口，增加保存从上往下截图的参数
    parser = argparse.ArgumentParser(description="可视化并可选保存牙颌咬合热力图和从上往下截图序列")
    parser.add_argument("--upper", default="data/test1_upper.ply", help="Upper jaw PLY file")
    parser.add_argument("--lower", default="data/test1_lower_final.ply", help="Lower jaw PLY file")
    parser.add_argument("--threshold", type=float, default=0.5, help="咬合阈值 mm")
    parser.add_argument("--colormap", default="jet", help="colormap name")
    parser.add_argument("--out_dir", default=".", help="输出目录")
    parser.add_argument("--save_vertical_steps", action="store_true", help="是否保存从上往下的截图序列")
    parser.add_argument("--vertical_steps", type=int, default=10, help="从上往下截图的帧数")
    args = parser.parse_args()

    upper_file = args.upper
    lower_file = args.lower
    threshold = args.threshold
    colormap = args.colormap
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    try:
        upper_mesh, lower_mesh, distance_range = visualize_occlusion_heatmap(
            upper_file, lower_file, colormap, threshold
        )
        print("\n可视化完成！")

        # 若需要保存从上往下的截图序列（保存并退出）
        if args.save_vertical_steps:
            # 使用着色后的网格进行截图（上/下均为 open3d mesh）
            print(f"开始保存从上往下的截图序列，共 {args.vertical_steps} 帧...")
            capture_top_to_bottom([upper_mesh, lower_mesh], out_dir, steps=args.vertical_steps, img_size=(800,800))
            print("序列保存完成。")
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")
