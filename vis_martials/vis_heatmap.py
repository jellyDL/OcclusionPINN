import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    upper_file = "data/test1_upper.ply"
    lower_file = "data/test1_lower_open.ply"
    
    # 咬合距离阈值（单位：mm）
    threshold = 0.5
    
    # 可选的colormap: 'jet', 'rainbow', 'viridis', 'plasma', 'hot', 'cool'
    colormap = 'jet'
    
    try:
        upper_mesh, lower_mesh, distance_range = visualize_occlusion_heatmap(
            upper_file, lower_file, colormap, threshold
        )
        print("\n可视化完成！")
        
        # 可选：保存着色后的网格
        # o3d.io.write_triangle_mesh("upper_heatmap.ply", upper_mesh)
        # o3d.io.write_triangle_mesh("lower_heatmap.ply", lower_mesh)
        
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")
        