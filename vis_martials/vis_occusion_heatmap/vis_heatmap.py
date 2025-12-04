import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import argparse
import matplotlib as mpl
from PIL import Image

"功能说明: 可视化上下颌的咬合状态，增加热力分布显示功能，并分上下颌分别截图保存"
"exp python vis_heatmap.py --lower data/test1_lower_open.ply"

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
    
    # 归一化距离值 (反转映射：小值->0.0->红色，大值->1.0->蓝色)
    normalized_distances = np.zeros_like(distances)
    if np.sum(mask) > 0:
        valid_distances = distances[mask]
        # 反转归一化：0mm->1.0(红色), 0.5mm->0.0(蓝色)
        normalized_distances[mask] = 1.0 - (valid_distances - vmin) / (vmax - vmin)
    
    # 应用colormap (jet: 0->红色, 1->蓝色)
    cmap = cm.get_cmap(colormap)
    colors = cmap(normalized_distances)[:, :3]
    
    # 对于超过阈值的顶点，设置为灰色（不显著）
    colors[~mask] = [0.7, 0.7, 0.7]  # 灰色
    
    # 设置网格颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh, vmin, vmax

# 新增：把 colorbar 拼接到截图右侧
def add_colorbar_to_image(image_path, out_path, vmin=0.0, vmax=1.0, colormap='jet', pad_ratio=0.1, cbar_width_ratio=0.12):
	# 读取主图
	main_img = Image.open(image_path).convert("RGBA")
	h = main_img.height
	# 生成 colorbar（matplotlib）- 调整高度为主图的 60%
	cbar_height_ratio = 0.85  # 新增：控制 colorbar 高度占主图比例
	cbar_h = int(h * cbar_height_ratio)
	fig = plt.figure(figsize=(1.0, cbar_h/100.0), dpi=150)
	ax = fig.add_axes([0.1, 0.1, 0.3, 0.8])
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	cmap = mpl.cm.get_cmap(colormap)
	# 反转 colormap：让小值（底部）显示红色，大值（顶部）显示蓝色
	cmap_reversed = cmap.reversed()
	cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap_reversed, norm=norm, orientation='vertical')
	cb.set_label('Distance(mm)', fontsize=20)
	# 设置刻度标签字体大小
	cb.ax.tick_params(labelsize=18)
	tmp_cbar = os.path.splitext(out_path)[0] + "_cbar_tmp.png"
	fig.savefig(tmp_cbar, bbox_inches='tight', pad_inches=0.1)
	plt.close(fig)
	cbar_img = Image.open(tmp_cbar).convert("RGBA")
	# 调整 colorbar 宽度（高度已由 figsize 控制）
	cbar_w = max(140, int(h * cbar_width_ratio))
	print("Resizing colorbar to :", cbar_w, cbar_h)
	cbar_img = cbar_img.resize((cbar_w, cbar_h), resample=Image.BICUBIC)
	# 组合 - colorbar 垂直居中放置
	pad = max(4, int(h * pad_ratio))
	canvas = Image.new("RGBA", (main_img.width + pad + cbar_img.width, h), (255, 255, 255, 255))
	canvas.paste(main_img, (0, 0))
	cbar_y_offset = (h - cbar_h) // 2  # 居中
	canvas.paste(cbar_img, (main_img.width + pad, cbar_y_offset), cbar_img)
	canvas.convert("RGB").save(out_path)
	try:
		os.remove(tmp_cbar)
	except Exception:
		pass
	print("Saved screenshot with colorbar:", out_path)

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

def visualize_occlusion_heatmap(upper_file, lower_file, colormap='jet', threshold=0.5, out_dir="."):
    """
    主函数：可视化牙颌咬合热力图
    threshold: 只显示小于此距离的咬合区域 (单位: mm)
    """
    print("加载上颌网格...")
    upper_mesh = load_mesh(upper_file)
    
    print("加载下颌网格...")
    lower_mesh = load_mesh(lower_file)
    
    # 使用顶点法线以获得更好的光照/细节表现
    if not upper_mesh.has_vertex_normals():
        upper_mesh.compute_vertex_normals()
    if not lower_mesh.has_vertex_normals():
        lower_mesh.compute_vertex_normals()
    
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
    # 着色后务必重新计算顶点法线，确保光照与高光显示网格细节
    try:
        upper_mesh_colored.compute_vertex_normals()
    except Exception:
        pass
    try:
        lower_mesh_colored.compute_vertex_normals()
    except Exception:
        pass
    
    # 可视化
    print("准备可视化...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='牙颌咬合热力图', width=960, height=800)
    
    vis.add_geometry(upper_mesh_colored)
    vis.add_geometry(lower_mesh_colored)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True
    # 可选：调整背景色与增强光照对比，方便观察细节
    try:
        opt.background_color = np.asarray([1.0, 1.0, 1.0])  # 白底
    except Exception:
        pass
    
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
    
    # vis.run()
    # vis.destroy_window()
    
    return upper_mesh_colored, lower_mesh_colored, (vmin, vmax)

def capture_top_to_bottom(mesh, type, out_dir, steps=10, img_size=(800,960), prefix="vertical"):
    """
    沿 Z 轴从上到下截取多张图并保存。
    mesh: open3d TriangleMesh
    out_dir: 保存目录
    steps: 帧数
    img_size: (width, height)
    """
    os.makedirs(out_dir, exist_ok=True)
    # # 使用单个 mesh 的包围框计算中心与尺度
    aabb = mesh.get_axis_aligned_bounding_box()
    mins = np.asarray(aabb.get_min_bound())
    maxs = np.asarray(aabb.get_max_bound())
    
    center = (mins + maxs) / 2.0
    extent = np.max(maxs - mins)
    distance = max(0.1, extent * 2.2)

    # 准备可视化窗口（离屏）
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=False)

    vis.add_geometry(mesh)
    vc = vis.get_view_control()
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True

    # 从上到下：z 从 center+distance 到 center-distance
    if type == "upperjaw":
        z_offset = distance  # 从 +distance 到 -distance
        zoom_factor = 0.54
    elif type == "lowerjaw":
        z_offset = -distance  # 从 +distance 到 -distance
        zoom_factor = 0.5
        
        
    eye = center + np.array([0.0, 0.0, z_offset])
    front = (center - eye)
    front = front / np.linalg.norm(front)
    # 设置相机
    vc.set_lookat(center.tolist())
    vc.set_front(front.tolist())
    # 保持顶向量一致（可根据需要调整）
    vc.set_up([0.0, 1.0, 0.0])
    vc.set_zoom(zoom_factor)

    vis.poll_events()
    vis.update_renderer()
    fname = os.path.join(out_dir, f"{prefix}_{str(type)}.png")
    vis.capture_screen_image(fname, do_render=True)
    
    if type == "lowerjaw":
        # add colorbar
        fname_with_cbar = os.path.join(out_dir, "heatmap_with_colorbar.png")
        add_colorbar_to_image(fname, fname_with_cbar, vmin=0, vmax=0.5, colormap=colormap)

    vis.destroy_window()

def combine_upper_lower_images(upper_img_path, lower_with_cbar_path, out_path, bg_color=(255,255,255)):
    """
    将上颌图片旋转180度后，右侧填白使其与下颌带colorbar图等宽，然后垂直拼接。
    
    Args:
        upper_img_path: vertical_upperjaw.png 路径
        lower_with_cbar_path: heatmap_with_colorbar.png 路径
        out_path: 输出拼接后的图片路径
        bg_color: 填充背景色 (R,G,B)
    """
    # 读取两张图片
    upper_img = Image.open(upper_img_path).convert("RGB")
    lower_img = Image.open(lower_with_cbar_path).convert("RGB")
    
    # 1. 旋转上颌图片180度
    upper_rotated = upper_img.rotate(180, expand=True)
    
    # 2. 获取下颌图片宽度（目标宽度）
    target_width = lower_img.width
    upper_h = upper_rotated.height
    
    # 3. 如果上颌图宽度小于目标宽度，右侧填白
    if upper_rotated.width < target_width:
        new_upper = Image.new("RGB", (target_width, upper_h), bg_color)
        new_upper.paste(upper_rotated, (0, 0))
        upper_final = new_upper
    else:
        # 如果上颌图更宽，居中裁剪
        crop_x = (upper_rotated.width - target_width) // 2
        upper_final = upper_rotated.crop((crop_x, 0, crop_x + target_width, upper_h))
    
    # 4. 垂直拼接（上颌在上，下颌在下）
    total_height = upper_final.height + lower_img.height
    canvas = Image.new("RGB", (target_width, total_height), bg_color)
    canvas.paste(upper_final, (0, 0))
    canvas.paste(lower_img, (0, upper_final.height))
    
    canvas.save(out_path)
    print("Saved combined image:", out_path)

# 使用示例
if __name__ == "__main__":
    # 替换原示例入口，增加保存从上往下截图的参数
    parser = argparse.ArgumentParser(description="可视化并可选保存牙颌咬合热力图和从上往下截图序列")
    parser.add_argument("--upper", default="data/test1_upper.ply", help="Upper jaw PLY file")
    parser.add_argument("--lower", default="data/test1_lower_final.ply", help="Lower jaw PLY file")
    parser.add_argument("--threshold", type=float, default=0.5, help="咬合阈值 mm")
    parser.add_argument("--colormap", default="jet", help="colormap name")
    parser.add_argument("--out_dir", default=".", help="输出目录")
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
            upper_file, lower_file, colormap, threshold, out_dir=out_dir
        )
        print("\n可视化完成！")

        # 若需要保存从上往下的截图序列（保存并退出）
        # 使用着色后的网格进行截图（上/下均为 open3d mesh）
        print(f"开始保存从上往下的截图...")
        capture_top_to_bottom(upper_mesh, "upperjaw", out_dir, steps=args.vertical_steps, img_size=(860,800))
        capture_top_to_bottom(lower_mesh, "lowerjaw", out_dir, steps=args.vertical_steps, img_size=(860,800))
        print("序列保存完成。")
        
        # 拼接上下颌图片
        upper_img_path = os.path.join(out_dir, "vertical_upperjaw.png")
        lower_with_cbar_path = os.path.join(out_dir, "heatmap_with_colorbar.png")
        combined_path = os.path.join(out_dir, "combined_upper_lower.png")
        if os.path.exists(upper_img_path) and os.path.exists(lower_with_cbar_path):
            combine_upper_lower_images(upper_img_path, lower_with_cbar_path, combined_path)
        else:
            print("警告: 缺少必要图片，无法生成拼接图。")
                
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
    except Exception as e:
        print(f"错误: {e}")
