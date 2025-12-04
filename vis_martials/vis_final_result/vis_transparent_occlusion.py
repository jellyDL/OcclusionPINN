import open3d as o3d
import numpy as np
import argparse
import os
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def load_mesh(file_path):
    """加载PLY网格文件"""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"无法加载网格文件: {file_path}")
    mesh.compute_vertex_normals()
    return mesh

def visualize_transparent_jaw(upper_file, lower_file):
    """
    可视化上下颌，上颌透明显示
    """ 
    print("加载上颌网格...")
    upper_mesh = load_mesh(upper_file)

    print("加载下颌网格...")
    lower_mesh = load_mesh(lower_file)
    
    # 先截图保存（使用兼容方式）
    print("\n正在生成 Z 轴方向截图...")
    os.makedirs("z_transparent_views", exist_ok=True)
    
    # 尝试使用支持透明度的新版 API
    if hasattr(o3d.visualization, "draw"):
        print("使用 Open3D 新版可视化 API (支持透明度)")
        import open3d.visualization.rendering as rendering
        
        # 上颌透明材质
        mat_upper = rendering.MaterialRecord()
        mat_upper.shader = "defaultLitTransparency"
        mat_upper.base_color = [0.8, 0.2, 0.2, 0.5]  # RGBA，alpha=0.5 表示透明
        
        # 下颌不透明材质
        mat_lower = rendering.MaterialRecord()
        mat_lower.shader = "defaultLit"
        
        # 坐标系材质
        mat_coord = rendering.MaterialRecord()
        mat_coord.shader = "defaultLit"
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        
        # 使用新版 draw API
        o3d.visualization.draw([
            {'name': 'upper_jaw', 'geometry': upper_mesh, 'material': mat_upper},
            {'name': 'lower_jaw', 'geometry': lower_mesh, 'material': mat_lower},
            {'name': 'coordinate', 'geometry': coordinate_frame, 'material': mat_coord}
        ], title='上下颌咬合状态 (上颌透明)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化上下颌咬合状态，上颌透明显示")
    parser.add_argument("-u", "--upper", default="../data/upper.ply", help="上颌PLY文件路径")
    parser.add_argument("-l", "--lower", default="../data/lower.ply", help="下颌PLY文件路径")
    parser.add_argument("--z-steps", type=int, default=15, help="Z轴序列截图帧数")
    parser.add_argument("--no-sequence", action="store_true", help="跳过Z轴序列截图")
    args = parser.parse_args()
    
    try:
        visualize_transparent_jaw(args.upper, args.lower)
    except Exception as e:
        print(f"错误: {e}")
