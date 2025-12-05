import open3d as o3d
import numpy as np
import argparse
import os
import sys

# 添加父目录到路径以支持跨目录导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vis_occusion_heatmap.vis_heatmap import visualize_occlusion_heatmap
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

def load_mesh(file_path):
    """加载PLY网格文件"""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"无法加载网格文件: {file_path}")
    mesh.compute_vertex_normals()
    return mesh

def capture_screenshot_with_transparency(upper_mesh, lower_mesh, zoom_factor, output_dir="."):
    """
    使用 GUI 渲染器捕获透明网格的截图
    """
    os.makedirs(output_dir, exist_ok=True)

    app = gui.Application.instance
    app.initialize()

    # 创建窗口
    window = app.create_window("透明网格截图", 960, 860)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # 设置背景色（白色）
    scene.scene.set_background([1, 1, 1, 1])

    # 上颌透明材质 - 保留原始纹理/顶点颜色
    mat_upper = rendering.MaterialRecord()
    mat_upper.shader = "defaultLitTransparency"
    mat_upper.base_color = [1.0, 1.0, 1.0, 0.6]  # 白色基底 + 透明度，让原始颜色显示
    mat_upper.base_roughness = 0.3
    mat_upper.base_reflectance = 0.5

    # 下颌不透明材质 - 保留原始纹理/顶点颜色
    mat_lower = rendering.MaterialRecord()
    mat_lower.shader = "defaultLit"
    mat_lower.base_color = [1.0, 1.0, 1.0, 0.8]  # 白色基底，让原始颜色显示
    mat_lower.base_roughness = 0.3
    mat_lower.base_reflectance = 0.5

    # 添加网格到场景
    scene.scene.add_geometry("upper_jaw", upper_mesh, mat_upper)
    scene.scene.add_geometry("lower_jaw", lower_mesh, mat_lower)

    # 添加坐标系
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    # mat_coord = rendering.MaterialRecord()
    # mat_coord.shader = "defaultUnlit"
    # scene.scene.add_geometry("coordinate", coordinate_frame, mat_coord)

    # 设置相机参数
    bounds = scene.scene.bounding_box
    center = bounds.get_center()

    # 定义多个视角
    views = {
        "front": ([center[0], center[1], center[2] + bounds.get_extent()[2] * 2], center, [0, -1, 0]),
        # "back": ([center[0], center[1], center[2] - bounds.get_extent()[2] * 2], center, [0, 1, 0]),
        # "left": ([center[0] - bounds.get_extent()[0] * 2, center[1], center[2]], center, [0, 1, 0]),
        # "right": ([center[0] + bounds.get_extent()[0] * 2, center[1], center[2]], center, [0, 1, 0]),
        # "top": ([center[0], center[1] + bounds.get_extent()[1] * 2, center[2]], center, [0, 0, -1]),
        # "bottom": ([center[0], center[1] - bounds.get_extent()[1] * 2, center[2]], center, [0, 0, 1]),
    }

    screenshot_taken = [False]
    view_index = [0]
    view_names = list(views.keys())

    def capture_view():
        if view_index[0] < len(view_names):
            view_name = view_names[view_index[0]]
            eye, lookat, up = views[view_name]

            # 设置相机
            scene.setup_camera(60, bounds, center)
            scene.look_at(center, eye, up)
            
            # 设置缩放（通过调整视野角度或相机距离）
            # 获取当前相机并调整
            cam = scene.scene.camera
            # 缩放因子：值越小，物体在画面中越大
            new_eye = [
                center[0] + (eye[0] - center[0]) * zoom_factor,
                center[1] + (eye[1] - center[1]) * zoom_factor,
                center[2] + (eye[2] - center[2]) * zoom_factor
            ]
            scene.look_at(center, new_eye, up)

            # 延迟截图以确保渲染完成
            def do_screenshot():
                output_path = os.path.join(output_dir, f"view_{view_name}.png")
                scene.scene.scene.render_to_image(lambda img: save_image(img, output_path))
                print(f"已保存截图: {output_path}")

                view_index[0] += 1
                if view_index[0] < len(view_names):
                    app.post_to_main_thread(window, capture_view)
                else:
                    app.post_to_main_thread(window, window.close)

            # 延迟执行以确保渲染完成
            app.run_one_tick()
            app.post_to_main_thread(window, do_screenshot)

    def save_image(image, path):
        o3d.io.write_image(path, image)

    # 开始截图
    app.post_to_main_thread(window, capture_view)

    app.run()
    
def visualize_transparent_jaw(upper_file, lower_file, threshold, zoom_factor, colormap, out_dir):
    """
    可视化上下颌，上颌透明显示
    """ 
    print("加载上颌网格...")
    upper_mesh = load_mesh(upper_file)
    
    print("加载下颌网格...")
    lower_mesh = load_mesh(lower_file)
    
    upper_mesh, lower_mesh, distance_range = visualize_occlusion_heatmap(
        upper_file, lower_file, colormap, threshold, out_dir=out_dir
    )
    
    # 自动截图
    capture_screenshot_with_transparency(upper_mesh, lower_mesh, zoom_factor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化上下颌咬合状态，上颌透明显示")
    parser.add_argument("-u", "--upper", default="../data/upper.ply", help="上颌PLY文件路径")
    parser.add_argument("-l", "--lower", default="../data/lower.ply", help="下颌PLY文件路径")
    parser.add_argument("-c", "--case", default="", help="案例名称")
    parser.add_argument("--threshold", type=float, default=0.5, help="咬合阈值 mm")
    parser.add_argument("-z", "--zoom_factor", type=float, default=0.95, help="缩放因子：值越小，物体在画面中越大")
    parser.add_argument("--colormap", default="jet", help="colormap name")
    parser.add_argument("--out_dir", default=".", help="输出目录")
    args = parser.parse_args()
    
    if args.case != "":
        upper = args.case+"-UpperJaw.stl"
        if not os.path.exists(upper):
            upper = upper[:-4]+".ply"
        
        lower = args.case+"-LowerJaw.stl"
        if not os.path.exists(lower):
            lower = upper[:-4]+".ply"
    else:
        upper = args.upper
        lower = args.lower
        
    try:
        visualize_transparent_jaw(upper, lower,
                                  args.threshold, args.zoom_factor, args.colormap, args.out_dir)
    except Exception as e:
        print(f"错误: {e}")
