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

def capture_screenshot_with_transparency(upper_mesh, lower_mesh, output_dir="z_transparent_views"):
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
    
    # 自动截图
    # print("\n正在生成多视角透明截图...")
    capture_screenshot_with_transparency(upper_mesh, lower_mesh)
    # # 尝试使用支持透明度的新版 API
    # if hasattr(o3d.visualization, "draw"):
    #     print("使用 Open3D 新版可视化 API (支持透明度)")
    #     import open3d.visualization.rendering as rendering
        
    #     # 上颌透明材质
    #     mat_upper = rendering.MaterialRecord()
    #     mat_upper.shader = "defaultLitTransparency"
    #     mat_upper.base_color = [0.8, 0.2, 0.2, 0.5]  # RGBA，alpha=0.5 表示透明
        
    #     # 下颌不透明材质
    #     mat_lower = rendering.MaterialRecord()
    #     mat_lower.shader = "defaultLit"
        
    #     # 坐标系材质
    #     mat_coord = rendering.MaterialRecord()
    #     mat_coord.shader = "defaultLit"
        
    #     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        
    #     # 使用新版 draw API
    #     o3d.visualization.draw([
    #         {'name': 'upper_jaw', 'geometry': upper_mesh, 'material': mat_upper},
    #         {'name': 'lower_jaw', 'geometry': lower_mesh, 'material': mat_lower},
    #         {'name': 'coordinate', 'geometry': coordinate_frame, 'material': mat_coord}
    #     ], title='上下颌咬合状态 (上颌透明)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化上下颌咬合状态，上颌透明显示")
    parser.add_argument("-u", "--upper", default="../data/upper.ply", help="上颌PLY文件路径")
    parser.add_argument("-l", "--lower", default="../data/lower.ply", help="下颌PLY文件路径")
    args = parser.parse_args()
    
    try:
        visualize_transparent_jaw(args.upper, args.lower)
    except Exception as e:
        print(f"错误: {e}")
