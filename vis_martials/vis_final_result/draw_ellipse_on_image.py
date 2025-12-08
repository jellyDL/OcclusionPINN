import cv2
import argparse
import os
import numpy as np
import math

"在图片上绘制椭圆"
"python draw_ellipse_on_image.py -i view_front_case.png -o view_front_case_final.png"

def draw_dashed_ellipse(img, center, axes, angle, color, thickness=15, dash_length=40):
    """
    绘制虚线椭圆
    """
    center_x, center_y = center
    major_axis, minor_axis = axes
    angle_rad = np.deg2rad(angle)
    
    # 计算椭圆周长近似值，用于决定分段数量
    perimeter = 2 * np.pi * np.sqrt((major_axis**2 + minor_axis**2) / 2)
    num_segments = int(perimeter / dash_length)
    if num_segments % 2 != 0: num_segments += 1 # 保证偶数段
    
    theta = np.linspace(0, 2 * np.pi, num_segments + 1)
    
    # 参数方程计算点
    # x = a * cos(t)
    # y = b * sin(t)
    # 然后旋转和平移
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    pts = []
    for t in theta:
        x_local = major_axis * np.cos(t)
        y_local = minor_axis * np.sin(t)
        
        x_rot = x_local * cos_a - y_local * sin_a + center_x
        y_rot = x_local * sin_a + y_local * cos_a + center_y
        pts.append((int(x_rot), int(y_rot)))
        
    # 绘制线段，每隔一段画一段
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i+1], color, thickness)
        
    return img
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在图片上绘制椭圆")
    parser.add_argument("-i", "--input", required=True, help="输入图片路径")
    parser.add_argument("-o", "--output", default="output_ellipse.png", help="输出图片路径")
    # parser.add_argument("-c", "--center", type=int, nargs=2, required=True, help="中心坐标 x y")
    # parser.add_argument("-ax", "--axes", type=int, nargs=2, required=True, help="长轴和短轴半径 (radius_x, radius_y)")
    # parser.add_argument("-a", "--angle", type=float, default=0.0, help="旋转角度")
    parser.add_argument("--color", type=int, nargs=3, default=[0, 0, 255], help="颜色 B G R (默认红色: 0 0 255)")
    parser.add_argument("--thickness", type=int, default=15, help="线宽")

    args = parser.parse_args()

    image_path = args.input
    output_path = args.output
    
    # w :1920. h:1720
    
    w = 1920
    h = 1720
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        exit(0)

    
    #### LINE 1 ####
    (x, y) = 490,680
    (lr,sr) = 410,170
    angle = -60
    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (0,   0, 255))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (87, 139, 46))
    img = draw_dashed_ellipse(img, (x+2*w, y), (lr,sr), angle,  (87, 139, 46))

    (x, y) = 1450,840
    (lr,sr) = 460,180
    angle = 66
    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (0,   0, 255))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (87, 139, 46))
    img = draw_dashed_ellipse(img, (x+w*2, y), (lr,sr), angle,  (87, 139, 46))

    #### LINE 2 ####
    (x, y) = 1400,760+h
    (lr,sr) = 580,210
    angle = 67
    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (0,   0, 255))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (34, 139, 34))
    img = draw_dashed_ellipse(img, (x+2*w, y), (lr,sr), angle,  (34, 139, 34))
    
    #### LINE 3 ####
    (x, y) = 330,1240+2*h
    (lr,sr) = 250,180
    angle = -60
    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (0,   0, 255))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (34, 139, 34))
    img = draw_dashed_ellipse(img, (x+2*w, y), (lr,sr), angle,  (34, 139, 34))
    
    (x, y) = 1550,1230+2*h
    (lr,sr) = 350,240
    angle = 69
    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (0,   0, 255))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (34, 139, 34))
    img = draw_dashed_ellipse(img, (x+2*w, y), (lr,sr), angle,  (34, 139, 34))
    
        
    #### LINE 4 ####
    (x, y) = 390,1050+3*h
    (lr,sr) = 370,190
    angle = -61

    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (255,  0,  0))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (87, 139, 46))
    img = draw_dashed_ellipse(img, (x+2*w, y), (lr,sr), angle,  (87, 139, 46))

    (x, y) = 1480,1040+3*h
    (lr,sr) = 520,220
    angle = 75
    img = draw_dashed_ellipse(img, (x, y),     (lr,sr), angle,  (255,  0,  0))
    img = draw_dashed_ellipse(img, (x+w, y),   (lr,sr), angle,  (87, 139, 46))
    img = draw_dashed_ellipse(img, (x+w*2, y), (lr,sr), angle,  (87, 139, 46))
    
        # 保存图片
    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")