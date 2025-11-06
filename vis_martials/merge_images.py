"""
图片合并工具 - 独立演示脚本

功能：
- 支持水平（左右）和垂直（上下）合并
- 可自定义图片间距
- 可设置四周边距（左、右、上、下）
- 自动居中对齐
"""

import os
import argparse
from PIL import Image


def combine_images(image_paths, out_path, direction='horizontal', 
                   spacing=0, left_margin=0, right_margin=0, 
                   top_margin=0, bottom_margin=0, bg_color=(255,255,255)):
    """
    合并多张图片到一张图中。
    
    Args:
        image_paths: 图片路径列表
        out_path: 输出路径
        direction: 合并方向 'horizontal' (水平左右) 或 'vertical' (垂直上下)
        spacing: 图片之间的间距（像素）
        left_margin: 左边距（像素）
        right_margin: 右边距（像素）
        top_margin: 上边距（像素）
        bottom_margin: 下边距（像素）
        bg_color: 背景填充色 (R,G,B)
    """
    if not image_paths:
        raise ValueError("图片路径列表为空")
    
    # 读取所有图片
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    if direction == 'horizontal':
        # 水平合并（左右）
        total_width = sum(img.width for img in images) + spacing * (len(images) - 1) + left_margin + right_margin
        max_height = max(img.height for img in images) + top_margin + bottom_margin
        
        canvas = Image.new("RGB", (total_width, max_height), bg_color)
        
        x_offset = left_margin
        for img in images:
            # 垂直居中
            y_offset = top_margin + (max_height - top_margin - bottom_margin - img.height) // 2
            canvas.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
            
    elif direction == 'vertical':
        # 垂直合并（上下）
        max_width = max(img.width for img in images) + left_margin + right_margin
        total_height = sum(img.height for img in images) + spacing * (len(images) - 1) + top_margin + bottom_margin
        
        canvas = Image.new("RGB", (max_width, total_height), bg_color)
        
        y_offset = top_margin
        for img in images:
            # 水平居中
            x_offset = left_margin + (max_width - left_margin - right_margin - img.width) // 2
            canvas.paste(img, (x_offset, y_offset))
            y_offset += img.height + spacing
    else:
        raise ValueError(f"不支持的合并方向: {direction}，请使用 'horizontal' 或 'vertical'")
    
    canvas.save(out_path)
    print(f"✓ 合并完成 ({direction}): {out_path}")
    print(f"  - 输出尺寸: {canvas.width}x{canvas.height}")
    print(f"  - 合并图片数: {len(images)}")


def main():
    parser = argparse.ArgumentParser(
        description="合并多张图片工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 水平合并三张图片，间距20px
  python combine_images_demo.py img1.png img2.png img3.png -o output.png -d horizontal -s 20
  
  # 垂直合并，带边距
  python combine_images_demo.py img1.png img2.png -o output.png -d vertical -s 15 --left 10 --right 10 --top 20 --bottom 20
        """
    )
    
    parser.add_argument('images', nargs='+', help='要合并的图片路径列表')
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')
    parser.add_argument('-d', '--direction', choices=['horizontal', 'vertical'], 
                       default='horizontal', help='合并方向 (默认: horizontal)')
    parser.add_argument('-s', '--spacing', type=int, default=0, 
                       help='图片之间的间距（像素，默认: 0）')
    parser.add_argument('--left', type=int, default=0, help='左边距（像素，默认: 0）')
    parser.add_argument('--right', type=int, default=0, help='右边距（像素，默认: 0）')
    parser.add_argument('--top', type=int, default=0, help='上边距（像素，默认: 0）')
    parser.add_argument('--bottom', type=int, default=0, help='下边距（像素，默认: 0）')
    parser.add_argument('--bg-color', type=str, default='255,255,255',
                       help='背景色 RGB，格式: R,G,B (默认: 255,255,255)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"错误: 文件不存在 - {img_path}")
            return
    
    # 解析背景色
    try:
        bg_color = tuple(map(int, args.bg_color.split(',')))
        if len(bg_color) != 3 or any(c < 0 or c > 255 for c in bg_color):
            raise ValueError
    except:
        print("错误: 背景色格式不正确，应为 R,G,B 格式（如 255,255,255）")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # 合并图片
    try:
        print(f"\n开始合并 {len(args.images)} 张图片...")
        print(f"方向: {args.direction}")
        print(f"间距: {args.spacing}px")
        print(f"边距: 左{args.left} 右{args.right} 上{args.top} 下{args.bottom}")
        print()
        
        combine_images(
            image_paths=args.images,
            out_path=args.output,
            direction=args.direction,
            spacing=args.spacing,
            left_margin=args.left,
            right_margin=args.right,
            top_margin=args.top,
            bottom_margin=args.bottom,
            bg_color=bg_color
        )
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
