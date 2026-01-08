"""
图片合并工具 - 独立演示脚本

功能：
- 支持水平（左右）和垂直（上下）合并
- 可自定义图片间距
- 可设置四周边距（左、右、上、下）
- 自动居中对齐

exp. python merge_images.py test1_open_combined_.png test1_final_combined_.png -o test1_final.png --spacing 100
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont


# 新增：解析可缩放字体，确保 font_size 生效
def _resolve_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
	# 1) 用户传入字体
	if font_path and os.path.exists(font_path):
		try:
			return ImageFont.truetype(font_path, font_size)
		except Exception:
			pass
	# 2) 尝试常见系统字体
	common_paths = [
		# macOS
		"/System/Library/Fonts/PingFang.ttc",
		"/System/Library/Fonts/STHeiti Medium.ttc",
		"/System/Library/Fonts/Helvetica.ttc",
		# Linux
		"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
		"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
		# Windows
		"C:/Windows/Fonts/msyh.ttc",
		"C:/Windows/Fonts/simhei.ttf",
		"C:/Windows/Fonts/simsun.ttc",
		"C:/Windows/Fonts/arial.ttf",
	]
	for p in common_paths:
		if os.path.exists(p):
			try:
				return ImageFont.truetype(p, font_size)
			except Exception:
				continue
	# 3) 回退位图字体（字号不可调）
	return ImageFont.load_default()


def add_caption_below(
	image_path: str,
	out_path: str,
	text: str,
	font_path: str | None = None,
	font_size: int = 32,
	text_color=(0, 0, 0),
	bg_color=(255, 255, 255),
	offset_left: int = 0,
	offset_top: int = 8,
	padding_bottom: int = 8,
	line_spacing: int = 4,
):
	"""
	在图像下方增加文字标注并保存为新图片。
	参数：
	- image_path: 输入图片路径
	- out_path: 输出图片路径
	- text: 文本内容（支持多行，用 '\n' 分隔）
	- font_path: 字体路径（ttf/otf/ttc），不传则自动选择系统可缩放字体
	- font_size: 字号
	- text_color: 文本颜色 (R,G,B)
	- bg_color: 新增区域背景色 (R,G,B)
	- offset_left: 文本相对“原图最左侧”的水平偏移（像素）
	- offset_top: 文本相对“原图底部”的垂直偏移（像素）（即文本到原图底边的距离）
	- padding_bottom: 文本块底部额外留白（像素）
	- line_spacing: 行间距（像素）
	"""
	# 打开原图
	img = Image.open(image_path).convert("RGB")
	W, H = img.size

	# 字体
	font = _resolve_font(font_path, int(font_size))

	# 计算多行文本尺寸
	draw_tmp = ImageDraw.Draw(img)
	lines = str(text).split("\n")

	def measure(txt: str):
		try:
			bbox = draw_tmp.textbbox((0, 0), txt if txt else " ", font=font)
			return bbox[2] - bbox[0], bbox[3] - bbox[1]
		except Exception:
			return draw_tmp.textsize(txt if txt else " ", font=font)

	line_sizes = [measure(line) for line in lines]
	line_heights = [h for _, h in line_sizes]
	text_block_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)

	# 新画布高度：原图 + 上偏移 + 文本高 + 下内边距
	caption_h = max(0, int(offset_top)) + text_block_h + max(0, int(padding_bottom))
	new_H = H + caption_h

	# 生成新画布与绘制
	canvas = Image.new("RGB", (W, new_H), bg_color)
	canvas.paste(img, (0, 0))
	draw = ImageDraw.Draw(canvas)

	# 文本起始坐标
	x = max(0, int(offset_left))
	y = H + max(0, int(offset_top))

	for i, line in enumerate(lines):
		lw, lh = measure(line)
		draw.text((x, y), line, fill=text_color, font=font)
		y += lh + line_spacing

	# 保存
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	canvas.save(out_path)
	print(f"✓ 已保存带文字标注图片: {out_path}")


def combine_images(image_paths, out_path, direction='horizontal', 
                   spacing=0, left_margin=0, right_margin=0, 
                   top_margin=0, bottom_margin=0, bg_color=(255,255,255),
                   spacing_color=None):
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
        spacing_color: 间距区域颜色 (R,G,B)，不传则使用 bg_color
    """
    if not image_paths:
        raise ValueError("图片路径列表为空")
    
    # 间距颜色默认使用背景色
    if spacing_color is None:
        spacing_color = bg_color
    
    # 读取所有图片
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    if direction == 'horizontal':
        # 水平合并（左右）
        total_width = sum(img.width for img in images) + spacing * (len(images) - 1) + left_margin + right_margin
        max_height = max(img.height for img in images) + top_margin + bottom_margin
        
        canvas = Image.new("RGB", (total_width, max_height), bg_color)
        
        x_offset = left_margin
        for i, img in enumerate(images):
            # 垂直居中
            y_offset = top_margin + (max_height - top_margin - bottom_margin - img.height) // 2
            canvas.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
            # 在图片之间填充间距颜色
            if i < len(images) - 1 and spacing > 0:
                spacing_img = Image.new("RGB", (spacing, max_height), spacing_color)
                canvas.paste(spacing_img, (x_offset, 0))
                x_offset += spacing
            
    elif direction == 'vertical':
        # 垂直合并（上下）
        max_width = max(img.width for img in images) + left_margin + right_margin
        total_height = sum(img.height for img in images) + spacing * (len(images) - 1) + top_margin + bottom_margin
        
        canvas = Image.new("RGB", (max_width, total_height), spacing_color)
        
        y_offset = top_margin
        for i, img in enumerate(images):
            # 水平居中
            x_offset = left_margin + (max_width - left_margin - right_margin - img.width) // 2
            canvas.paste(img, (x_offset, y_offset))
            y_offset += img.height
            
            # 在图片之间填充间距颜色
            if i < len(images) - 1 and spacing > 0:
                spacing_img = Image.new("RGB", (max_width, spacing), spacing_color)
                canvas.paste(spacing_img, (0, y_offset))
                y_offset += spacing
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
    parser.add_argument('--spacing-color', type=str, default=None,
                       help='间距区域颜色 RGB，格式: R,G,B (默认: 与背景色相同)')
    parser.add_argument("-c", '--add-caption', type=str, default='',
                       help='合并后图片下方添加文字说明（可选）')
    
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
    
    # 解析间距颜色
    spacing_color = None
    if args.spacing_color:
        try:
            spacing_color = tuple(map(int, args.spacing_color.split(',')))
            if len(spacing_color) != 3 or any(c < 0 or c > 255 for c in spacing_color):
                raise ValueError
        except:
            print("错误: 间距颜色格式不正确，应为 R,G,B 格式（如 200,200,200）")
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
            bg_color=bg_color,
            spacing_color=spacing_color
        )
    except Exception as e:
        print(f"错误: {e}")

    if args.add_caption == "heatmap":
        font_size = 50
        add_caption = "(a) The original collected occlusion             (b) The predicted occlusion"
        print("\n添加图片下方文字说明...")
        add_caption_below(image_path=args.output, out_path=args.output,
        text=add_caption,
        font_path=None, font_size=font_size, text_color=(0, 0, 0), bg_color=(255, 255, 255), 
        offset_left=50, offset_top=0, padding_bottom= 8, line_spacing=4)
        
    elif args.add_caption == "final":
        font_size = 128
        add_caption = "                Original Input                                    After Optimization                                  Ground Truth"
        print("\n添加图片下方文字说明...")
        add_caption_below(image_path=args.output, out_path=args.output,
        text=add_caption,
        font_path=None, font_size=font_size, text_color=(0, 0, 0), bg_color=(255, 255, 255), 
        offset_left=80, offset_top=50, padding_bottom= 50, line_spacing=4)

if __name__ == "__main__":
    main()
