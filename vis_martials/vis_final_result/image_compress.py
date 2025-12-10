#!/usr/bin/env python3
"""
img_compress.py
批量图像压缩工具（Pillow 版）
用法：
    python img_compress.py 源目录 输出目录 -W 800 -q 35
    python ./image_compress.py -i view_front_case_final.png -o view_front_case_final-c.png -q 40 
"""
import argparse
import os
import sys
from pathlib import Path
from PIL import Image

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp'}

def compress_one(src: str, dst: str, max_side: int, quality: int):
    """单张图压缩"""
    try:
        with Image.open(src) as im:
            im = im.convert('RGB')  # 去透明通道
            # 等比缩放
            if max_side > 0:
                w, h = im.size
                if max(w, h) > max_side:
                    ratio = max_side / float(max(w, h))
                    new_size = (int(w * ratio), int(h * ratio))
                    im = im.resize(new_size, Image.LANCZOS)
            # 保存
            print(f"Saving compressed image to {dst}")
            im.save(dst, format='JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f'[ERROR] {src}: {e}', file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='批量图像压缩（缩放+JPEG）')
    parser.add_argument('-i', '--input',  type=str, help='含图片的文件夹')
    parser.add_argument('-o', '--output', type=str, help='输出文件夹')
    parser.add_argument('-w', '--width',  type=int, default=1280,
                        help='长边最大像素，0 表示不缩放（默认 1280）')
    parser.add_argument('-q', '--quality', type=int, default=75,
                        help='JPEG 质量 1-95，默认 75')
    args = parser.parse_args()

    compress_one(args.input, args.output, args.width, args.quality)

if __name__ == '__main__':
    main()
