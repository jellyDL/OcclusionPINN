from PIL import Image, ImageDraw, ImageFont
import os


def draw_vertial_line(img, x):
    center_x = x
    img_height = img.height
    dash_length = 30  # 虚线段长度
    gap_length = 15    # 虚线间隔长度
    line_color = (64, 64, 64)  # 深灰色
    line_width = 8

    y = 0
    while y < img_height:
        draw.line([(center_x, y), (center_x, min(y + dash_length, img_height))], 
                fill=line_color, width=line_width)
        y += dash_length + gap_length

def draw_text(draw, text, x, y, color=(64, 64, 64), font_size=64, margin=10):
    # 设置字体和大小 (优先使用系统无衬线字体)
    font = ImageFont.load_default()
    font_paths = [
        "Arial.ttf",  # Windows
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/System/Library/Fonts/PingFang.ttc",  # macOS 中文
    ]

    for fp in font_paths:
        try:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        except:
            continue

    position = (x, y)  # 左上角，留出边距
    # 在图片上绘制文字
    draw.text(position, text, font=font, fill=color)
    
    
if __name__ == "__main__":

    # 读取图片
    img = Image.open('mid_plane.png')
    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    
    base_x = img.width // 2 - 28
    base_y = img.height
    
    # 1. 在图片中间绘制一条垂直虚线
    draw_vertial_line(img, img.width // 2 - 28)
 
    # 2. 设置文字内容和位置
    draw_text(draw, "Mid Plane", img.width//2+20, img.height//2, font_size=64)
   
    # 3. 设置牙位
    # upper
    uxoff = [200,140,100,60,70,60]
    uyoff = [90,120,170,170,220,240]
    uxoffb = 150
    uyoffb = 20
    draw_text(draw, "11", base_x-uxoffb,
                uyoffb)
    draw_text(draw, "12", base_x-uxoffb-uxoff[0],
                uyoffb+uyoff[0])
    draw_text(draw, "13", base_x-uxoffb-uxoff[0]-uxoff[1],
                uyoffb+uyoff[0]+uyoff[1])
    draw_text(draw, "14", base_x-uxoffb-uxoff[0]-uxoff[1]-uxoff[2],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2])
    draw_text(draw, "15", base_x-uxoffb-uxoff[0]-uxoff[1]-uxoff[2]-uxoff[3],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2]+uyoff[3])
    draw_text(draw, "16", base_x-uxoffb-uxoff[0]-uxoff[1]-uxoff[2]-uxoff[3]-uxoff[4],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2]+uyoff[3]+uyoff[4])
    draw_text(draw, "17", base_x-uxoffb-uxoff[0]-uxoff[1]-uxoff[2]-uxoff[3]-uxoff[4]-uxoff[5],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2]+uyoff[3]+uyoff[4]+uyoff[5])
            
    uxoffb = 60
    uyoffb = 20
    draw_text(draw, "21", base_x+uxoffb,
                uyoffb)
    draw_text(draw, "22", base_x+uxoffb+uxoff[0],
                uyoffb+uyoff[0])
    draw_text(draw, "23", base_x+uxoffb+uxoff[0]+uxoff[1],
                uyoffb+uyoff[0]+uyoff[1])
    draw_text(draw, "24", base_x+uxoffb+uxoff[0]+uxoff[1]+uxoff[2],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2])
    draw_text(draw, "25", base_x+uxoffb+uxoff[0]+uxoff[1]+uxoff[2]+uxoff[3],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2]+uyoff[3])     
    draw_text(draw, "26", base_x+uxoffb+uxoff[0]+uxoff[1]+uxoff[2]+uxoff[3]+uxoff[4],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2]+uyoff[3]+uyoff[4])  
    draw_text(draw, "27", base_x+uxoffb+uxoff[0]+uxoff[1]+uxoff[2]+uxoff[3]+uxoff[4]+uxoff[5],
                uyoffb+uyoff[0]+uyoff[1]+uyoff[2]+uyoff[3]+uyoff[4]+uyoff[5])
    
    # lower
    lxoffb = 40
    lyoffb = 70
    lxoff = [150,200,100,100,80,60]
    lyoff = [30,130,180,140,240,260]
    
    draw_text(draw, "31", base_x+lxoffb,
                base_y-lyoffb)
    draw_text(draw, "32", base_x+lxoffb+lxoff[0],
                base_y-lyoffb-lyoff[0])
    draw_text(draw, "33", base_x+lxoffb+lxoff[0]+lxoff[1],
                base_y-lyoffb-lyoff[0]-lyoff[1])
    draw_text(draw, "34", base_x+lxoffb+lxoff[0]+lxoff[1]+lxoff[2],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2])     
    draw_text(draw, "35", base_x+lxoffb+lxoff[0]+lxoff[1]+lxoff[2]+lxoff[3],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2]-lyoff[3])     
    draw_text(draw, "36", base_x+lxoffb+lxoff[0]+lxoff[1]+lxoff[2]+lxoff[3]+lxoff[4],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2]-lyoff[3]-lyoff[4])
    draw_text(draw, "37", base_x+lxoffb+lxoff[0]+lxoff[1]+lxoff[2]+lxoff[3]+lxoff[4]+lxoff[5],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2]-lyoff[3]-lyoff[4]-lyoff[5])
    
    lxoffb = 100
    lyoffb = 70
    draw_text(draw, "41", base_x-lxoffb,
                base_y-lyoffb)
    draw_text(draw, "42", base_x-lxoffb-lxoff[0],
                base_y-lyoffb-lyoff[0])
    draw_text(draw, "43", base_x-lxoffb-lxoff[0]-lxoff[1],
                base_y-lyoffb-lyoff[0]-lyoff[1])
    draw_text(draw, "44", base_x-lxoffb-lxoff[0]-lxoff[1]-lxoff[2],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2])
    draw_text(draw, "45", base_x-lxoffb-lxoff[0]-lxoff[1]-lxoff[2]-lxoff[3],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2]-lyoff[3]) 
    draw_text(draw, "46", base_x-lxoffb-lxoff[0]-lxoff[1]-lxoff[2]-lxoff[3]-lxoff[4],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2]-lyoff[3]-lyoff[4])  
    draw_text(draw, "47", base_x-lxoffb-lxoff[0]-lxoff[1]-lxoff[2]-lxoff[3]-lxoff[4]-lxoff[5],
                base_y-lyoffb-lyoff[0]-lyoff[1]-lyoff[2]-lyoff[3]-lyoff[4]-lyoff[5])
    
    # 保存图片
    img.save('mid_plane_final.png')
    