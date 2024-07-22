from PIL import Image, ImageDraw

# 创建一个空的RGB图像，大小为256x256
image = Image.new("RGB", (256, 256))
# 使用ImageDraw模块进行绘制
draw = ImageDraw.Draw(image)

# 获取图像的大小
width, height = image.size

# 定义边框颜色和内部颜色
border_color = (255, 0, 20)
fill_color = (0, 255, 10)

# 定义边框宽度
border_width = 0

# 绘制边框
draw.rectangle([(0, 0), (width-1, height-1)], outline=border_color, width=border_width)

# 绘制内部区域
draw.rectangle([(border_width, border_width), (width-border_width-1, height-border_width-1)], fill=fill_color)

# 将图像保存为文件（可选步骤）
image.save("no_border_image.png")