import os
from PIL import Image

# 指定包含 PNG 图片的目录
input_dir = '/root/autodl-tmp/T-SROIE/mask'
output_dir = '/root/autodl-tmp/T-SROIE/mask'

# 如果输出目录不存在，则创建它
os.makedirs(output_dir, exist_ok=True)

# 遍历目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  # 只处理 PNG 文件
        # 构建完整的文件路径
        png_path = os.path.join(input_dir, filename)
        jpg_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpg")
        
        try:
            # 打开 PNG 文件
            with Image.open(png_path) as img:
                # 转换为 RGB 模式并保存为 JPG
                img.convert('RGB').save(jpg_path, 'JPEG')
            print(f"Converted {filename} to JPG.")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("Conversion completed!")
