import os
import json
import cv2
import numpy as np
from PIL import Image

def process_mask_image(image_path):
    """
    处理单张掩码图像，提取边界框信息并返回 JSON 格式的数据
    :param image_path: 掩码图像文件路径
    :return: JSON 格式的数据
    """
    # 读取掩码图像转为灰度图
    image = Image.open(image_path)
    mask = np.array(image)
    
    # 确保掩码是二值图像
    mask[mask != 0] = 255
    
    # 确保掩码是单通道灰度图像
    if len(mask.shape) == 3:  # 如果是三通道图像
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 使用cv2.findContours找到掩码图像中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = {
            "top_left": {"x": x, "y": y},
            "bottom_right": {"x": x + w, "y": y + h}
        }
        center = {
            "x": x + w // 2,
            "y": y + h // 2
        }
        regions.append({
            "center": center,
            "bbox": bbox
        })
    
    # 构建 JSON 数据
    json_data = {
        "id": os.path.splitext(os.path.basename(image_path))[0].replace("mask","image")+'.jpg',
        "regions": regions,
        "tamper": "multi",
        "Tampertype": "Unknown"
    }
    
    return json_data

def save_json_data(json_data, json_file_path):
    """
    将 JSON 数据保存到文件
    :param json_data: 要保存的 JSON 数据
    :param json_file_path: 输出 JSON 文件路径
    """
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

# 示例用法
folder_path = r'/root/autodl-tmp/T-SROIE/mask'  # 掩码图像文件夹路径
json_path= r'/root/autodl-tmp/T-SROIE/allregion'
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # 检查是否是文件
    if os.path.isfile(file_path):
        # 处理掩码图像
        json_data = process_mask_image(file_path)
        # 构建输出 JSON 文件路径
        json_file_path = os.path.join(json_path, os.path.splitext(filename)[0].replace("mask","image") + '.json')
        # 保存 JSON 数据
        save_json_data(json_data, json_file_path)
        print(f"JSON 数据已保存到 {json_file_path}")