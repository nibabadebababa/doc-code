import cv2
import numpy as np
import json
import os

def find_yellow_regions_center(image_path, json_path, debug_image_folder):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}，跳过处理。")
        return

    # 将图像从 BGR 转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黄色的 HSV 范围
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_black = np.array([0, 0, 0])  # 黑色的最低范围
    upper_black = np.array([180, 255, 30])  # 黑色的最高范围

    # 创建黄色掩码
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 查找黄色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"图像 {image_path} 没有找到轮廓，跳过处理。")
        return

    regions = []
    tamper = None

    if len(contours) <= 5:
        # 处理5个及以下轮廓
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                region_info = {
                    "center": {"x": cX, "y": cY},
                    "bbox": {
                        "top_left": {"x": x, "y": y},
                        "bottom_right": {"x": x + w, "y": y + h}
                    }
                }
                regions.append(region_info)
    else:
        # 处理超过5个轮廓的情况
        centers = []
        valid_contours = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY])
                valid_contours.append(contour)

        if len(centers) == 0:
            print(f"图像 {image_path} 没有有效轮廓，跳过处理。")
            return

        centers_np = np.array(centers, dtype=np.float32)

        if len(centers_np) <= 5:
            # 有效中心点数量不足5，直接处理
            for contour in valid_contours:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                region_info = {
                    "center": {"x": cX, "y": cY},
                    "bbox": {
                        "top_left": {"x": x, "y": y},
                        "bottom_right": {"x": x + w, "y": y + h}
                    }
                }
                regions.append(region_info)
        else:
            # 使用K-means聚类分成5组
            K = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            attempts = 10
            _, labels, _ = cv2.kmeans(centers_np, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

            clusters = {i: [] for i in range(K)}
            for idx, label in enumerate(labels):
                cluster_id = label[0]
                clusters[cluster_id].append(valid_contours[idx])

            # 合并每个聚类的轮廓
            regions = []
            for cluster_id in range(K):
                cluster_contours = clusters[cluster_id]
                if not cluster_contours:
                    continue

                all_points = []
                for contour in cluster_contours:
                    points = contour.reshape(-1, 2)
                    all_points.extend(points.tolist())
                all_points = np.array(all_points, dtype=np.int32)

                if len(all_points) == 0:
                    continue

                x, y, w, h = cv2.boundingRect(all_points)
                cX = x + w // 2
                cY = y + h // 2

                region_info = {
                    "center": {"x": cX, "y": cY},
                    "bbox": {
                        "top_left": {"x": x, "y": y},
                        "bottom_right": {"x": x + w, "y": y + h}
                    }
                }
                regions.append(region_info)
            regions = regions[:5]  # 确保最多5个区域

    # 设置tamper类型
    if not regions:
        tamper = None
    else:
        tamper = "multi" if len(regions) > 1 else "only"

    # 绘制所有区域
    for region_info in regions:
        cX = region_info["center"]["x"]
        cY = region_info["center"]["y"]
        x = region_info["bbox"]["top_left"]["x"]
        y = region_info["bbox"]["top_left"]["y"]
        br_x = region_info["bbox"]["bottom_right"]["x"]
        br_y = region_info["bbox"]["bottom_right"]["y"]

        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(image, f"({cX}, {cY})", (cX + 10, cY),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(image, (x, y), (br_x, br_y), (255, 0, 0), 2)

    # 构建JSON数据
    image_name = os.path.basename(image_path)
    if 'cover' in image_name or 'inpaint' in image_name:
        tampertype = "Removal"
    elif 'splice' in image_name or 'insert' in image_name:
        tampertype = "Addition"
    elif 'cpmv' in image_name:
        tampertype = "Replacement"
    else:
        tampertype = "Unknown"

    image_name_modified = image_name.replace('label', 'image')
    json_data = {
        "id": image_name_modified,
        "regions": regions,
        "tamper": tamper,
        "Tampertype": tampertype
    }

    # 构建JSON数据并保存
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"区域信息已保存到 {json_path}")

    # 创建调试图像的文件名，并确保它被保存在特定的文件夹中
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_image_name = f"{file_name}_debug.jpg"
    debug_image_path = os.path.join(debug_image_folder, debug_image_name)

    cv2.imwrite(debug_image_path, image)
    print(f"调试图像已保存到 {debug_image_path}")


def process_folder(image_folder, output_folder, debug_image_folder):
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, file_name)
            json_file_name = os.path.splitext(file_name)[0] + '.json'
            json_path = os.path.join(output_folder, json_file_name)

            # 确保调试图像文件夹存在
            os.makedirs(debug_image_folder, exist_ok=True)

            find_yellow_regions_center(image_path, json_path, debug_image_folder)

# 使用示例
image_folder = '/home/victory/zr/TPLM-main/dataset/DocTamper/DocTamperV1-TrainingSet/DocTamperV1-TrainingSet_label'
output_folder = '/home/victory/zr/TPLM-main/dataset/DocTamper/DocTamperV1-TrainingSet/DocTamperV1-TrainingSet_json_allregion'
debug_image_folder = '/home/victory/zr/TPLM-main/dataset/DocTamper/DocTamperV1-TrainingSet/DocTamperV1-TrainingSet_debugimage'  # 新增：指定调试图像存储位置

os.makedirs(output_folder, exist_ok=True)
process_folder(image_folder, output_folder, debug_image_folder)