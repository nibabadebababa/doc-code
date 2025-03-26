import glob
import json
import os
import random
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import io

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from model.llava.constants import (
    TAMPER_QUESTION_LIST, NOTAMPER_ANSWER_LIST, ANSWER_START,
    ONLY_ANSWER_LIST, MULTI_ANSWER_LIST, ONLY_ANSWER_SEG_LIST,MULTI_ANSWER_SEG_LIST,
    ANSWER_LIST_END, ANSWER_LIST_DELAY
)
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,VISION_START_TOKEN,VISION_END_TOKEN,LLAVA_IMAGE_TOKEN

class DocTamperDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([177.01686, 175.04828, 172.7625]).view(-1, 1, 1)#([123.675, 116.28, 103.53]).view(-1, 1, 1) now([177.01686, 175.04828, 172.7625])
    pixel_std = torch.Tensor([51.388493,52.128727,53.16032]).view(-1, 1, 1)#([58.395, 57.12, 57.375]) now([51.388493,52.128727,53.16032])
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=5000,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 5,
        exclude_val=False,
        doctam_data="DocTamperV1-TrainingSet",
        conv_type="llava_v1",
        min_quality=50,   # 最低质量50 
        T=8192,           # 衰减系数
    ):
        self.exclude_val = exclude_val
        self.doctam_data = doctam_data
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.clip_image_processor = AutoProcessor.from_pretrained(vision_tower)
        self.conv_type = conv_type
        self.min_quality = min_quality
        self.T = T
        self.current_step = 0  # 添加步数跟踪

        # 创建共享变量来跟踪步数
        self.shared_step = mp.Value('i', 0)

        # image_dir = os.path.join(base_image_dir, "DocTamper", doctam_data, "DocTamperV1-TrainingSet_image")
        # mask_dir = os.path.join(base_image_dir, "DocTamper", doctam_data, "DocTamperV1-TrainingSet_label")
        # json_dir = os.path.join(base_image_dir, "DocTamper", doctam_data, "DocTamperV1-TrainingSet_json_allregion")
        image_dir = "/root/autodl-tmp/DocTamper/image"
        mask_dir = "/root/autodl-tmp/DocTamper/mask"
        json_dir = "/root/autodl-tmp/DocTamper/allregion"

        json_paths = glob.glob(os.path.join(json_dir, "*.json"))
        self.data = []

        for json_path in json_paths:
            json_name = os.path.basename(json_path)
            base_name = os.path.splitext(json_name)[0]
            # mask_path = os.path.join(mask_dir, base_name + ".png")
            mask_path = os.path.join(mask_dir, base_name.replace('image', 'mask') + ".jpg")

            with open(json_path, 'r') as f:
                info = json.load(f)
            image_name = info.get("id")
            image_path = os.path.join(image_dir, image_name)

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })

        print("Doctamper dataset number of samples:", len(self.data))

    def __len__(self):
        return self.samples_per_epoch

    # def enlarge_mask(self, mask_image, dilation_iterations=2):
    #     kernel = np.ones((3, 3), np.uint8)
    #     dilated_mask = cv2.dilate(mask_image, kernel, iterations=dilation_iterations)
    #     return dilated_mask

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def set_current_step(self, step):
        """更新当前训练步数"""
        with self.shared_step.get_lock():
            self.shared_step.value = step

    def get_dynamic_quality(self):
        """动态计算当前的压缩质量范围"""
        with self.shared_step.get_lock():
            current_step = self.shared_step.value
        
        # 根据当前步数动态计算B1
        B1 = 100 - current_step / self.T
        B1 = max(B1, self.min_quality)
        
        # 从(B1, 100)范围内随机选择质量因子
        quality = random.uniform(B1, 100)
        return quality

    def interpolate_jpeg_quality(self, image_pil, quality):
        """在两个整数质量值之间进行插值压缩"""
        quality_floor = int(np.floor(quality))
        quality_ceil = int(np.ceil(quality))
        
        if quality_floor == quality_ceil:
            # 使用 BytesIO 替代临时文件
            buffer = io.BytesIO()
            image_pil.save(buffer, format="JPEG", quality=quality_floor)
            buffer.seek(0)
            return Image.open(buffer)
        
        # 计算插值权重
        alpha = quality - quality_floor
        
        # 使用 BytesIO 进行压缩
        buffer1 = io.BytesIO()
        buffer2 = io.BytesIO()
        
        # 分别使用下限和上限质量进行压缩
        image_pil.save(buffer1, format="JPEG", quality=quality_floor)
        image_pil.save(buffer2, format="JPEG", quality=quality_ceil)
        
        buffer1.seek(0)
        buffer2.seek(0)
        
        img1 = Image.open(buffer1)
        img2 = Image.open(buffer2)
        
        # 将图像转换为numpy数组进行插值
        img1_array = np.array(img1).astype(float)
        img2_array = np.array(img2).astype(float)
        
        # 线性插值
        interpolated = img1_array * (1 - alpha) + img2_array * alpha
        
        # 清理缓冲区
        buffer1.close()
        buffer2.close()
        
        # 转回PIL图像
        return Image.fromarray(interpolated.astype(np.uint8))

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self.data) - 1)
        sample = self.data[random_idx]

        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        json_path = sample["json_path"]

        # 读取图像和掩码
        image = Image.open(image_path).convert('RGB')
        mask_image = cv2.imread(mask_path)
        if mask_image is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        # 处理掩码 - 直接从 mask_image 提取掩码
        hsv_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)
        mask_binary = cv2.inRange(hsv_mask, np.array([20, 100, 100]), np.array([30, 255, 255]))
        final_mask = (mask_binary > 0).astype(np.uint8) * 255

        # 读取 JSON 文件获取中心点和边界框信息
        with open(json_path, 'r') as f:
            info = json.load(f)
        
        # 获取原始图像尺寸
        original_height, original_width = mask_binary.shape[:2]
        
        # 存储中心点和边界框
        # centers = []
        boxes = []
        
        # 从 JSON 中提取区域信息
        for region in info.get("regions", []):
            # 获取中心点
            # center = region["center"]
            # centers.append((center["x"], center["y"]))
            
            # 获取边界框
            bbox = region["bbox"]
            top_left = bbox["top_left"]
            bottom_right = bbox["bottom_right"]
            boxes.append({
                "top_left": (top_left["x"], top_left["y"]),
                "bottom_right": (bottom_right["x"], bottom_right["y"])
            })

        # 将 final_mask 转换为张量
        if final_mask.sum() > 0:
            masks = torch.from_numpy(final_mask).float().unsqueeze(0) / 255.0
        else:
            masks = torch.zeros((1, mask_binary.shape[0], mask_binary.shape[1]), dtype=torch.float32)

        # 图像压缩和预处理
        quality = random.randint(75, 100)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        image = Image.open(buffer)
        
        # 转换为 numpy 数组
        image = np.array(image)
        
        
        resize = image.shape[:2]
        
        # 计算缩放比例
        scale_x = resize[1] / original_width
        scale_y = resize[0] / original_height
        
        # 缩放中心点和边界框
        # scaled_centers = []
        scaled_boxes = []
        # for center, box in zip(centers, boxes):
        for box in boxes:
            # scaled_center = {
            #     "x": int(center[0] * scale_x),
            #     "y": int(center[1] * scale_y)
            # }
            # scaled_centers.append(scaled_center)
            
            # 计算缩放后的边界框坐标
            scaled_box = [
                int(box["top_left"][0] * scale_x),      # x1
                int(box["top_left"][1] * scale_y),      # y1
                int(box["bottom_right"][0] * scale_x),  # x2
                int(box["bottom_right"][1] * scale_y)   # y2
            ]
            scaled_boxes.append(scaled_box)

        # 生成对话
        question = random.choice(TAMPER_QUESTION_LIST)
        questions = [question]
        answer_parts = []
        tamper = info.get("tamper")
        
        if tamper is None:
            answer = random.choice(NOTAMPER_ANSWER_LIST)
        else:
            answer_start = random.choice(ANSWER_START)
            answer_parts.append(answer_start)

            if tamper == "only":
                # center = scaled_centers[0]
                only_answer = random.choice(ONLY_ANSWER_SEG_LIST).format(i=1)
                answer_parts.append(only_answer)
                answer_parts.append(random.choice(ANSWER_LIST_END))
            elif tamper == "multi":
                for i, center in enumerate(scaled_boxes):
                    multi_answer = random.choice(MULTI_ANSWER_SEG_LIST).format(
                        order={0: "first", 1: "second", 2: "third"}.get(i, f"{i+1}th"),
                        i=i+1
                    )
                    answer_parts.append(multi_answer)
                    if i < len(scaled_boxes) - 1:
                        answer_parts.append(random.choice(ANSWER_LIST_DELAY))
                    else:
                        answer_parts.append(random.choice(ANSWER_LIST_END))
            answer = " ".join(answer_parts)

        answers = [answer]
        # Qwen chat template
        # message = [{"role": "user", "content": [
        #         {"type": "image", "image": image_path},
        #         {"type": "text", "text": questions[0]}
        #     ]},
        #         {"role": "assistant", "content": [
        #         {"type": "text", "text": answers[0]}
        #     ]}]
        # image_inputs, video_inputs = process_vision_info(message)
        
        # llava chat template
        message = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": questions[0]}
            ]},
                {"role": "assistant", "content": [
                {"type": "text", "text": answers[0]}
            ]}]
        image_inputs = Image.open(image_path)
        conversation = self.clip_image_processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            )
        inputs = self.clip_image_processor(
                text=[conversation],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        conversations = []
        

        conversations.append(conversation)
        image_clip = inputs["pixel_values"][0]
        # image_grid_thw = inputs["image_grid_thw"]
        image_grid_thw = inputs["image_sizes"]
        
        
        label = torch.ones(mask_binary.shape, dtype=torch.float32) * self.ignore_label
        # inference = True

        # 转换中心点和边界框为张量格式
        if len(scaled_boxes) > 0:
            # centers_tensor = torch.tensor([[c["x"], c["y"]] for c in scaled_centers], dtype=torch.float32)
            boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)  # shape: [N, 4]
        else:
            # centers_tensor = torch.empty((0, 2), dtype=torch.float32)
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        return (
            image_path,
            image_clip,
            image_grid_thw,
            conversations,
            masks,
            label,
            resize,
            questions,
            # centers_tensor,
            boxes_tensor,  # 现在返回张量格式的边界框
        )

# 以下代码保持不变

class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=5000,
        precision: str = "fp32",
        image_size: int = 1024,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="DocTamper",
        sample_rate=[1],
        doctam_data="DocTamperV1-TrainingSet",
        conv_type="llava_v1",  # 新增参数，传递给子数据集
        min_quality=75,  # 最低质量75
        T=8192,         # 衰减系数
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = self.dataset.split("||")

        self.all_datasets = []
        # 初始化 DocTamperDataset
        if "DocTamper" in self.datasets:
            self.all_datasets.append(
                DocTamperDataset(
                    base_image_dir,
                    tokenizer,
                    vision_tower,
                    samples_per_epoch,
                    precision,
                    image_size,
                    num_classes_per_sample,
                    exclude_val,
                    doctam_data,
                    conv_type=conv_type,  # 传递 conv_type 参数
                    min_quality=min_quality,
                    T=T,
                )
            )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 根据 sample_rate 选择一个子数据集
        dataset_ind = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        dataset = self.all_datasets[dataset_ind]

        # 获取子数据集的一个样本，使用传入的索引
        sample = dataset[idx % len(dataset)]

        inference = False
        return (*sample, inference)

# 测试代码

if __name__ == "__main__":
    import torch
    from transformers import Qwen2TokenizerFast
    
    # base_image_dir = "/home/victory/zr/TPLM-main/dataset"
    # tokenizer = None  # 请替换为实际的 tokenizer 实例
    # vision_tower = "/home/victory/zr/LISA-main/openai/clip-vit-large-patch14"
    base_image_dir = "/root/autodl-tmp/DocTamper/image"
    tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
    vision_tower = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"

    # 实例化 HybridDataset
    hybrid_dataset = HybridDataset(
        base_image_dir=base_image_dir,
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        samples_per_epoch=5000,
        precision="fp32",
        image_size=1024,
        num_classes_per_sample=3,
        exclude_val=False,
        dataset="DocTamper",
        sample_rate=[1],
        doctam_data="DocTamperV1-TrainingSet",
        conv_type="llava_v1",
        min_quality=75,  # 最低质量75
        T=8192,         # 衰减系数
    )

    # 获取 DocTamperDataset 实例
    if len(hybrid_dataset.all_datasets) > 0:
        doctam_dataset = hybrid_dataset.all_datasets[0]

        # 测试课程学习的压缩质量变化
        print("\n测试课程学习的压缩质量变化:")
        steps = [0, 1000, 2000, 4000, 6000, 8192]  # 测试不同训练步数
        qualities = []
        
        for step in steps:
            doctam_dataset.set_current_step(step)
            # 对每个步数采样多次获取平均质量
            step_qualities = []
            for _ in range(100):  # 每个步数采样100次
                quality = doctam_dataset.get_dynamic_quality()
                step_qualities.append(quality)
            avg_quality = sum(step_qualities) / len(step_qualities)
            qualities.append(avg_quality)
            print(f"Step {step}: Average compression quality = {avg_quality:.2f}")

        # 绘制压缩质量变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(steps, qualities, 'b-o')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Compression Quality')
        plt.title('Compression Quality Curriculum Learning')
        plt.grid(True)
        plt.savefig('compression_quality_curve.png')
        plt.close()

        # 测试不同训练步数下的图像压缩效果
        print("\n测试不同训练步数下的图像压缩效果:")
        test_steps = [0, 4000, 8192]  # 选择几个关键步数进行测试
        sample_idx = 0  # 使用第一个样本进行测试
        
        plt.figure(figsize=(15, 5))
        for i, step in enumerate(test_steps):
            doctam_dataset.set_current_step(step)
            sample = doctam_dataset[sample_idx]
            # 打印sample的内容
            # print(sample)
            
            # 获取压缩后的图像 - 修改这里
            # 解包返回的元组，image_tensor是第二个元素
            _, image_tensor, _, _, _, _, _, _, _, _ = sample
            
            # 显示压缩后的图像
            plt.subplot(1, 3, i+1)
            # 确保图像数据在合理范围内
            img_display = image_tensor.clone()
            if img_display.max() > 1:
                img_display = img_display / 255.0
            plt.imshow(img_display.permute(1, 2, 0).numpy())
            plt.title(f'Step {step}\nQuality ≈ {doctam_dataset.get_dynamic_quality()}')
            plt.axis('off')
        
        plt.suptitle('Image Compression at Different Training Steps')
        plt.tight_layout()
        plt.savefig('compression_examples.png')
        plt.close()

        # 测试基本功能
        print("\n测试基本数据加载功能:")
        for i in range(3):  # 测试前3个样本
            sample = doctam_dataset[i]
            (
                image_path,
                image_tensor,
                image_clip,
                image_grid_thw,
                conversations,
                masks,
                label,
                resize,
                questions,
                # centers_tensor,  # 解包 centers_tensor
                boxes_tensor,
            ) = sample

            print(f"\nSample {i + 1}:")
            print(f"Image Path: {image_path}")
            print(f"Image Tensor Shape: {image_tensor.shape}")
            print(f"Image Clip Tensor Shape: {image_clip.shape}")
            print(f"Conversations: {conversations}")
            print(f"Masks Shape: {masks.shape}")
            print(f"Label Shape: {label.shape}")
            print(f"Resize: {resize}")
            print(f"Questions: {questions}")
            # print(f"Centers Tensor Shape: {centers_tensor.shape}")
            # print(f"Centers Tensor: {centers_tensor}")
            print(f"box Tensor: {boxes_tensor}")
            print(f"Current compression quality: {doctam_dataset.get_dynamic_quality()}")
            print("-" * 50)
    else:
        print("No datasets found in hybrid_dataset.all_datasets")
