# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import glob
import os
import json
import cv2
import random
import numpy as np
import tempfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from pycocotools import mask
from transformers import CLIPImageProcessor
import io

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything import ResizeLongestSide

# 假设 `doctamper_dataset.py` 和 `rt_tam_dataset.py` 已经修改
# 使得它们的 `__getitem__` 方法返回 `centers_tensor`
from .doctamper_dataset import DocTamperDataset
from .rt_tam_dataset import RTManipuateDataset
from .receiptid_dataset import RecieptIDDataset
from .tsroie_dataset import TSROIEDataset
from model.llava.constants import (
    TAMPER_QUESTION_LIST, NOTAMPER_ANSWER_LIST, EXPLANATORY_TAMPER_QUESTION_LIST,
    ANSWER_START, ONLY_ANSWER_SEG_LIST,MULTI_ANSWER_SEG_LIST, ANSWER_CENTER,
    ANSWER_TAMPERTYPE, ANSWER_LIST_END, ANSWER_LIST_DELAY
)

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    centers_tensor_list = []
    boxes_tensor_list = []  # 新增边界框列表
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        centers_tensor,
        boxes_tensor,  # 新增 boxes
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks)
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        centers_tensor_list.append(centers_tensor)
        boxes_tensor_list.append(boxes_tensor)  # 添加边界框
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:  # conv_type == 'llava_llama_2'
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)

            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

    if inferences[0] == False:
        Np = images_clip_list[0].size(1) * images_clip_list[0].size(2) // 196
        truncate_len = tokenizer.model_max_length - (Np - 1)

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "centers_tensor_list": centers_tensor_list,
        "boxes_tensor_list": boxes_tensor_list,  # 新增边界框列表
        "inference": inferences[0],
        "conversation_list": conversation_list
    }


class MixedTrainingDataset(TorchDataset):
    pixel_mean = torch.Tensor([177.01686, 175.04828, 172.7625]).view(-1, 1, 1)#([123.675, 116.28, 103.53]).view(-1, 1, 1) now([177.01686, 175.04828, 172.7625])
    pixel_std = torch.Tensor([51.388493,52.128727,53.16032]).view(-1, 1, 1)#([58.395, 57.12, 57.375]) now([51.388493,52.128727,53.16032])
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 336,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="RealTextManipulation||DocTamper||T-SROIE||Receipt_ID",
        sample_rate=[1, 1],
        doctam_data="DocTamperV1-TrainingSet",
        rtm_data="RealTextManipulation|train",
        tsr_data="T-SROIE|train",
        rid_data="Receipt_ID|train",
        explanatory=0.1,
        no_sampling=False,
        min_quality=50,   # 最低质量50
        T=8192,           # 衰减系数
    ):
        self.exclude_val = exclude_val
        self.no_sampling = no_sampling
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.min_quality = min_quality
        self.T = T
        self.current_step = 0  # 训练步数跟踪
        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "RealTextManipulation":
                self.all_datasets.append(
                    RTManipuateDataset(
                    base_image_dir,
                    tokenizer,
                    vision_tower,
                    samples_per_epoch,
                    precision,
                    image_size,
                    num_classes_per_sample,
                    exclude_val,
                    rtm_data
                    )
                )
            elif dataset == "DocTamper":
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
                    min_quality=min_quality,   
                    T=T,           
                    )
                )
            elif dataset == "T-SROIE":
                self.all_datasets.append(
                    TSROIEDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        tsr_data
                    )
                )
            elif dataset == "Receipt_ID":
                self.all_datasets.append(
                    RecieptIDDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        rid_data,
                    )
                )
                

    def __len__(self):
            return self.samples_per_epoch
        
    def set_current_step(self, step):
        """将步数传递给所有子数据集"""
        self.current_step = step
        
        # 直接更新所有数据集的步数
        for dataset in self.all_datasets:
            if hasattr(dataset, 'set_current_step'):
                dataset.set_current_step(step)

    def __getitem__(self, idx):
        ind = np.random.choice(len(self.all_datasets), p=self.sample_rate)
        dataset = self.all_datasets[ind]
        sample = dataset[idx % len(dataset)]
        return (*sample, False)



class ValDataset(TorchDataset):
    pixel_mean = torch.Tensor([177.01686, 175.04828, 172.7625]).view(-1, 1, 1)#([123.675, 116.28, 103.53]).view(-1, 1, 1) now([177.01686, 175.04828, 172.7625])
    pixel_std = torch.Tensor([51.388493,52.128727,53.16032]).view(-1, 1, 1)#([58.395, 57.12, 57.375]) now([51.388493,52.128727,53.16032])
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024
    ):
        self.base_image_dir = base_image_dir
        val, splits = val_dataset.split("|")
        if val == "RealTextManipulation":
            image_dir = os.path.join(base_image_dir, val, splits, "JPEGImages")
            mask_dir = os.path.join(base_image_dir,  val, splits, "SegmentationClass")
            json_dir = os.path.join(base_image_dir,  val, splits, "SegmentJSON_allregion")
            # 获取所有 JSON 文件名（不带扩展名）
            json_files = [os.path.splitext(f)[0] for f in os.listdir(json_dir) if f.endswith('.json')]
            json_set = set(json_files)

            self.data = []

            # 获取所有图像路径
            image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                base_name = os.path.splitext(image_name)[0]
                mask_name = base_name + ".png"
                json_name = base_name + ".json"

                mask_path = os.path.join(mask_dir, mask_name)
                json_path = os.path.join(json_dir, json_name)

                # 检查是否存在对应的 JSON 文件
                if base_name not in json_set:
                    continue  # 跳过没有对应 JSON 文件的图像

                # 检查掩码文件是否存在
                if not os.path.exists(mask_path):
                    continue

                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })
            self.val = val
        elif val == "DocTamper":
            if splits == "DocTamperV1-TestingSet":
                image_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-TestingSet_image")
                mask_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-TestingSet_label")
                json_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-TestingSet_json_allregion")
            elif splits == "DocTamperV1-SCD":
                image_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-SCD_image")
                mask_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-SCD_label")
                json_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-SCD_json_allregion")
            elif splits == "DocTamperV1-FCD":
                image_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-FCD_image")
                mask_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-FCD_label")
                json_dir = os.path.join(base_image_dir, val, splits, "DocTamperV1-FCD_json_allregion")
        
            json_paths = glob.glob(os.path.join(json_dir, "*.json"))
            print(f"Found {len(json_paths)} JSON files in {json_dir}")

            self.data = []

            for json_path in json_paths:
                json_name = os.path.basename(json_path)
                base_name = os.path.splitext(json_name)[0]  # 获取 JSON 文件名（不含扩展名），如 'label_0'

                # 构建掩码文件名，掩码文件名与 JSON 文件名相同，只是扩展名为 .png
                mask_name = base_name + ".png"
                mask_path = os.path.join(mask_dir, mask_name)

                # 读取 JSON 文件，获取图像文件名
                with open(json_path, 'r') as f:
                    info = json.load(f)

                image_name = info.get("id")
                if not image_name:
                    print(f"Image ID not found in JSON file: {json_path}")
                    continue

                image_path = os.path.join(image_dir, image_name)

                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue

                # 检查掩码文件是否存在
                if not os.path.exists(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })
            self.val = val
        # Handle CERTD, IDCD, PSCD datasets with different processing
        elif val == "CERTD":
            image_dir = os.path.join(base_image_dir, val, "img")
            mask_dir = os.path.join(base_image_dir, val, "label")
            json_dir = os.path.join(base_image_dir, val, "json_allregion")

            json_paths = glob.glob(os.path.join(json_dir, "*.json"))
            print(f"Found {len(json_paths)} JSON files in {json_dir}")

            self.data = []
            for json_path in json_paths:
                json_name = os.path.basename(json_path)
                base_name = os.path.splitext(json_name)[0]

                mask_name = base_name + ".png"
                mask_path = os.path.join(mask_dir, mask_name)

                image_name = base_name + ".jpg"  # For CERTD, image and mask share the same base name but different extensions
                image_path = os.path.join(image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })
            self.val = val
        elif val =="T-SROIE":#, "PSCD"
            image_dir = os.path.join(base_image_dir, val, "image", splits)
            mask_dir = os.path.join(base_image_dir, val, "label", splits)
            json_dir = os.path.join(base_image_dir, val, "json_allregion", splits)

            json_paths = glob.glob(os.path.join(json_dir, "*.json"))
            print(f"Found {len(json_paths)} JSON files in {json_dir}")

            self.data = []
            for json_path in json_paths:
                json_name = os.path.basename(json_path)
                base_name = os.path.splitext(json_name)[0]

                mask_name = base_name + ".png"
                mask_path = os.path.join(mask_dir, mask_name)

                # 读取 JSON 文件，获取图像文件名
                with open(json_path, 'r') as f:
                    info = json.load(f)

                image_name = info.get("id")
                if not image_name:
                    print(f"Image ID not found in JSON file: {json_path}")
                    continue

                image_path = os.path.join(image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })
            self.val = val
        elif val =="IDCD":#, "PSCD"
            image_dir = os.path.join(base_image_dir, val, "image")
            mask_dir = os.path.join(base_image_dir, val, "label")
            json_dir = os.path.join(base_image_dir, val, "json_allregion")

            json_paths = glob.glob(os.path.join(json_dir, "*.json"))
            print(f"Found {len(json_paths)} JSON files in {json_dir}")

            self.data = []
            for json_path in json_paths:
                json_name = os.path.basename(json_path)
                base_name = os.path.splitext(json_name)[0]

                mask_name = base_name + ".png"
                mask_path = os.path.join(mask_dir, mask_name)

                # 读取 JSON 文件，获取图像文件名
                with open(json_path, 'r') as f:
                    info = json.load(f)

                image_name = info.get("id")
                if not image_name:
                    print(f"Image ID not found in JSON file: {json_path}")
                    continue

                image_path = os.path.join(image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })
            self.val = val
        elif val =="PSCD":
            image_dir = os.path.join(base_image_dir, val, "img")
            mask_dir = os.path.join(base_image_dir, val, "label")
            json_dir = os.path.join(base_image_dir, val, "json_allregion")

            json_paths = glob.glob(os.path.join(json_dir, "*.json"))
            print(f"Found {len(json_paths)} JSON files in {json_dir}")

            self.data = []
            for json_path in json_paths:
                json_name = os.path.basename(json_path)
                base_name = os.path.splitext(json_name)[0]

                mask_name = base_name + ".jpg"
                mask_path = os.path.join(mask_dir, mask_name)

                # 读取 JSON 文件，获取图像文件名
                with open(json_path, 'r') as f:
                    info = json.load(f)

                image_name = info.get("id")
                if not image_name:
                    print(f"Image ID not found in JSON file: {json_path}")
                    continue

                image_path = os.path.join(image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Mask file not found: {mask_path}")
                    continue

                self.data.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "json_path": json_path
                })
            self.val = val
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    
    def __len__(self):
        return len(self.data)

    # def enlarge_mask(self, mask_image, dilation_iterations=2):
    #     kernel = np.ones((3, 3), np.uint8)
    #     dilated_mask = cv2.dilate(mask_image, kernel, iterations=dilation_iterations)
    #     return dilated_mask

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        json_path = sample["json_path"]

        # 读取图像和掩码
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        if self.val in ["RealTextManipulation","CERTD","PSCD","T-SROIE"]:
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
            
            _, mask_binary = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
        elif self.val == "IDCD":
            # IDCD 的掩码处理方式
            mask_image = cv2.imread(mask_path)
            if mask_image is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
            # 转换为 HSV 颜色空间
            hsv_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)
            # 创建掩码：非黑色区域为前景
            mask_binary = cv2.inRange(hsv_mask, np.array([0, 0, 1]), np.array([180, 255, 255]))
            mask_binary = (mask_binary > 0).astype(np.uint8) * 255
        elif self.val == "DocTamper":
            mask_image = cv2.imread(mask_path)
            if mask_image is None:
                raise ValueError(f"Cannot read mask: {mask_path}")
            hsv_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)
            mask_binary = cv2.inRange(hsv_mask, np.array([20, 100, 100]), np.array([30, 255, 255]))
            mask_binary = (mask_binary > 0).astype(np.uint8) * 255
        
        # 处理图像
        if self.val=="DocTamper":
            image_pil = Image.fromarray(image)
            quality = random.randint(75, 100)
            buffer = io.BytesIO()
            image_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image_pil = Image.open(buffer)
            
            compressed_image = image_pil
            image = self.transform.apply_image(np.array(image_pil))
            image_clip = self.clip_image_processor.preprocess(np.array(compressed_image), return_tensors="pt")["pixel_values"][0]
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image = self.transform.apply_image(image)

        resize = image.shape[:2]
        image_tensor = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # 处理掩码
        if mask_binary.sum() > 0:
            masks = torch.from_numpy(mask_binary).float().unsqueeze(0) / 255.0
        else:
            masks = torch.zeros((1, mask_binary.shape[0], mask_binary.shape[1]), dtype=torch.float32)

        # 读取 JSON 文件获取中心点和边界框信息
        try:
            with open(json_path, 'r') as f:
                info = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read JSON file {json_path}: {e}")
            info = {"regions": []}

        # 获取原始图像尺寸
        original_height, original_width = mask_binary.shape[:2]

        # 存储中心点和边界框
        centers = []
        boxes = []

        # 从 JSON 中提取区域信息
        for region in info.get("regions", []):
            try:
                # 获取中心点
                center = region["center"]
                centers.append((center["x"], center["y"]))

                # 获取边界框
                bbox = region["bbox"]
                top_left = bbox["top_left"]
                bottom_right = bbox["bottom_right"]
                boxes.append({
                    "top_left": (top_left["x"], top_left["y"]),
                    "bottom_right": (bottom_right["x"], bottom_right["y"])
                })
            except KeyError as e:
                print(f"Warning: Missing key in region data: {e}")
                continue

        # 计算缩放比例
        scale_x = resize[1] / original_width
        scale_y = resize[0] / original_height

        # 缩放中心点和边界框
        scaled_centers = []
        scaled_boxes = []
        for center, box in zip(centers, boxes):
            scaled_center = {
                "x": int(center[0] * scale_x),
                "y": int(center[1] * scale_y)
            }
            scaled_centers.append(scaled_center)

            scaled_box = [
                int(box["top_left"][0] * scale_x),      # x1
                int(box["top_left"][1] * scale_y),      # y1
                int(box["bottom_right"][0] * scale_x),  # x2
                int(box["bottom_right"][1] * scale_y)   # y2
            ]
            scaled_boxes.append(scaled_box)

        # 转换为张量
        if len(scaled_centers) > 0:
            centers_tensor = torch.tensor([[c["x"], c["y"]] for c in scaled_centers], dtype=torch.float32)
            boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
        else:
            centers_tensor = torch.empty((0, 2), dtype=torch.float32)
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

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
                center = scaled_centers[0]
                only_answer = random.choice(ONLY_ANSWER_SEG_LIST).format(i=1,center=f"({center['x']}, {center['y']})")
                answer_parts.append(only_answer)
                answer_parts.append(random.choice(ANSWER_LIST_END))
            elif tamper == "multi":
                for i, center in enumerate(scaled_centers):
                    multi_answer = random.choice(MULTI_ANSWER_SEG_LIST).format(
                        order={0: "first", 1: "second", 2: "third"}.get(i, f"{i+1}th"),
                        i=i+1,
                        center=f"({center['x']}, {center['y']})"
                    )
                    answer_parts.append(multi_answer)
                    if i < len(scaled_centers) - 1:
                        answer_parts.append(random.choice(ANSWER_LIST_DELAY))
                    else:
                        answer_parts.append(random.choice(ANSWER_LIST_END))
            answer = " ".join(answer_parts)

        answers = [answer]
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conv.append_message(roles["human"], questions[0])
        conv.append_message(roles["gpt"], answers[0])
        conv.prompt = conv.get_prompt()
        conversations.append(conv.prompt)

        label = torch.ones(mask_binary.shape, dtype=torch.float32) * self.ignore_label
        inference = True

        return (
            image_path,
            image_tensor,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            None,
            centers_tensor,
            boxes_tensor,
            inference
        )

if __name__ == "__main__":
    import torch

    base_image_dir = "/home/victory/zr/TPLM-main/dataset"
    tokenizer = None  # 请替换为实际的 tokenizer 实例
    vision_tower = "/home/victory/zr/LISA-main/openai/clip-vit-large-patch14"

    # 实例化 MixedTrainingDataset
    hybrid_dataset = MixedTrainingDataset(
        base_image_dir=base_image_dir,
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        samples_per_epoch=5000,
        precision="fp32",
        image_size=1024,
        num_classes_per_sample=3,
        exclude_val=False,
        dataset="RealTextManipulation||DocTamper",
        sample_rate=[1,1],
        rtm_data="RealTextManipulation|train",
        doctam_data="DocTamperV1-TrainingSet",
        # conv_type="llava_v1",  # 指定 conv_type 参数
    )

    # 获取 RTManipuateDataset 实例
    if len(hybrid_dataset.all_datasets) > 0:
        rtm_dataset = hybrid_dataset.all_datasets[0]

        # 获取前6个样本
        for i in range(6):
            sample = rtm_dataset[i]
            (
                image_path,
                image_tensor,
                image_clip,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_sents,
                centers_tensor,  # 解包 centers_tensor
                boxes_tensor,  # 解包 boxes_tensor
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
            print(f"Sampled Sentences: {sampled_sents}")
            print(f"Centers Tensor Shape: {centers_tensor.shape}")
            print(f"Centers Tensor: {centers_tensor}")
            print(f"Boxes Tensor Shape: {boxes_tensor.shape}")
            print(f"Boxes Tensor: {boxes_tensor}")
            # 打印掩码的形状
            print(f"Masks shape: {masks.shape}")  # 应该是 [N, H, W]
    else:
        print("No datasets found in hybrid_dataset.all_datasets")
