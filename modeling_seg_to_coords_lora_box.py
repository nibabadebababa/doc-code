'''
Author: git config user.name && git config user.email
Date: 2025-03-13 14:36:12
LastEditors: git config user.name && git config user.email
LastEditTime: 2025-03-13 14:38:27
FilePath: \毕设\doc-codeedit\doc-code\modeling_seg_to_coords_lora_box.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from model.segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from losses import dice_loss_new, sigmoid_ce_loss_new, focal_loss, modified_dice_loss, focal_tversky_loss, sinkhorn_distance, combined_box_loss
import logging
# 导入距离损失函数（L2 Loss）
from torch.nn.functional import mse_loss

from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLModel,Qwen2VLPreTrainedModel,Qwen2_5_VLPreTrainedModel,Qwen2_5_VLModel

class LisaGSVAMetaModel:
    #TODO 这里属于自定义的设置，基本不用改，把下面创建ViT的部分修改下就行
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config
        # 如果配置中没有 train_mask_decoder 属性，初始化相关参数
        if not hasattr(self.config, "train_mask_decoder"):
            #self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.config.in_dim = kwargs["in_dim"]
            # self.init_seg_and_proj(self.config)#!
            #self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
        else:
            #self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
            self.init_seg_and_proj(self.config)

    def init_seg_and_proj(self, config):
        # 配置日志记录
        logger = logging.getLogger('InitSegAndProj')
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)


        ###self.model.visual_model是SAM
        # SAM 根据 segmentation_model_path 选择不同的 SAM 模型构建函数
        #TODO 
        # builder_sam = build_sam_vit_h if "sam_vit_h" in self.segmentation_model_path else \
        #     build_sam_vit_l if "sam_vit_l" in self.segmentation_model_path else build_sam_vit_b
        # # 构建 SAM 视觉模型
        # self.visual_model = builder_sam(self.segmentation_model_path)
        # logger.info(f"Built SAM visual model: {self.segmentation_model_path}")

        # 冻结视觉模型的所有参数
        # for param in self.visual_model.parameters():
        #     param.requires_grad = False  # 冻结视觉模型的所有参数
        # logger.info("Frozen all parameters of visual_model.")

        # 如果需要训练掩码解码器，则解冻其参数
        # if config.train_mask_decoder:
        #     self.visual_model.mask_decoder.train()  # 设置为训练模式
        #     for param in self.visual_model.mask_decoder.parameters():
        #         param.requires_grad = True  # 解冻掩码解码器的参数以进行训练
        #     logger.info("Unfroze parameters of mask_decoder and set it to train mode.")
        # else:
        #     logger.info("train_mask_decoder is False. mask_decoder remains frozen.")

        # if hasattr(self.visual_model, 'image_encoder'):
        #     for name, param in self.visual_model.image_encoder.named_parameters():
        #         if 'lora' in name.lower():
        #             param.requires_grad = True
        #             logger.info(f"Unfroze LoRA parameter: {name}")
        # else:
        #     logger.warning("visual_model.image_encoder not found. LoRA parameters not unfrozen.")

        # 初始化中心隐藏层
        in_dim = config.in_dim #lm层的in_festures
        out_dim = config.out_dim
        #### center_fc = [
        #     nn.Linear(in_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, out_dim),
        #     nn.Dropout(0.0),
        #     nn.Linear(out_dim, 2),
        #     nn.Sigmoid(),  # 确保输出在 [0,1] 之间
        # ]
        # center_fc = [
        #     nn.Linear(in_dim, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 64),
        #     nn.Dropout(0.0),
        #     nn.Linear(64, 2),
        #     nn.Sigmoid(),  # 确保输出在 [0,1] 之间
        # ]#单纯用center的时候才是这个
        #### self.center_hidden_fcs = nn.ModuleList([nn.Sequential(*center_fc)])
        # self.center_hidden_fcs.train()  # 设置为训练模式
        # for param in self.center_hidden_fcs.parameters():
        #     param.requires_grad = True  # 解冻文本隐藏层的参数以进行训练
        # logger.info("Initialized center_hidden_fcs and set them to train mode.")
        # 在初始化center_hidden_fcs后添加
        box_fc = [
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
            nn.Dropout(0.0),
            nn.Linear(out_dim, 4),
            nn.Sigmoid(),  # 输出归一化的坐标
        ]
        self.box_hidden_fcs = nn.ModuleList([nn.Sequential(*box_fc)])
        self.box_hidden_fcs.train()
        for param in self.box_hidden_fcs.parameters():
            param.requires_grad = True
        logger.info("Initialized box_hidden_fcs.")
        # # 打印可训练参数的数量和名称
        # trainable_params = [name for name, param in self.named_parameters() if param.requires_grad]
        # logger.info(f"Number of trainable parameters: {len(trainable_params)}")
        # logger.info(f"Trainable parameters: {trainable_params}")

class LisaGSVAModel(LisaGSVAMetaModel, Qwen2_5_VLModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        # 初始化配置中的一些参数
        self.config.use_cache = False
        # self.config.vision_tower = self.config.mm_vision_tower
        # self.config.mm_vision_select_feature = "patch"
        # self.config.image_aspect_ratio = "square"
        # self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        # self.config.mm_use_im_patch_token = False
        ##self.pot_token_idx = kwargs.get("pot_token_idx", 0)
        self.box_token_idx = kwargs.get("box_token_idx", 0)

# Qwen2_5_VLForConditionalGeneration,Qwen2VLModel,Qwen2VLPreTrainedModel,LlavaLlamaForCausalLM,LlavaLlamaModel
class LisaGSVAForCausalLM(Qwen2_5_VLForConditionalGeneration):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            # 如果配置中没有 train_mask_decoder 属性，初始化模型相关参数
            # config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            # config.mm_vision_tower = kwargs.get(
            #     "vision_tower", "openai/clip-vit-large-patch14-336"
            # )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            # self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            # self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            ###self.center_loss_weight = kwargs.pop("center_loss_weight", 1.0)  # 新增 center_loss 权重
            self.box_loss_weight = kwargs.pop("box_loss_weight", 1.0)  # 新增 box_loss 权重
           
        # 初始化分割和拒绝标记的索引
        ###self.pot_token_idx = kwargs.get("pot_token_idx", None)
        self.box_token_idx = kwargs.get("box_token_idx", None)
        # self.rej_token_idx = kwargs.pop("rej_token_idx")
        self.llm_tokenizer = kwargs.get("tokenizer", None)
        super().__init__(config, **kwargs)
          
        self.model = LisaGSVAModel(config, **kwargs)
        # 定义语言模型头，将隐藏层输出映射到词汇表大小
        # self.lm_head = nn.Linear(config.in_dim, config.vocab_size, bias=False)
        # self.Np = self.model.vision_tower.num_patches
        self.Np = 140 #!
        self.post_init()
        
    def get_model(self):
        return self.model
        
    
        # 定义中心点损失函数，使用 L2 损失
        ###self.center_loss_fn = nn.MSELoss(reduction='mean')

    ##SAM
    # def get_visual_embs(self, pixel_values: torch.FloatTensor):
    #     image_embeddings_list = []
    #     # 遍历每张图像，计算视觉嵌入
    #     for i in range(pixel_values.shape[0]):
    #         torch.cuda.empty_cache()
    #         image_embeddings = self.model.visual_model.image_encoder(
    #             pixel_values[i].unsqueeze(0)
    #         )
    #         image_embeddings_list.append(image_embeddings)
    #     torch.cuda.empty_cache()
    #     # 将嵌入列表拼接成一个张量
    #     image_embeddings = torch.cat(image_embeddings_list, 0)
    #     return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def pad_sequnce_and_stack(self, input_ids, attention_masks, labels):
        # 填充输入序列、注意力掩码和标签，使它们具有相同的长度
        input_ids = nn.utils.rnn.pad_sequence(input_ids, True, 0)
        attention_masks = nn.utils.rnn.pad_sequence(attention_masks, True, False)
        labels = nn.utils.rnn.pad_sequence(labels, True, IGNORE_INDEX)
        return input_ids, attention_masks, labels
    
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        input_ids: torch.LongTensor, # torch.Size([2, 109])
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        #masks_list: List[torch.FloatTensor],
        #label_list: List[torch.Tensor],  ##
        resize_list: List[tuple],
        ##centers_tensor_list: List[torch.Tensor],  # 新增的 centers_tensor_list
        boxes_tensor_list: List[torch.Tensor],  # 新增的 boxes_tensor_list
        inference: bool = False,
        # reeval: bool = False,
        **kwargs,
    ):
        device, dtype = images.device, images.dtype
        ##从SAM获取的image_embeddings
        #image_embeddings = self.get_visual_embs(images)
        # if self.pot_token_idx is not None:
        #     self.min_pot_token_idx = min(self.pot_token_idx)
        #     self.max_pot_token_idx = max(self.pot_token_idx)
        # else:
        #     self.min_pot_token_idx = None
        #     self.max_pot_token_idx = None
        if self.box_token_idx is not None:
            self.min_box_token_idx = min(self.box_token_idx)
            self.max_box_token_idx = max(self.box_token_idx)
        else:
            self.min_box_token_idx = None
            self.max_box_token_idx = None
        # 获取视觉嵌入
        #batch_size = image_embeddings.shape[0]
        batch_size=len(offset) - 1
        #assert batch_size == len(offset) - 1 #offset是0到batchsize进行顺序排列
        if inference: # Segmentation Eval
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            # 将 images_clip 扩展以匹配输入的长度
            # print(images_clip)
            # images_clip_unsqueezed = images_clip.unsqueeze(0)
            # images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            images_clip_extend = images_clip
            # images_clip_extend = images_clip_unsqueezed.expand(length, -1, -1).contiguous()
            output_hidden_states = []
            output_ids = []
             # 循环处理每个批次，这里 n_batch 为 1，所以只会执行一次循环
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                # llava调用
                #! bug is here
                output_i = super().forward(
                    input_ids=input_ids,
                    pixel_values=images_clip,
                    # images=images_clip_extend[: end_i - start_i],
                    # inputs_embeds=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks,
                    image_grid_thw = image_grid_thw,
                    output_hidden_states=True
                )
                # print(input_ids) #!
                output_hidden_states.append(output_i.hidden_states)
                for k in range(length):
                    pred_ids = input_ids[k].clone()
                    # print(pred_ids) #!
                    # img_idx = (pred_ids == IMAGE_TOKEN_INDEX).nonzero().item()
                    # # 用零填充图像标记位置（假设只有一个图像且位于序列的前部）
                    # pred_ids = torch.cat([pred_ids[0:img_idx], torch.zeros(self.Np, device=device, dtype=torch.int64), pred_ids[img_idx + 1:]], dim=0) # torch.Size([322])
                    image_token_count = (pred_ids != 0).sum().item() #!
                    # print(f"Number of IMAGE_TOKEN_INDEX: {image_token_count}") #!
                    
                    # 找到所有 IMAGE_TOKEN_INDEX 的位置
                    img_indices = (pred_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0] #!
                    
                    # 将找到的所有图像标记位置替换为 0
                    pred_ids[img_indices] = 0
                    output_ids.append(pred_ids)
                torch.cuda.empty_cache()
            
            output_hidden_states_list = []
            # print(type(output_hidden_states))
            # print(output_hidden_states) 
            #!
            output_hidden_states = [x[0] if isinstance(x, tuple) else x for x in output_hidden_states]

            # 检查输出的形状
            # for i, tensor in enumerate(output_hidden_states):
            #     print(f"Shape of tensor {i}: {tensor.shape}")
            #!
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)# torch.Size([1, 322, 4096])
            output_hidden_states_list.append(output_hidden_states_level)
            # 用重新评估后的隐藏状态替换之前的结果
            output_hidden_states = output_hidden_states_list
            output = None


        else:  # Training 训练模式
            images_clip_list = []
            # print(offset)
            for i in range(len(offset) - 1):  # offset marks each begin and end index for each images.
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            # VLM inference, obtain LLaVA output 调用父类 forward，获取 LLaVA 输出
            # print("Labels:\n")
            # print(labels)
            
            # print("Forward Inputs:")
            # print(f"input_ids shape: {input_ids.shape}")
            # print(f"images_clip shape: {images_clip.shape}")
            # print(f"attention_mask shape: {attention_masks.shape}")
            # print(f"image_grid_thw: {image_grid_thw}")
            # print(f"labels shape: {labels.shape}")
            stacked_image_grid_thw = torch.cat([image_grid_thw] * 2, dim=0)  # 在新维度上堆叠
            output = super().forward(
                input_ids=input_ids,
                pixel_values=images_clip,
                attention_mask=attention_masks,
                image_grid_thw = stacked_image_grid_thw,
                labels=labels,
                output_hidden_states=True
            )
            output_hidden_states = output.hidden_states
        # 提取最后一层的隐藏状态
        hidden_states = []
        #assert len(self.model.center_hidden_fcs) == 1
        hidden_states.append(output_hidden_states[-1])
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        # 创建 pot_token 和 box_token 的掩码
        # if self.min_pot_token_idx is not None and self.max_pot_token_idx is not None:
        #     pot_token_mask = (input_ids[:, 1:] >= self.min_pot_token_idx) & (input_ids[:, 1:] <= self.max_pot_token_idx)
        # else:
        #     pot_token_mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)

        if self.min_box_token_idx is not None and self.max_box_token_idx is not None:
            box_token_mask = (input_ids[:, 1:] >= self.min_box_token_idx) & (input_ids[:, 1:] <= self.max_box_token_idx)
        else:
            box_token_mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)

        # 处理掩码
        # pot_token_mask = torch.cat([
        #     pot_token_mask,
        #     torch.zeros(pot_token_mask.shape[0], 1, dtype=torch.bool, device=device)
        # ], dim=1)
        # pot_token_mask = torch.cat([
        #     torch.zeros(pot_token_mask.shape[0], self.Np - 1, dtype=torch.bool, device=device),
        #     pot_token_mask
        # ], dim=1)

        box_token_mask = torch.cat([
            box_token_mask,
            torch.zeros(box_token_mask.shape[0], 1, dtype=torch.bool, device=device)
        ], dim=1)
        box_token_mask = torch.cat([
            torch.zeros(box_token_mask.shape[0], 1, dtype=torch.bool, device=device),
            box_token_mask
        ], dim=1)

        # 提取对应位置的特征并转换为坐标和框
        # pot_features = last_hidden_state[pot_token_mask]  # 提取点特征
        image_token_count = (input_ids != 0).sum().item()
        # box_token_mask = box_token_mask[:, :image_token_count]  # 限制为 image_token_count #!估计是动态编码的原因，这个也要不停变化
        seq_len = last_hidden_state.shape[1]  # 获取第二维长度 391
        box_token_mask = box_token_mask[:, :seq_len]  # 只取匹配的部分
        box_features = last_hidden_state[box_token_mask]  # 提取框特征

        # 通过各自的网络转换为坐标和框
        assert len(self.model.box_hidden_fcs) == 1
        # pred_points = self.model.center_hidden_fcs[0](pot_features)  # [num_pot_tokens, 2]
        pred_boxes_raw = self.model.box_hidden_fcs[0](box_features)  # [num_box_tokens, 4]

        # 确保框坐标的顺序（左上右下）
        x_coords = pred_boxes_raw[:, [0, 2]]
        x_sorted = torch.sort(x_coords, dim=1)[0]
        y_coords = pred_boxes_raw[:, [1, 3]]
        y_sorted = torch.sort(y_coords, dim=1)[0]
        pred_boxes = torch.stack([x_sorted[:,0], y_sorted[:,0], x_sorted[:,1], y_sorted[:,1]], dim=1)

        # 计算每个样本的 token 数量和偏移
        # pot_token_counts = pot_token_mask.int().sum(-1)
        # pot_token_offset = pot_token_counts.cumsum(-1)
        # pot_token_offset = torch.cat([torch.tensor([0], dtype=torch.int64, device=device), pot_token_offset], dim=0)

        box_token_counts = box_token_mask.int().sum(-1)
        box_token_offset = box_token_counts.cumsum(-1)
        box_token_offset = torch.cat([torch.tensor([0], dtype=torch.int64, device=device), box_token_offset], dim=0)

        # 按照偏移量提取预测的点和框
        # predicted_points_list = []
        predicted_boxes_list = []
        # predicted_points_list_resized = []
        predicted_boxes_list_resized = []

        for i in range(len(box_token_offset) - 1):
            # 处理点
            # start_i, end_i = pot_token_offset[i], pot_token_offset[i + 1]
            # points = pred_points[start_i:end_i]
            # predicted_points_list.append(points)

            # 处理框
            start_i, end_i = box_token_offset[i], box_token_offset[i + 1]
            boxes = pred_boxes[start_i:end_i]
            predicted_boxes_list.append(boxes)

            # 缩放到原始尺寸
            h, w = resize_list[i]
            # if points.shape[0] > 0:
            #     resized_points = points * torch.tensor([w, h], device=device, dtype=dtype)
            #     predicted_points_list_resized.append(resized_points)
            # else:
            #     predicted_points_list_resized.append(torch.empty((0, 2), device=device, dtype=dtype))

            if boxes.shape[0] > 0:
                resized_boxes = boxes * torch.tensor([w, h, w, h], device=device, dtype=dtype)
                predicted_boxes_list_resized.append(resized_boxes)
            else:
                predicted_boxes_list_resized.append(torch.empty((0, 4), device=device, dtype=dtype))

        # 创建 plabels_list，与 predicted_points_list 对应
        # plabels_list = []
        # for pred_points_resized in predicted_points_list_resized:
        #     if pred_points_resized.numel() == 0:
        #         plabels_list.append(None)  # 无预测点，标记为 None
        #         continue
        #     num_points = pred_points_resized.shape[0]
        #     plabels = torch.ones(1, num_points, dtype=torch.int, device=device)  # 标签全为 1，形状为 [1, N]
        #     plabels_list.append(plabels)


        #TODO 
        # 使用 predicted_points 作为中心点提示，调用 prompt_encoder
        # pred_masks = []
        # pred_ious = []
        # for i in range(len(predicted_boxes_list_resized)):
        #     if plabels_list[i] is None:
        #         # 没有预测点，生成全零掩码
        #         h, w = resize_list[i]
        #         combined_mask = torch.zeros(label_list[i].shape, dtype=torch.float32, device=device)
        #         combined_mask = combined_mask.unsqueeze(0)  # 在第0维添加批次维度
        #         pred_masks.append(combined_mask)  # [H, W]
        #         pred_ious.append(torch.tensor([0.0], device=device, dtype=dtype))  # 无效的 IOU
        #         continue
        #     pred_boxes_resized = predicted_boxes_list_resized[i]  # shape: [N_box, 4]
        #     gt_boxes = boxes_tensor_list[i] 
        #     if pred_boxes_resized.numel() > 0:
        #         # 给 prompt_encoder 的 boxes 参数传 [1, N_box, 4]
        #         boxes_for_sam = pred_boxes_resized.unsqueeze(0)  # 在batch维度上扩展
        #     else:
        #         boxes_for_sam = None
        #     if gt_boxes.numel() > 0:
        #         # 给 prompt_encoder 的 boxes 参数传 [1, N_box, 4]
        #         boxes_for_sam_gt = gt_boxes.unsqueeze(0)  # 在batch维度上扩展
        #     else:
        #         boxes_for_sam_gt = None
        #     #points = predicted_points_list_resized[i].unsqueeze(0)  # [1, N, 2]
        #     plabels = plabels_list[i]  # [1, N]
        #     sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
        #         points=None,#(points, plabels)
        #         boxes=boxes_for_sam,
        #         masks=None,
        #         text_embeds=None,  # 不使用文本嵌入
        #     )
        #     转换嵌入的类型以匹配 dtype
        #     sparse_embeddings = sparse_embeddings.to(dtype)
        #     low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
        #         image_embeddings=image_embeddings[i].unsqueeze(0),
        #         image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=False
        #     )
        #     pred_mask = self.model.visual_model.postprocess_masks(
        #         low_res_masks,
        #         input_size=resize_list[i],
        #         original_size=label_list[i].shape
        #     )
        #     pred_masks.append(pred_mask[:, 0])
        #     pred_ious.append(iou_predictions[:, 0])

        # 保存模型输出和真实掩码
        model_output = output
        #gt_masks = masks_list

        if inference:
            return {
                "pred_box":predicted_boxes_list_resized, # [B, N, 4] 
                "gt_box":boxes_tensor_list,
                "resize_list":resize_list,# [B, 2][0]为h，[1]为w
                "output_ids": output_ids
            }

        # 计算中心点损失 center_loss
        # center_loss = 0.0
        # num_centers = 0
        # for i in range(batch_size):
        #     pred_points_norm = predicted_points_list[i]  # [N_pred, 2]
        #     gt_centers = centers_tensor_list[i]  # [M,2], in pixel coordinates
        #     if gt_centers.numel() == 0 or pred_points_norm.numel() == 0:
        #         # 如果没有中心点，跳过计算
        #         continue
        #     # 归一化 gt_centers 到 [0,1]
        #     h, w = resize_list[i]
        #     gt_points_norm = gt_centers.to(dtype) / torch.tensor([w, h], device=device, dtype=dtype)  # [M,2]
        #     # 如果预测的中心点数量和真实中心点数量不同，需要进行匹配
        #     min_num = min(pred_points_norm.shape[0], gt_points_norm.shape[0])
        #     pred_points_norm = pred_points_norm[:min_num]
        #     gt_points_norm = gt_points_norm[:min_num]
        #     # 计算 MSE
        #     distance = sinkhorn_distance(pred_points_norm, gt_points_norm, epsilon=0.1, max_iters=100, tol=1e-9)
        #     center_loss += distance
        #     num_centers += 1
            
        # if num_centers > 0:
        #     center_loss = self.center_loss_weight * center_loss / num_centers
        # else:
        #     center_loss = torch.tensor(0.0, device=device, dtype=dtype)

        ######确保训练seg以及确保与mask的数量对齐##############
        # if not inference:
        #     assert len(gt_masks) == len(pred_masks)
        # # 确保每个预测掩码和对应的真实掩码的shape相同
        # for b in range(batch_size):
        #     pm = pred_masks[b]
        #     gm = gt_masks[b]
        #     assert pm.shape == gm.shape, f"b_idx: {b}, pm.shape: {pm.shape}, gm.shape: {gm.shape}"

        # 计算损失
        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        # loss = 0
        # mask_bce_loss = torch.tensor(0.0, device=device, dtype=dtype)
        # mask_dice_loss = torch.tensor(0.0, device=device, dtype=dtype)
        # num_masks = 0
        # 遍历每个批次，计算 BCE 和 DICE 损失
        # for batch_idx in range(len(pred_masks)):
        #     if batch_idx >= len(gt_masks):
        #         raise ValueError(f"gt_masks are not in good shape with b_idx={batch_idx} >= len(gt_masks)={len(gt_masks)}, also len(preds)={len(pred_masks)}.")
        #     gt_mask = gt_masks[batch_idx]
        #     pred_mask = pred_masks[batch_idx]
        #     if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        #         # 都没有掩码，跳过损失计算
        #         continue
        #     if (
        #         gt_mask.shape[0] != pred_mask.shape[0]
        #     ):
        #         i0, i1 = input_ids[0], input_ids[1]
        #         i0, i1 = i0[i0 != IMAGE_TOKEN_INDEX], i1[i1 != IMAGE_TOKEN_INDEX]
        #         print(f"gt: {gt_mask.shape}, pred: {pred_mask.shape}\n" + \
        #             f"Prompt0: {self.llm_tokenizer.decode(i0)}\n" + \
        #             f"Prompt1: {self.llm_tokenizer.decode(i1)}\n" + \
        #             f"GT_MASK sum :{gt_mask.sum(dim=(1, 2))}\n"
        #         )
        #         raise RuntimeError("Found it!")
        #     mask_bce_loss += (
        #         focal_tversky_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])  # sigmoid_ce_loss
        #         * gt_mask.shape[0]
        #     )
        #     mask_dice_loss += (
        #         modified_dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])  # dice_loss
        #         * gt_mask.shape[0]
        #     )
        #     num_masks += gt_mask.shape[0]
        # 计算平均 BCE 和 DICE 损失
        # mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        # mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        # mask_loss = mask_bce_loss + mask_dice_loss
        # 计算框损失
        box_loss = 0.0
        num_boxes = 0
        for i in range(batch_size):
            pred_boxes = predicted_boxes_list[i]  # [N_pred, 4]
            gt_boxes = boxes_tensor_list[i]  # [M, 4]
            
            if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
                continue
            
            # 归一化框坐标
            h, w = resize_list[i]
            gt_boxes_norm = gt_boxes.to(dtype) / torch.tensor([w, h, w, h], device=device, dtype=dtype)
            
            # 确保预测和真实框数量匹配
            min_num = min(pred_boxes.shape[0], gt_boxes_norm.shape[0])
            pred_boxes = pred_boxes[:min_num]
            gt_boxes_norm = gt_boxes_norm[:min_num]
            
            # 计算组合损失
            b_loss = sinkhorn_distance(pred_boxes, gt_boxes_norm, epsilon=0.1, max_iters=100, tol=1e-9)
            box_loss += b_loss
            num_boxes += min_num

        if num_boxes > 0:
            box_loss = self.box_loss_weight * box_loss / num_boxes
        else:
            box_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 总损失
        loss = ce_loss  + box_loss#ce_loss + mask_loss + center_loss + box_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            #"mask_bce_loss": mask_bce_loss,
            #"mask_dice_loss": mask_dice_loss,
            #"mask_loss": mask_loss,
            #"center_loss": center_loss,
            "box_loss": box_loss
        }