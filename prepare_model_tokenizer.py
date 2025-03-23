# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import torch
from peft import get_peft_model, LoraConfig
from model.llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
import os
import logging

def add_task_tokens(tokenizer, args):
    # 1. 将 pad_token 设置为 unk_token
    # tokenizer.pad_token = tokenizer.unk_token

    # 2. 定义所有需要的额外特殊标记
    # task_special_tokens = [f'<{i}>' for i in range(1, 1025)]
    # 额外添加5个segtoken
    pot_special_tokens = ["[POT1]", "[POT2]" ,"[POT3]" ,"[POT4]" ,"[POT5]"]
    box_special_tokens = ["[BOX1]", "[BOX2]" ,"[BOX3]" ,"[BOX4]" ,"[BOX5]"]
    additional_special_tokens = pot_special_tokens + box_special_tokens

    special_tokens_dict = {'additional_special_tokens': additional_special_tokens}

    # 3. 添加这些特殊标记到分词器
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # args.task_token_idx = tokenizer.convert_tokens_to_ids(task_special_tokens)
    # 存储 segmentation tokens 的 token IDs
    args.pot_token_idx = tokenizer.convert_tokens_to_ids(pot_special_tokens)
    args.box_token_idx = tokenizer.convert_tokens_to_ids(box_special_tokens)
    # 5. 打印它们对应的 token IDs
    # print("=== Task Tokens and their IDs ===")
    # for token, idx in zip(task_special_tokens, args.task_token_idx):
    #     print(f"Token: {token} --> ID: {idx}")
    print("=== Pot Tokens and their IDs ===")
    for token, idx in zip(pot_special_tokens, args.pot_token_idx):
        print(f"Token: {token} --> ID: {idx}")
    print("=== Box Tokens and their IDs ===")
    for token, idx in zip(box_special_tokens, args.box_token_idx):
        print(f"Token: {token} --> ID: {idx}")
    
    # print("\n=== Segmentation Tokens and their IDs ===")
    # for token, idx in zip(seg_special_tokens, args.seg_token_idx):
    #     print(f"Token: {token} --> ID: {idx}")

    # 5. 如果需要，添加 <im_start> 和 <im_end> 标记
    # if args.use_mm_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    return tokenizer, args

def print_special_tokens(tokenizer, args):
    # 打印 <1> 到 <1024> 的 token 及其 ID
    print("\n<1> 到 <1024> 的特殊 token 和对应的 ID:")
    for i, token_id in enumerate(args.box_token_idx, 1):
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f'Token: {token}, Number: {i}, ID: {token_id}')
    
    # 如果添加了 <im_start> 和 <im_end>，也打印出来
    if hasattr(args, 'im_start_token_id') and hasattr(args, 'im_end_token_id'):
        print(f'\n<im_start> token: "<im_start>", ID: {args.im_start_token_id}')
        print(f'<im_end> token: "<im_end>", ID: {args.im_end_token_id}')

def init_vision_seg_for_model(model, tokenizer, args):
    # Register special token ids
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Set up gradckpt for saving memory
    model.gradient_checkpointing_enable() #!
    # Init CLIP-ViT
    # for attr in dir(model):
    #     print(attr)
    # model.get_model().initialize_vision_modules(model.get_model().config)
    # vision_tower = model.get_model().get_vision_tower()
    # vision_tower.to(dtype=args.torch_dtype, device=args.local_rank)
    # Init segmentation module
    model.get_model().init_seg_and_proj(model.get_model().config)
    model.resize_token_embeddings(len(tokenizer))
    # 请确保在调整嵌入层大小之前加载预训练权重
#######################需要添加############################
        # 加载权重
    merge_lora_path = args.merge_lora_path
    fine_tuned_modules_path = args.fine_tuned_modules_path

    if merge_lora_path and os.path.exists(merge_lora_path):
        print(f"Loading LoRA adapter model from {merge_lora_path}")
        adapter_model = torch.load(merge_lora_path)
        model.load_state_dict(adapter_model, strict=False)  # 加载 LoRA 权重到模型
    else:
        print("No LoRA weights to load.")

    if fine_tuned_modules_path and os.path.exists(fine_tuned_modules_path):
        print(f"Loading fine-tuned model weights from {fine_tuned_modules_path}")
        fine_tuned_state_dict = torch.load(fine_tuned_modules_path)
        model.load_state_dict(fine_tuned_state_dict, strict=False)  # 加载微调权重
    else:
        print("No fine-tuned modules to load.")

    # print(model)
    
    # Freeze all parameters
    for n, p in model.named_parameters():
        p.requires_grad_(False)
        
    # # 解冻最后两层
    # num_layers = len(model.get_model().layers)  # 修正：使用 layers 而不是 encoder.blocks
    # for i in range(num_layers - 1, num_layers):
    #     for n, p in model.get_model().layers[i].named_parameters():  # 遍历解码器层
    #         p.requires_grad_(True)


    # Get Lora model, validation lora_r must be 0
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                # "visual_model",
                                "mask_decoder",
                                "vision_tower",
                                "mm_projector",
                                "center_hidden_fcs",
                                "box_hidden_fcs",
                                "lm_head"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        print(lora_target_modules)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = get_peft_model(model, lora_config)
        # print(model)
        print(f"LoRA finetuning with rank = {lora_r}.")
    
    # model.resize_token_embeddings(len(tokenizer))
    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_parts_keys = ["lm_head", "embed_tokens", "mask_decoder", "center_hidden_fcs","box_hidden_fcs"]
    if lora_r < 0:
        trainable_parts_keys.append("model.layers")
        print("No LoRA, full LLM finetuning.")
    elif lora_r == 0:
        print("LLM left frozen.")
    if not args.eval_only:
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in trainable_parts_keys
                ]
            ):
                p.requires_grad_(True)
        # Set up input with grads
        model.enable_input_require_grads()
        
        # # 添加日志记录
        # logger = logging.getLogger('AdapterTraining')
        # logger.setLevel(logging.INFO)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        # logger.addHandler(handler)
        
        # # 打印可训练参数的数量和名称
        # trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
        # logger.info(f"Number of trainable parameters: {len(trainable_params)}")
        # logger.info(f"Trainable parameters: {trainable_params}")
    return model
