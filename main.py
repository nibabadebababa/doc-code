# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import argparse
import os
from pickle import NONE
import shutil
from functools import partial
from turtle import Turtle

from regex import T
import torch
import transformers
import deepspeed
import deepspeed.comm as dist
from torch.utils.data import DataLoader, DistributedSampler

import model.llava.conversation as conversation_lib
from prepare_model_tokenizer import add_task_tokens, init_vision_seg_for_model, print_special_tokens
from modeling_seg_to_coords_lora_box import LisaGSVAForCausalLM
from dataset import MixedTrainingDataset, ValDataset, collate_fn
from solver import train_one_epoch1,train_one_epoch2, validate, eval_gres
from utils import get_logger
## from model.llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="GSVA Training and Evaluation")
    parser.add_argument("--local_rank", default=0, type=int, help="For local rank in distributed training")
    parser.add_argument(
        "--mllm_model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"#/home/victory/zr/LISA-main/LLaVA-Lightning-7B-delta-v1-1//home/victory/zr/llava-v1.6-vicuna-7b
    )#"/home/victory/zr/TPLM-main/outputs/default/ckpt_model_12-SAMlora/merged_model",
    parser.add_argument("--dataset_dir", required=False, type=str, help="Where do we store the huge datasets?", default="/root/autodl-tmp/DocTamper")# 把required改成false了，加了个default
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"], help="precision for training and inference")
    parser.add_argument("--image_size", default=1024, type=int, help="Image size of segmentation model.")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct", type=str)#/home/victory/zr/LISA-main/openai/clip-vit-large-patch14,/home/victory/zr/TPLM-main/openai/clip-vit-large-patch14-336
    parser.add_argument(
        "--dataset", default="DocTamper", type=str#DocTamper||Receipt_ID||RealTextManipulation||T-SROIE
    )
    parser.add_argument("--doctam_data", default="DocTamperV1-TrainingSet", type=str)
    parser.add_argument("--rtm_data", default="RealTextManipulation|train", type=str)
    parser.add_argument("--tsr_data", default="T-SROIE|train", type=str)   
    parser.add_argument("--rid_data", default="Receipt_ID|train", type=str)
    
    parser.add_argument("--sample_rates", default="50", type=str)#指定不同数据集的采样率,用逗号分割, eg. 44,5,1
    parser.add_argument("--val_dataset", default="DocTamper|DocTamperV1-SCD", type=str)
    #DocTamper|DocTamperV1-TestingSet, DocTamper|DocTamperV1-SCD
    #DocTamper|DocTamperV1-FCD  RealTextManipulation|test, CERTD|test, IDCD|test, PSCD|test
    #T-SROIE|test
    
    parser.add_argument("--log_base_dir", default="./outputs", type=str)
    parser.add_argument("--exp_name", default="default", type=str)
    parser.add_argument("--epochs", default=24, type=int)
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=2, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=4.0, type=float)
    ##parser.add_argument("--center_loss_weight", default=5.0, type=float)
    parser.add_argument("--box_loss_weight", default=5.0, type=float)

    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,image_encoder", type=str)#image_encoder
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=5, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    ##parser.add_argument("--segmentation_model_path", default="/home/victory/zr/TPLM-main/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    # !因为每个模型的lm_head的层数不同，所以需要指定层数
    parser.add_argument("--in_dim", default=3584, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False, help='Whether resume the latest checkpoint when training is interrupted.')
    parser.add_argument("--no_sampling", action="store_true", default=True, help="Only one dataset finetuning, train on full length dataset.")
    parser.add_argument('--val_refzom', action='store_true', default=False, help='Default gres/zom evaluation, if True, RefZOM, else gRefCOCO.')
    parser.add_argument(
        "--conv_type",
        default="llava_v1", 
        type=str,
        choices=["llava_v1", "llava_llama_2","qwen2_vl"],
    )
    parser.add_argument("--merge_lora_path", type=str, default=None, help="Path to destination HF checkpoint.")#"/home/victory/zr/TPLM-main/outputs/default/lora_adapter/adapter_model.bin"
    parser.add_argument(
        "--fine_tuned_modules_path", type=str, default=None, help="Path to fine-tuned modules, is a pth file"
    )#"/home/victory/zr/TPLM-main/outputs/default/fine_tuned_modules.pth"
    parser.add_argument("--weight", type=str, default=None, help="Path to a bin ckpt.")#"/home/victory/zr/TPLM-main/tplm_gen_model/SAM-lora-boxpoint-20best-FINISH.bin"
    parser.add_argument("--min_quality", type=int, default=75, help="Minimum compression quality")
    parser.add_argument("--T", type=int, default=25, help="Decay coefficient for quality")
    
    # 添加 distributed 参数
    parser.add_argument("--distributed", action="store_true", default=True,
                      help="Use distributed training")
    
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    # Get arguments from commandline
    args = parse_args()
    
    # Set up Deepspeed distributed environment
    torch.cuda.set_device(args.local_rank)
    dist.init_distributed()
    args.world_size = world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()

    # Set up logging dir
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir, rank, name=args.exp_name)

    # Create model
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.mllm_model_path,
    #     cache_dir=None,
    #     model_max_length=args.model_max_length,
    #     padding_side="right",
    #     use_fast=False
    # )
    
    # from transformers import Qwen2TokenizerFast
    # tokenizer = Qwen2TokenizerFast.from_pretrained("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    
    processor.tokenizer, args = add_task_tokens(processor.tokenizer, args)
    print_special_tokens(processor.tokenizer, args)
    
    # test_sentence = "This is a test sentence with [BOX1] and [BOX2]."
    # encoded_input = tokenizer(test_sentence, return_tensors="pt")

    # print("Encoded Input IDs:", encoded_input.input_ids)
    # print("Tokenized Output:", tokenizer.convert_ids_to_tokens(encoded_input.input_ids[0]))

    # Determine working model precision
    args.torch_dtype = torch.float32
    if args.precision == "bf16":
        args.torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        args.torch_dtype = torch.half
    
  
    # Prepare model creation arguments
    #! 这里算是一个接口输入，debug的话可以直接在这改
    model_args = {
        # "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "in_dim": args.in_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        # "center_loss_weight": args.center_loss_weight,
        "box_loss_weight": args.box_loss_weight,
        # "num_start_token_idx": args.num_start_token_idx,
        # "num_end_token_idx": args.num_end_token_idx,
        # "task_token_idx": args.task_token_idx,
        "box_token_idx": args.box_token_idx,
        "pot_token_idx": args.pot_token_idx,
        # "segmentation_model_path": args.segmentation_model_path,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        # "tokenizer": tokenizer
    }
    # 创建模型时不传递 tokenizer
    model = LisaGSVAForCausalLM.from_pretrained(
        args.mllm_model_path, 
        torch_dtype=args.torch_dtype,
        # device_map="auto",
        low_cpu_mem_usage=False,
        # empty_init=False, #!
        **model_args
    )
    print(model)
    
    
    # new_tokens = ["[POT1]", "[POT2]" ,"[POT3]" ,"[POT4]" ,"[POT5]", "[BOX1]", "[BOX2]" ,"[BOX3]" ,"[BOX4]" ,"[BOX5]"]
    # new_tokens_to_add = [token for token in new_tokens if token not in processor.tokenizer.get_vocab()]
    # if new_tokens_to_add:
    #     # 向 tokenizer 添加新标记
    #     num_added_toks = processor.tokenizer.add_tokens(new_tokens_to_add)
    #     print(f'Added {num_added_toks} tokens.')
        # 调整模型的嵌入矩阵大小以适应新增加的标记
        # model.resize_token_embeddings(len(processor.tokenizer)) 
    
    # 单独设置 tokenizer
    model.tokenizer = processor.tokenizer
    
    # Explicitly set tokenizer for the model
    # Verify model.tokenizer consistency
    # assert model.tokenizer("[POT1]", add_special_tokens=False).input_ids[0] == args.pot_token_idx, \
    #     "[DEBUG] Model tokenizer and args.pot_token_idx are not consistent!"
    # assert model.tokenizer("[BOX1]", add_special_tokens=False).input_ids[0] == args.box_token_idx, \
    #     "[DEBUG] Model tokenizer and args.box_token_idx are not consistent!"
    # Set up two vision models for whole model, and lora
    model = init_vision_seg_for_model(model, processor.tokenizer, args)
    # Evaluation or finetuning, btw, merge-lora always fails
    
    if args.weight is not None: # `args.weight`` is a large `*.bin` file.
        state_dict = torch.load(args.weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Load trained weights successfully!")
    # Specify the conversation type
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    # Build training set
    if args.eval_only:
        train_dataset = None
    else:
        # 这里会进入到数据集对应的dataset.py文件
        train_dataset = MixedTrainingDataset(
            args.dataset_dir,
            processor.tokenizer,
            args.vision_tower,
            samples_per_epoch=args.batch_size
            * args.grad_accumulation_steps
            * args.steps_per_epoch
            * world_size,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=args.exclude_val,
            dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")],
            doctam_data=args.doctam_data,
            rtm_data=args.rtm_data,
            tsr_data=args.tsr_data,
            rid_data=args.rid_data,
            explanatory=args.explanatory,
            no_sampling=args.no_sampling,
            min_quality=args.min_quality,
            T=args.T
        )
    if args.no_eval:
        val_dataset = None
        logger.info(f"Training with {len(train_dataset)} examples.")
    else:
        val_dataset = ValDataset(
            args.dataset_dir,
            processor.tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size
        )

        if args.eval_only:
            logger.info(f"Testing with {len(val_dataset)} examples.")
        # else:
        #     logger.info(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples, also validating on gRefCOCO with {len(grefcoco_val_ds)} examples.")
    # The accelerated training configurations only work for ZeRO-2.
    if args.eval_only:
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            }
        }
    else:
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "weight_decay": 0.0,
                    "betas": (args.beta1, args.beta2),
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": args.epochs * args.steps_per_epoch,
                    "warmup_min_lr": 0,#1.0e-7
                    "warmup_max_lr": args.lr,
                    "warmup_num_steps": 100,#200
                    "warmup_type": "linear",
                },
            },
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
                },
                "offload_param": {
                "device": "cpu",
                "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            }, 
        }
    # Build a model engine wrapped with Deepspeed
    if args.eval_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config
        )
    else:
        logger.info('Before initializing deepspeed zero optimizer...')
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                processor=processor,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=local_rank,
            ),
            config=ds_config
        )

        train_loader.num_local_io_workers = args.workers
        logger.info('After initializing deepspeed zero optimizer!')
    # resume deepspeed checkpoint, `auto-resume` snippets are borrowed from Swin Transfomer codebase:
    # https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/utils.py#L163
    if args.auto_resume:
        checkpoints = os.listdir(args.log_dir)
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.startswith('ckpt_model')]
        if len(checkpoints) > 0:
            args.resume = max([os.path.join(args.log_dir, d) for d in checkpoints], key=os.path.getmtime)
            logger.info(f"Auto resume found latest: {args.resume}")
        else:
            logger.info("No auto resume.")
    if args.resume: # resume from training, scattered checkpoints (list of ***.pt)
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        logger.info(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )
    # Build validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                processor=processor,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=local_rank
            )
        )

    if args.eval_only:
        eval_gres(val_loader, model_engine, 0, args, logger)
        return
    # Otherwise, we train the model using the initialized Deepspeed-Zero model engine.
    logger.info("Training begin!")
    train_iter = iter(train_loader)
    best_score = 0.0
    global_step = 0  # 用于课程学习的全局步数
    
    for epoch in range(args.start_epoch, args.epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        steps_this_epoch = 0
        while steps_this_epoch < args.steps_per_epoch:
            global_step += 1  # 课程学习的步数持续增加
            
            if hasattr(train_loader.dataset, 'set_current_step'):
                if dist.is_initialized():
                    dist.barrier()
                train_loader.dataset.set_current_step(global_step)
            
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)
            
            # 传入 steps_this_epoch + 1 作为显示用的步数，并更新 train_iter
            train_iter, steps_this_epoch = train_one_epoch2(
                train_loader, model_engine, epoch, train_iter,
                args, logger, input_dict, steps_this_epoch
            )         
        # 不需要再增加 steps_this_epoch，因为在 train_one_epoch2 中已经增加了
        dist.barrier()
        # 在每个 epoch 结束后进行评估
        if not args.no_eval:
            # giou, ciou, f1, precision, recall, acc, miou, iou = eval_gres(
            #     val_loader, model_engine, epoch, args, logger)
            f1, precision, recall, iou = eval_gres(
                val_loader, model_engine, epoch, args, logger)
            if rank == 0:
                with open(os.path.join(args.log_dir, "quick_look_result.log"), "a") as t:
                    t.write(
                        f"[{epoch + 1}] Doctamper: F1:{f1:.4f},IoU:{iou:.4f},"
                        f"Precision:{precision:.4f},Recall:{recall:.4f}.\n"
                    )
                current_f1 = f1
                is_best = current_f1 > best_score
                best_score = max(current_f1, best_score)
            else:
                is_best = False
                current_f1 = 0.0
            # 同步is_best到所有进程
            is_best_tensor = torch.tensor([is_best], dtype=torch.bool, device=model_engine.device)
            dist.broadcast(is_best_tensor, src=0)
            is_best = is_best_tensor.item()
            # 保存最佳模型
            if is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model_best")
                # 同步所有进程
                dist.barrier()
                if rank == 0:
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                dist.barrier()
                model_engine.save_checkpoint(save_dir)
                dist.barrier()
                if rank == 0:
                    logger.info(f"Saved best model checkpoint at {save_dir} with F1 {current_f1:.4f}")
            

def worker_init_fn(worker_id):
    """初始化 worker 进程"""
    print(f"\n=== Worker {worker_id} initializing ===")
    worker = torch.utils.data.get_worker_info()
    dataset = worker.dataset
    
    # 确保 worker 进程能够访问共享变量
    if hasattr(dataset, 'all_datasets'):
        print(f"Worker {worker_id} has {len(dataset.all_datasets)} datasets")
        for ds in dataset.all_datasets:
            if hasattr(ds, 'shared_step'):
                print(f"Worker {worker_id} found shared_step in dataset type: {type(ds)}")
                # 重新连接到共享变量
                ds.shared_step = ds.shared_step
                print(f"Worker {worker_id} reconnected shared_step, value = {ds.shared_step.value}")

if __name__ == "__main__":
    main()