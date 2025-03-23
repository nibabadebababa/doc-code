# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import torch
import time
import tqdm
from utils import AverageMeter, ProgressMeter, Summary
from dataset import MixedTrainingDataset
from doctamper_dataset import DocTamperDataset
import numpy as np
def train_one_epoch1(train_loader, model_engine, epoch, train_iter, args, logger): 
    """main epoch"""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    #mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    #mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    #mask_losses = AverageMeter("MaskLoss", ":.4f")
    # center_losses = AverageMeter("CenterLoss", ":.4f")  # 新增，用于记录 center_loss
    box_losses= AverageMeter("BoxLoss", ":.4f")

    progress = ProgressMeter(
        len(train_loader) if args.no_sampling else args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            #mask_losses,
            #mask_bce_losses,
            #mask_dice_losses,
            # center_losses,  # 将 center_losses 添加到进度显示中
            box_loss
        ],
        prefix="Epoch: [{}/{}]".format(epoch + 1, args.epochs),
        logger=logger
    )

    # 切换到训练模式
    model_engine.train()
    end = time.time()
    if args.no_sampling:
        for global_step, input_dict in enumerate(train_loader):
            
            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                #input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                #input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                #input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            output_dict = model_engine(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            #mask_bce_loss = output_dict["mask_bce_loss"]
            #mask_dice_loss = output_dict["mask_dice_loss"]
            #mask_loss = output_dict["mask_loss"]
            # center_loss = output_dict["center_loss"]  # 从输出字典中获取 center_loss
            box_loss = output_dict["box_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            #mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            #mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            #mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            # center_losses.update(center_loss.item(), input_dict["images"].size(0))  # 更新 center_losses
            box_losses.update(box_loss.item(), input_dict["images"].size(0))

            model_engine.backward(loss)
            model_engine.step()
                
            # 记录耗时
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    losses.all_reduce()
                    ce_losses.all_reduce()
                    # mask_bce_losses.all_reduce()
                    # mask_dice_losses.all_reduce()
                    # mask_losses.all_reduce()
                    # center_losses.all_reduce()
                    box_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                # mask_bce_losses.reset()
                # mask_dice_losses.reset()
                # mask_losses.reset()
                # center_losses.reset()
                box_losses.reset()

        return train_iter
    else:
        for global_step in range(args.steps_per_epoch):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if args.precision == "fp16":
                    #input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    #input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    #input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()

                output_dict = model_engine(**input_dict)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                # mask_bce_loss = output_dict["mask_bce_loss"]
                # mask_dice_loss = output_dict["mask_dice_loss"]
                # mask_loss = output_dict["mask_loss"]
                # center_loss = output_dict["center_loss"]  # 获取 center_loss
                box_loss = output_dict["box_loss"]  # 获取 box_loss

                losses.update(loss.item(), input_dict["images"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
                # mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                # mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                # mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
                # center_losses.update(center_loss.item(), input_dict["images"].size(0))  # 更新 center_losses
                box_losses.update(box_loss.item(), input_dict["images"].size(0))  # 更新 box_losses

                model_engine.backward(loss)
            
                model_engine.step()
                
            # 记录耗时
            batch_time.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % args.print_freq == 0:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                # mask_bce_losses.all_reduce()
                # mask_dice_losses.all_reduce()
                # mask_losses.all_reduce()
                # center_losses.all_reduce()
                box_losses.all_reduce()

                if args.rank == 0:
                    progress.display(1 + global_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                # mask_bce_losses.reset()
                # mask_dice_losses.reset()
                # mask_losses.reset()
                # center_losses.reset()
                box_losses.reset()

        return train_iter

def train_one_epoch2(train_loader, model_engine, epoch, train_iter, args, logger, input_dict=None, current_step=0):
    """main epoch"""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    # mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    # mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    # mask_losses = AverageMeter("MaskLoss", ":.4f")
    #center_losses = AverageMeter("CenterLoss", ":.4f")
    quality_meter = AverageMeter("CompressQuality", ":.1f")
    box_losses = AverageMeter("BoxLoss", ":.4f")
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, 
        #  mask_losses, 
        #  mask_bce_losses, 
        #  mask_dice_losses, 
        #  center_losses, 
         quality_meter, box_losses],
        prefix=f"Epoch: [{epoch + 1}/{args.epochs}]",
        logger=logger
    )

    model_engine.train()
    end = time.time()
    if args.no_sampling:
        if input_dict is None:
            input_dict = next(train_iter)
            
        if hasattr(train_loader, 'dataset') and isinstance(train_loader.dataset, MixedTrainingDataset):
            if len(train_loader.dataset.all_datasets) > 0 and isinstance(train_loader.dataset.all_datasets[0], DocTamperDataset):
                current_quality = train_loader.dataset.all_datasets[0].get_dynamic_quality()
                quality_meter.update(current_quality)
        
        data_time.update(time.time() - end)
        input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            #input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            #input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            #input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        output_dict = model_engine(**input_dict)

        loss = output_dict["loss"]
        ce_loss = output_dict["ce_loss"]
        # mask_bce_loss = output_dict["mask_bce_loss"]
        # mask_dice_loss = output_dict["mask_dice_loss"]
        # mask_loss = output_dict["mask_loss"]
        # center_loss = output_dict["center_loss"]  # 从输出字典中获取 center_loss
        box_loss = output_dict["box_loss"]  # 从输出字典中获取 box_loss
        losses.update(loss.item(), input_dict["images_clip"].size(0))
        ce_losses.update(ce_loss.item(), input_dict["images_clip"].size(0))
        # mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
        # mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
        # mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
        # center_losses.update(center_loss.item(), input_dict["images"].size(0))  # 更新 center_losses
        box_losses.update(box_loss.item(), input_dict["images_clip"].size(0))  # 更新 box_losses
        model_engine.backward(loss)
        model_engine.step()
            
        # 记录耗时
        batch_time.update(time.time() - end)
        end = time.time()

        if (current_step + 1) % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                # mask_bce_losses.all_reduce()
                # mask_dice_losses.all_reduce()
                # mask_losses.all_reduce()
                # center_losses.all_reduce()
                quality_meter.all_reduce()
                box_losses.all_reduce()
            if args.rank == 0:
                progress.display(1 + current_step)
                
            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            # mask_bce_losses.reset()
            # mask_dice_losses.reset()
            # mask_losses.reset()
            # center_losses.reset()
            quality_meter.reset()
            box_losses.reset()

        current_step += 1
        if hasattr(train_loader.dataset, 'set_current_step'):
            train_loader.dataset.set_current_step(current_step)

        return train_iter, current_step
    else:
        for _ in range(args.steps_per_epoch):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

                if hasattr(train_loader, 'dataset') and isinstance(train_loader.dataset, MixedTrainingDataset):
                    if len(train_loader.dataset.all_datasets) > 0 and isinstance(train_loader.dataset.all_datasets[0], DocTamperDataset):
                        current_quality = train_loader.dataset.all_datasets[0].get_dynamic_quality()
                        quality_meter.update(current_quality)

                data_time.update(time.time() - end)
                input_dict = dict_to_cuda(input_dict)

                if args.precision == "fp16":
                    #input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    #input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    #input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()

                output_dict = model_engine(**input_dict)

                loss = output_dict["loss"]
                ce_loss = output_dict["ce_loss"]
                # mask_bce_loss = output_dict["mask_bce_loss"]
                # mask_dice_loss = output_dict["mask_dice_loss"]
                # mask_loss = output_dict["mask_loss"]
                # center_loss = output_dict["center_loss"]  # 获取 center_loss
                box_loss = output_dict["box_loss"]  # 获取 box_loss

                losses.update(loss.item(), input_dict["images_clip"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images_clip"].size(0))
                # mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                # mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                # mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
                # center_losses.update(center_loss.item(), input_dict["images"].size(0))  # 更新 center_losses
                box_losses.update(box_loss.item(), input_dict["images_clip"].size(0))  # 更新 box_losses
                model_engine.backward(loss)
            
                model_engine.step()
                
            # 记录耗时
            batch_time.update(time.time() - end)
            end = time.time()

            if (current_step + 1) % args.print_freq == 0:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                # mask_bce_losses.all_reduce()
                # mask_dice_losses.all_reduce()
                # mask_losses.all_reduce()
                # center_losses.all_reduce()
                quality_meter.all_reduce()
                box_losses.all_reduce()
                if args.rank == 0:
                    progress.display(1 + current_step)
                    
                batch_time.reset()
                data_time.reset()
                losses.reset()
                ce_losses.reset()
                # mask_bce_losses.reset()
                # mask_dice_losses.reset()
                # mask_losses.reset()
                # center_losses.reset()
                quality_meter.reset()
                box_losses.reset()
            
            current_step += 1
            if hasattr(train_loader.dataset, 'set_current_step'):
                train_loader.dataset.set_current_step(current_step)

        return train_iter, current_step

@torch.no_grad()
def validate(val_loader, model_engine, epoch, args, logger):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            #input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            #input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            #input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        output_dict = model_engine(**input_dict)

        #pred_masks = output_dict["pred_masks"]
        #masks_list = output_dict["gt_masks"][0].long()
        #output_list = (pred_masks[0] > 0).long()
        #assert len(pred_masks) == 1

        pred_box = output_dict["pred_box"]
        gt_box=output_dict["gt_box"]
        resize_list=output_dict["resize_list"][0]
        height,width=resize_list[:2]
        gt_mask=np.zeros((height+1,width+1),dtype=np.uint8)
        pred_mask=np.zeros((height+1,width+1),dtype=np.uint8)   
        for bbox_i,gt_i in zip(pred_box, gt_box):
            bbox_i = bbox_i.cpu().numpy()
            gt_i = gt_i.cpu().numpy()
            pred_mask[int(bbox_i[1]):int(bbox_i[3]),int(bbox_i[0]):int(bbox_i[2])]=1
            gt_mask[int(gt_i[1]):int(gt_i[3]),int(gt_i[0]):int(gt_i[2])]=1
        intersection = (pred_mask*gt_mask).sum((0,1))
        preds = pred_mask.sum((0,1))
        target_sum = gt_mask.sum((0,1))
        prec=intersection/(preds+1e-8)
        recall=intersection/target_sum
        f1=2*prec*recall/(prec+recall+1e-8)
        union = (np.maximum(pred_mask, gt_mask)).sum((0,1))
        acc_iou= intersection/union

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou)
        # #接下来的要删掉，前面已实现iou，intersection和union的计算
        # device = pred_masks[0].device
        # intersection, union, acc_iou = 0.0, 0.0, 0.0  
        # for mask_i, output_i in zip(masks_list, output_list):
        #     # mask_i和output_i都是int64，device(type='cuda', index=0)，是tensor张量
        #     intersection_i, union_i, _ = intersectionAndUnionGPU(
        #         output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
        #     )
        #     intersection += intersection_i
        #     union += union_i
        #     acc_iou += intersection_i / (union_i + 1e-5)
        #     acc_iou[union_i == 0] += 1.0  # no-object target #tensor([1.9424e+00, 2.6278e-05], device='cuda:0')
        # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()# intersection的值是array([5.780637e+06, 3.000000e+00], dtype=float32)，union是array([5952093.,  171459.], dtype=float32)
        # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]# acc_iouarray([9.7119397e-01, 1.3138993e-05], dtype=float32)
        # intersection_meter.update(intersection), union_meter.update(
        #     union
        # ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    logger.info(f"[{epoch + 1:d}] On {val_loader.dataset.ds} giou: {giou:.4f}, ciou: {ciou:.4f}.")


    return giou, ciou

@torch.no_grad()
def eval_gres(val_loader, model_engine, epoch, args, logger):
    model_engine.eval()
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    precision_meter = AverageMeter("Precision", ":6.3f", Summary.AVERAGE)
    recall_meter = AverageMeter("Recall", ":6.3f", Summary.AVERAGE)
    f1_meter = AverageMeter("F1", ":6.3f", Summary.AVERAGE)
    iou_meter = AverageMeter("IoU", ":6.3f", Summary.AVERAGE)  # 新增 IoU meter

    with torch.no_grad():
        for sample_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
            
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                #input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                #input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                #input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            
            output_dict = model_engine(**input_dict)
            # pred_masks = output_dict["pred_masks"][0].ge(0).int()
            # gt_masks = output_dict["gt_masks"][0].int()
            
            output_ids = output_dict["output_ids"][0]
            # seg_index = ((output_ids == args.seg_token_idx)).nonzero(as_tuple=True)[0]
            # assert len(seg_index) == len(gt_masks)
            # assert len(pred_masks) == len(gt_masks)
            

            ##根据得到的pred_box和gt——box来计算而不是mask
            ##output dict格式：
            # 
            # outputdict = {
            #     "pred_box":predicted_boxes_list_resized, # [B, N, 4] 
            #     "gt_box":boxes_tensor_list,
            #     "resize_list":resize_list,# [B, 2][0]为h，[1]为w
            #     "output_ids": output_ids
            # }
            pred_box = output_dict["pred_box"]
            gt_box=output_dict["gt_box"]
            resize_list=output_dict["resize_list"][0]
            height,width=resize_list[:2]
            gt_mask=np.zeros((height+1,width+1),dtype=np.uint8)
            pred_mask=np.zeros((height+1,width+1),dtype=np.uint8)   
            for bbox_i,gt_i in zip(pred_box, gt_box):
                bbox_i = bbox_i.cpu().numpy()
                gt_i = gt_i.cpu().numpy()
                pred_mask[int(bbox_i[0,1]):int(bbox_i[0,3]),int(bbox_i[0,0]):int(bbox_i[0,2])]=1
                gt_mask[int(gt_i[0,1]):int(gt_i[0,3]),int(gt_i[0,0]):int(gt_i[0,2])]=1
            intersection = (pred_mask*gt_mask).sum((0,1))##tp
            preds = pred_mask.sum((0,1))#tp+fp
            target_sum = gt_mask.sum((0,1))#tp+fn
            precision=intersection/(preds+1e-8)
            recall=intersection/target_sum
            f1_score=2*precision*recall/(precision+recall+1e-8)
            union = (np.maximum(pred_mask, gt_mask)).sum((0,1))
            iou= intersection/union

            
            precision_meter.update(precision)
            recall_meter.update(recall)
            f1_meter.update(f1_score)
            inter_meter.update(intersection)
            union_meter.update(union)
            iou_meter.update(iou)  # 更新 IoU meter

            # #接下来的要删掉，前面已实现iou，intersection和union,f1,precision,recall的计算
            # #还缺少accurancy与giou，ciou和mean iou
            # for b_idx, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            #     inter_i, union_i, _ = intersectionAndUnionGPU(
            #         pred.contiguous().clone(),
            #         gt.contiguous().clone(),
            #         K=2, ignore_index=255
            #     )
            #     inter_i = inter_i.cpu().numpy()
            #     union_i = union_i.cpu().numpy()
            #     # 计算真阳性、假阳性、假阴性、真阴性
            #     tp = inter_i[1]  # 类别1的交集（True Positive）
            #     fp = (pred.sum().item() - tp)  # 预测为正但实际为负（False Positive）
            #     fn = (gt.sum().item() - tp)    # 实际为正但预测为负（False Negative）
            #     tn = (pred.numel() - pred.sum().item() - fn)  # 预测为负且实际为负（True Negative）
            #     # 确保指标为非负数
            #     tp = max(tp, 0)
            #     fp = max(fp, 0)
            #     fn = max(fn, 0)
            #     tn = max(tn, 0)
            #     # 计算精确率、召回率、F1分数
            #     precision = tp / (tp + fp + 1e-8)
            #     recall = tp / (tp + fn + 1e-8)
            #     f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            #     # 计算准确率
            #     accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            #     # 更新指标
            #     precision_meter.update(precision)
            #     recall_meter.update(recall)
            #     f1_meter.update(f1_score)
            #     accuracy_meter.update(accuracy)

            #     # 计算单个样本的IoU
            #     this_giou = inter_i / (union_i + 1e-8)
                
            #     # 更新交集、并集和IoU的计数器
            #     inter_meter.update(inter_i)
            #     union_meter.update(union_i)
            #     g_iou_meter.update(this_giou)
                
            #     # 计算平均IoU（Mean IoU），包括并集为0的类别
            #     mean_iou = this_giou.mean()
            #     mean_iou_meter.update(mean_iou)

            #     # 计算 IoU
            #     iou = inter_i[1] / (union_i[1] + 1e-8)  # 使用类别1的交并比
            #     iou_meter.update(iou)  # 更新 IoU

    # 汇总所有进程的计数器（用于分布式训练）
    inter_meter.all_reduce()
    union_meter.all_reduce()
    precision_meter.all_reduce()
    recall_meter.all_reduce()
    f1_meter.all_reduce()
    iou_meter.all_reduce()  # 新增 IoU 的 all_reduce


    precision_avg = precision_meter.avg
    recall_avg = recall_meter.avg
    f1_avg = f1_meter.avg
    iou_avg = iou_meter.avg  # 获取平均 IoU

    logger.info(
        f"[{epoch + 1:d}] {val_loader.dataset.val} "
        f" iou: {iou_avg:.4f}, f1: {f1_avg:.4f}, "
        f"precision: {precision_avg:.4f}, recall: {recall_avg:.4f}, "
    )
    return  f1_avg, precision_avg, recall_avg, iou_avg

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output_shape = {output.shape}, target_shape = {target.shape}"
    output = output.reshape(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
