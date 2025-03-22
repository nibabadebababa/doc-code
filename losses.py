# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Written by Zhuofan Xia
# --------------------------------------------------------

import torch
import torch.nn.functional as F

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss_new(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def dice_loss_new(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, eps=1e-6):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    # 计算前景比重
    foreground_fraction = targets.sum(dim=-1) / (targets.shape[-1] + eps)
    weights = 1.0 + 4.0 * foreground_fraction  # 提高前景比重
    
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = (1 - (numerator + eps) / (denominator + eps)) * weights
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, foreground_weight=4.0):
    # 对前景像素加权
    pos_weight = torch.ones_like(targets) * (1 + foreground_weight)
    pos_weight[targets == 0] = 1  # 背景像素不加权
    loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction="none")
    loss = loss.flatten(1).mean(1).sum() / (num_masks + 1e-8)
    return loss


def iou_loss(
    pred_iou: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    num_masks: float
):
    pred_iou = pred_iou.to(torch.float32).sigmoid()
    pred_mask_ = pred_mask.detach().clone()
    target_mask_ = target_mask.detach().clone()
    inter = (pred_mask_ * target_mask_).sum()
    union = pred_mask_.sum() + target_mask_.sum() - inter
    gt_iou = inter / (union + 1e-8)
    
    iou_loss = ((gt_iou - pred_iou) ** 2).sum() / (num_masks + 1e-8)
    return iou_loss

def modified_dice_loss(inputs, targets, num_masks, eps=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    intersection = (inputs * targets).sum(dim=1)
    union = inputs.sum(dim=1) + targets.sum(dim=1)
    dice_score = (2 * intersection + eps) / (union + eps)

    # 使用 Focal 的思想
    loss = (1 - dice_score) ** 1  # 可以调整指数以控制难易样本的权重
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

import torch

import torch

def sinkhorn_distance(x, y, epsilon=0.1, max_iters=100, tol=1e-9):
    """
    Compute the Sinkhorn distance between two point sets x and y.
    Args:
        x: Tensor of shape [N, D]
        y: Tensor of shape [M, D]
        epsilon: Regularization parameter
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
    Returns:
        Sinkhorn distance (scalar)
    """
    # 检查并转换数据类型
    original_dtype = x.dtype
    if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
        x = x.float()
    if y.dtype == torch.bfloat16 or y.dtype == torch.float16:
        y = y.float()

    N, D = x.shape
    M, _ = y.shape

    # 计算成本矩阵
    C = torch.cdist(x, y, p=2)  # [N, M]
    # 初始化对偶变量
    u = torch.zeros(N, device=x.device, dtype=x.dtype)
    v = torch.zeros(M, device=y.device, dtype=y.dtype)
    # 均匀边际
    mu = torch.full((N,), 1.0 / N, device=x.device, dtype=x.dtype)
    nu = torch.full((M,), 1.0 / M, device=x.device, dtype=x.dtype)
    K = torch.exp(-C / epsilon)  # [N, M]
    for _ in range(max_iters):
        u_prev = u.clone()
        u = mu / (torch.matmul(K, v) + 1e-8)
        v = nu / (torch.matmul(K.t(), u) + 1e-8)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break
    # 运输计划
    P = torch.diag(u) @ K @ torch.diag(v)  # [N, M]
    # Sinkhorn 距离
    distance = torch.sum(P * C)
    # 如果原始数据类型不是 Float32，转换回去
    if original_dtype != torch.float32:
        distance = distance.to(original_dtype)

    return distance


#alpha 和 beta 参数用于控制 FP（假阳性）和 FN（假阴性）的权重，gamma 控制对难分类样本的关注度。
def focal_tversky_loss(inputs, targets, num_masks, alpha=0.3, beta=0.7, gamma=1.0, eps=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    TP = (inputs * targets).sum(dim=1)
    FP = ((1 - targets) * inputs).sum(dim=1)
    FN = (targets * (1 - inputs)).sum(dim=1)

    tversky_index = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    loss = ((1 - tversky_index) ** gamma).sum() / (num_masks + 1e-8)
    return loss

#alpha 参数用于平衡前景和背景的权重，gamma 控制难易样本的调整力度。您可以根据实际情况调整这两个参数。
def focal_loss(inputs, targets, num_masks, alpha=0.8, gamma=2.5):
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 应用 alpha 平衡参数
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    loss = loss.flatten(1).mean(1).sum() / (num_masks + 1e-8)
    return loss

def box_iou_loss(pred_boxes, gt_boxes, eps=1e-6):
    """计算框的 IOU 损失
    Args:
        pred_boxes: 预测框坐标 [N, 4] (归一化后的)
        gt_boxes: 真实框坐标 [N, 4] (归一化后的)
    Returns:
        iou_loss: 平均 IOU 损失
    """
    # 计算交集
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1]) 
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算并集
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = pred_area + gt_area - intersection

    iou = intersection / (union + eps)
    iou_loss = (1 - iou).mean()
    return iou_loss

def box_sinkhorn_loss(pred_boxes, gt_boxes, epsilon=0.1, max_iters=100, tol=1e-9):
    """使用 Sinkhorn 算法计算框的最优传输损失
    Args:
        pred_boxes: 预测框坐标 [N, 4] (归一化后的)
        gt_boxes: 真实框坐标 [N, 4] (归一化后的)
    Returns:
        sinkhorn_loss: Sinkhorn 距离损失
    """
    return sinkhorn_distance(pred_boxes, gt_boxes, epsilon=epsilon, max_iters=max_iters, tol=tol)

def combined_box_loss(pred_boxes, gt_boxes, num_boxes, iou_weight=0.5, sinkhorn_weight=0.5):
    """组合 IOU 损失和 Sinkhorn 损失
    Args:
        pred_boxes: 预测框坐标 [N, 4] (归一化后的)
        gt_boxes: 真实框坐标 [N, 4] (归一化后的)
        num_boxes: 框的数量
        iou_weight: IOU 损失权重
        sinkhorn_weight: Sinkhorn 损失权重
    Returns:
        total_loss: 总损失
    """
    if num_boxes == 0:
        return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

    iou_loss = box_iou_loss(pred_boxes, gt_boxes)
    sinkhorn_loss = box_sinkhorn_loss(pred_boxes, gt_boxes)
    
    total_loss = iou_weight * iou_loss + sinkhorn_weight * sinkhorn_loss
    return total_loss
