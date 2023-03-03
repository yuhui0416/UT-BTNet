import torch
import torch.nn as nn
import torch.nn.functional as F
from . import metrics
import numpy as np


# def calculate_distance(label_pred, label_true, spacing, C):
#     # the input args are torch tensors
#     if label_pred.is_cuda:
#         label_pred = label_pred.cpu()
#         label_true = label_true.cpu()
#
#     label_pred = label_pred.numpy()
#     label_true = label_true.numpy()
#     spacing = spacing.numpy()
#
#     ASD_list = np.zeros(C-1)
#     HD_list = np.zeros(C-1)
#
#     for i in range(C-1):
#         tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
#         dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
#         ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2
#
#         HD = metrics.compute_robust_hausdorff(tmp_surface, 95)
#         HD_list[i] = HD
#
#     return ASD_list, HD_list


def calculate_distance(label_pred, label_true, C):
    # the input args are torch tensors
    if label_pred.is_cuda:
        label_pred = label_pred.cpu()
        label_true = label_true.cpu()

    label_pred = label_pred.numpy()
    label_true = label_true.numpy()
    # spacing = spacing.numpy()

    ASD_list = np.zeros(C - 1)
    HD_list = np.zeros(C - 1)

    for i in range(C - 1):
        # tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
        tmp_surface = metrics.compute_surface_distances(label_true == (i + 1), label_pred == (i + 1), [1, 2])
        dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2

        HD = metrics.compute_robust_hausdorff(tmp_surface, 95)
        HD_list[i] = HD

    return ASD_list, HD_list


def calculate_dice(pred, target, C):
    # pred and target are torch tensor
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.)

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.)

    intersection = pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)

    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.to(pred.device)
    dice = 2 * intersection / summ        #intersection: TP, summ:FN+TP+TP+FP


    return dice, intersection, summ


# 测试代码test
def calculate_score(pred, target, C):
    # pred and target are torch tensor
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.)
    targ_num = target_mask.sum(0).type(torch.float32)  # 得到数据中每类的数量

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.)
    pred_num = pred_mask.sum(0).type(torch.float32)  # 预测数据中每类的数量


    intersection = pred_mask * target_mask  # 得到各类分类正确的数量
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)  # 求每一列的和
    summ = summ.sum(0).type(torch.float32)

    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.to(pred.device)
    dice = 2 * intersection / summ      #intersection: TP, summ:FN+TP+TP+FP
    # IoU = intersection/(summ-intersection)
    IoU = dice/(2-dice)
    R = intersection/targ_num   # R=TP/(TP+FP)
    P = intersection/pred_num   # P=TP/(TP+FN)
    F = 2*P*R/(P+R)
    ACC = 100. * intersection.sum(0) / targ_num.sum(0)

    return dice, IoU,R,P,F,ACC,intersection, summ






