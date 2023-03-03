import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import xlwt
import random
from metric.utils import calculate_score
from load_model import preprocess
import torch
import numpy
import xlrd
from xlutils import copy


# General util function to get the boundary of a binary mask.
# 该函数用于获取二进制 mask 的边界
def mask_to_boundary(mask, dilation_ratio=0.001):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    mask = mask.astype(np.uint8)
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)

    # 因为之前向四周填充了0, 故而这里不再需要四周
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]

    # G_d intersects G in the paper.
    return mask - mask_erode


#
def boundary_iou(gt, dt, dilation_ratio=0.001):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)

    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou, gt_boundary, dt_boundary
