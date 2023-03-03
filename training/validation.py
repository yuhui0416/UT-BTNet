import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice
import numpy as np
import pdb
import cv2 as cv
import gudhi as gd


def validation(net, dataloader,epoch_num, fold_idx, args):
    
    net.eval()

    l_path = 'D:\\wgp\\CBIM-Medical-Image-Segmentation-main\\checkpoint\\validation'
    l_path1 = os.path.join(l_path, str(fold_idx) + str(epoch_num))
    if not os.path.isdir(l_path1):
        os.mkdir(l_path1)
    pre_path = os.path.join(l_path1, 'pre_label')
    label_path = os.path.join(l_path1, 'label')
    if not os.path.isdir(pre_path):
        os.mkdir(pre_path)
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    dice_list = np.zeros(args.classes-1) # background is not including in validation
    ASD_list = np.zeros(args.classes-1)
    HD_list = np.zeros(args.classes-1)
    
    inference = get_inference(args)

    counter = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            # spacing here is used for distance metrics calculation
            
            inputs, labels = images.float().cuda(), labels.long().cuda()
            # print(inputs.shape)
            # print(labels.shape)
            inputs = inputs.unsqueeze(0)
            # if args.dimension == '2d':
            #     inputs = inputs.permute(1, 0, 2, 3)

            name = str(i) + '.png'#  加入保存
            label = labels.squeeze(0)
            label = label.squeeze(0)
            label = label.cpu().numpy()
            pre_name = os.path.join(pre_path, name)
            label_name = os.path.join(label_path, name)
            cv.imencode('.png', label*255)[1].tofile(label_name)#。。。。。。

            pred = inference(net, inputs, args)

            _, label_pred = torch.max(pred, dim=1)


            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
                
            
            # print(i)

            # tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # ASD_list += np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            # HD_list += np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            print('Test number: %d, dice: %f' %(i, dice.cpu().numpy()[1:]))
            dice_list += dice.cpu().numpy()[1:]

            pre_label = label_pred.squeeze(0)#加入保存
            pre_label = pre_label.cpu().numpy()
            cv.imencode('.png', pre_label*255)[1].tofile(pre_name)  # 。。。。。。

            counter += 1

    dice_list /= counter
    # ASD_list /= counter
    # HD_list /= counter

    ASD_list = [0]
    HD_list = [0]

    return dice_list, ASD_list, HD_list

