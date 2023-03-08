import cv2
import numpy as np
import gudhi as gd
import torch
import torch.nn.functional as F
from torch import Tensor
import os
from tqdm import tqdm
from PIL import Image
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from inference.utils import get_inference
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from metric.utils import calculate_dice
from metric.utils import calculate_score

from training.utils import update_ema_variables
from training.losses import DiceLoss, FocalLoss
from training.validation import validation
from training.utils import exp_lr_scheduler_with_warmup, log_evaluation_result, get_optimizer
from training.topoloss_pytorch import getTopoLoss
from model.sobel import pre_edge
import yaml
import argparse
import time
import math
import os
import sys
import pdb
import warnings
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import matplotlib.pyplot as plt
from Betti_Compute.betti_compute import betti_error
import metric.BIOU

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Conv-Trans Segmentation')
    parser.add_argument('--dataset', type=str, default='coronary', help='dataset name')
    parser.add_argument('--model', type=str, default='UT-BTNet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')

    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='UT-BTNet', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    config_path = 'config/%s/%s_%s_%s.yaml' % (args.dataset, args.model,'bd', 'top')
    # config_path = 'config/%s/%s_%s_%s.yaml'%(args.dataset, args.model, 'cornary', 'top')
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s" % config_path)

    print('Loading configurations from %s' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        # net = torch.load(args.load)
        print('Model loaded from {}'.format(args.load))

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
    else:
        ema_net = None
    return net, ema_net


def preprocess(itk_img, itk_lab):
    img = itk_img
    lab = itk_lab

    max98 = np.percentile(img, 98)
    img = np.clip(img, 0, max98)

    y, x = img.shape
    if x < 512:
        diff = (self.args.training_size[0] + 10 - x) // 2
        img = np.pad(img, ((0, 0), (diff, diff)))
        lab = np.pad(lab, ((0, 0), (diff, diff)))
    if y < 512:
        diff = (self.args.training_size[1] + 10 - y) // 2
        img = np.pad(img, ((diff, diff), (0, 0)))
        lab = np.pad(lab, ((diff, diff), (0, 0)))

    img = img / max98

    img = img.astype(np.float32)
    lab = lab.astype(np.uint8)

    # tensor_img = torch.from_numpy(img).float()
    # tensor_lab = torch.from_numpy(lab).long()

    return img, lab

if __name__ == '__main__':
    model_path = './checkpoint/best.pth'
    data_path = './data/coronary'
    save_path = os.path.join('./preimage',  model_path.split("/")[-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_path = os.path.join(data_path, 'image')
    label_path = os.path.join(data_path, 'label')
    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)
    N = len(image_list)

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.ema = True
    args.load = model_path
    # args.pretrain = False
    args.pretrain = True

    # print(args)

    net, ema_net = init_network(args)
    net.cuda()
    # print(net)
    inference = get_inference(args)
    net.eval()
    params = list(net.named_parameters())
    for n in range(params.__len__()):
        print(params[n])

    with torch.no_grad():
        for i,name in enumerate(image_list):

            image = cv2.imread(os.path.join(image_path, name), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            label = label//255
            H = np.array(image).shape[0]
            W = np.array(image).shape[1]

            img, lab = preprocess(image, label)
            image = torch.tensor(img).float().cuda()
            label = torch.tensor(lab).long().cuda()
            inputs = image.unsqueeze(0)
            inputs = inputs.unsqueeze(0)
            # print(list(inputs.size()))
            # pred = inference(net, inputs, args)
            pred = net(inputs)
            ##dedcgcnee
            # y = pre_edge(inputs)
            # pred = net(inputs,y)

            # pre = F.softmax(pred,dim =1)
            # pre = pre[:, 1, :, :]
            # pre = pred.squeeze(0)
            _, label_pred = torch.max(pred, dim=1)
            pr = label_pred.cpu().numpy()
            colors = [(0, 0, 0), (255, 255, 255)]
            seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [H, W, -1])

            pre_image = Image.fromarray(np.uint8(seg_img))
            # label = label.unsqueeze(0)

            #####保存图片
            pre_image.save(os.path.join(save_path, name))

            # 计算DICE系数
            label_pred = label_pred.squeeze(0)
            # label = label.unsqueeze(0)

            dice,IoU,_,_,_,_, _, _ = calculate_score(label_pred.view(-1, 1), label.view(-1, 1), args.classes)
            # if dice.cpu().numpy()[1] > 0.95:
            #     pre_image.save(os.path.join(save_path, name))
                # print('Test number: %d, name：%s, dice: %f, IoU: %f, recall: %f, precision: %f, F1: %f, accuracy: %f' %
                #            (i+1, name, dice.cpu().numpy()[1], IoU.cpu().numpy()[1], R.cpu().numpy()[1],P.cpu().numpy()[1],F.cpu().numpy()[1],ACC))
            gt = cv2.imread(os.path.join(label_path, name), cv2.IMREAD_GRAYSCALE)
            pr = cv2.imread(os.path.join(save_path, name), cv2.IMREAD_GRAYSCALE)
            biou, _, _ = metric.BIOU.boundary_iou(gt, pr, dilation_ratio=0.001)
            Betti_error = betti_error(os.path.join(label_path, name), os.path.join(save_path, name))
            print('Test number: %d, dice: %f,IoU: %f, BIoU: %f, Betti_error: %f' %
                  (i+1, dice.cpu().numpy()[1], IoU.cpu().numpy()[1], biou, Betti_error))
        #     mdice+=dice.cpu().numpy()[1:]
        #     mIoU+=IoU.cpu().numpy()[1:]
        #     mR+=R.cpu().numpy()[1:]
        #     mP+=P.cpu().numpy()[1:]
        #     mF+=F.cpu().numpy()[1:]
        #     mACC+=ACC
        #
        # dice = mdice/(i+1)
        # IoU = mIoU/(i+1)
        # R = mR/(i+1)
        # P = mP/(i+1)
        # F = mF/(i+1)
        # ACC = mACC/(i+1)
        #
        # print('mean_dice: %f,mean_IoU: %f, mean_recall: %f, mean_precision: %f, mean_F1: %f,mean_accuracy: %f' %(dice,IoU,R,P,F,ACC))








