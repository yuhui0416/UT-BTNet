import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2 as cv
import yaml
import math
import random
import pdb
from training import augmentation

class CMRDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=5, k=0, seed=0):

        self.mode = mode
        self.args = args

        assert mode in ['train', 'test']

        # with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
        #     img_name_list = yaml.load(f, Loader=yaml.SafeLoader)

        with open(os.path.join(args.data_root, 'dataset.txt'), 'r') as f:
            img_name_list = f.read()
        # print(img_name_list)
        img_name_list = img_name_list.split('\n')
        random.Random(seed).shuffle(img_name_list)

        length = len(img_name_list)
        test_name_list = img_name_list[k*(length//k_fold):(k+1)*(length//k_fold)]
        train_name_list = list(set(img_name_list) - set(test_name_list))

        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list
        
        print('start loading %s data'%self.mode)
        
        path = args.data_root

        img_list = []
        lab_list = []
        spacing_list = []

        for name in img_name_list:
            img_name = './data/image/' + name + '.png'
            lab_name = './data/label/' + name + '.png'

            tem_img = cv.imread(img_name, 0)
            tem_lab = cv.imread(lab_name, 0)
            tem_lab = tem_lab//255

            img, lab = self.preprocess(tem_img, tem_lab)

            img_list.append(img)
            lab_list.append(lab)

            # for idx in [0, 1]:
            #
            #     img_name = name + '_%d.nii.gz'%idx
            #     lab_name = name + '_%d_gt.nii.gz'%idx
            #
            #     itk_img = sitk.ReadImage(os.path.join(path, img_name))
            #     itk_lab = sitk.ReadImage(os.path.join(path, lab_name))
            #
            #     spacing = np.array(itk_lab.GetSpacing()).tolist()
            #     spacing_list.append(spacing[::-1])
            #
            #     assert itk_img.GetSize() == itk_lab.GetSize()
            #
            #     img, lab = self.preprocess(itk_img, itk_lab)
            #
            #     img_list.append(img)
            #     lab_list.append(lab)

        self.img_slice_list = img_list
        self.lab_slice_list = lab_list
        # self.spacing_list = spacing_list
       
        # self.img_slice_list = []
        # self.lab_slice_list = []
        # if self.mode == 'train':
        #     for i in range(len(img_list)):
        #         tmp_img = img_list[i]
        #         tmp_lab = lab_list[i]
        #
        #         z, x, y = tmp_img.shape
        #
        #         for j in range(z):
        #             self.img_slice_list.append(tmp_img[j])
        #             self.lab_slice_list.append(tmp_lab[j])
        #
        # else:
        #     self.img_slice_list = img_list
        #     self.lab_slice_list = lab_list
        #     self.spacing_list = spacing_list

        print('load done, length of dataset:', len(self.img_slice_list))
        
    def __len__(self):
        return len(self.img_slice_list)

    def preprocess(self, itk_img, itk_lab):
        
        # img = sitk.GetArrayFromImage(itk_img)
        # lab = sitk.GetArrayFromImage(itk_lab)
        img = itk_img
        lab = itk_lab

        max98 = np.percentile(img, 98)
        img = np.clip(img, 0, max98)

            
        # z, y, x = img.shape
        # if x < self.args.training_size[0]:
        #     diff = (self.args.training_size[0] + 10 - x) // 2
        #     img = np.pad(img, ((0,0), (0,0), (diff, diff)))
        #     lab = np.pad(lab, ((0,0), (0,0), (diff,diff)))
        # if y < self.args.training_size[1]:
        #     diff = (self.args.training_size[1] + 10 -y) // 2
        #     img = np.pad(img, ((0,0), (diff, diff), (0,0)))
        #     lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))

        y, x = img.shape
        if x < self.args.training_size[0]:
            diff = (self.args.training_size[0] + 10 - x) // 2
            img = np.pad(img, ((0, 0), (diff, diff)))
            lab = np.pad(lab, ((0, 0), (diff, diff)))
        if y < self.args.training_size[1]:
            diff = (self.args.training_size[1] + 10 - y) // 2
            img = np.pad(img, ((diff, diff), (0, 0)))
            lab = np.pad(lab, ((diff, diff), (0, 0)))

        img = img / max98

        img = img.astype(np.float32)
        lab = lab.astype(np.uint8)

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab


    def __getitem__(self, idx):
        tensor_img = self.img_slice_list[idx]
        tensor_lab = self.lab_slice_list[idx]
        
       
        if self.mode == 'train':
            # tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
            # tensor_lab = tensor_lab.unsqueeze(0).unsqueeze(0)
            tensor_img = tensor_img.unsqueeze(0)
            tensor_lab = tensor_lab.unsqueeze(0)
          
            # Gaussian Noise
            # tensor_img = augmentation.gaussian_noise(tensor_img, std=self.args.gaussian_noise_std)
            # Additive brightness
            # tensor_img = augmentation.brightness_additive(tensor_img, std=self.args.additive_brightness_std)
            # gamma
            # tensor_img = augmentation.gamma(tensor_img, gamma_range=self.args.gamma_range, retain_stats=True)
            #不扩增
            # tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_2d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)

            # tensor_img, tensor_lab = augmentation.crop_2d(tensor_img, tensor_lab, self.args.training_size, mode='random')

            # tensor_img, tensor_lab = tensor_img.squeeze(0), tensor_lab.squeeze(0)
        else:
            # tensor_img, tensor_lab = self.center_crop(tensor_img, tensor_lab)
            pass
        
        assert tensor_img.shape == tensor_lab.shape
        
        if self.mode == 'train':
            return tensor_img, tensor_lab
        else:
            # return tensor_img, tensor_lab, np.array(self.spacing_list[idx])
            return tensor_img, tensor_lab

    def center_crop(self, img, label):
        D, H, W = img.shape

        diff_H = H - self.args.training_size[0]
        diff_W = W - self.args.training_size[1]

        rand_x = diff_H // 2
        rand_y = diff_W // 2

        croped_img = img[:, rand_x:rand_x+self.args.training_size[0], rand_y:rand_y+self.args.training_size[0]]
        croped_lab = label[:, rand_x:rand_x+self.args.training_size[1], rand_y:rand_y+self.args.training_size[1]]

        return croped_img, croped_lab
