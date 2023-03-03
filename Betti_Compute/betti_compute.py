import sys
sys.path.append('./source/')
import numpy as np
import matplotlib.image as mpimg
import os
from .ext_libs import Gudhi as gdh
from skimage import color
import pandas as pd
from . import histo_image as hi
import xlwt
import xlrd
from xlutils import copy
import random




def betti_error(gt,pr):

    gt_copy = color.rgb2gray(mpimg.imread(gt))
    pr_copy = color.rgb2gray(mpimg.imread(pr))
    # gt_copy = color.rgb2gray(mpimg.imread(os.path.join(gt_path, name)))
    # pr_copy = color.rgb2gray(mpimg.imread(os.path.join(pr_path, name)))
    gt = gt_copy.copy()
    pr = pr_copy.copy()

    W,H = pr.shape
    gt[W - 1, :] = 0
    gt[:, H - 1] = 0
    gt[0, :] = 0
    gt[:, 0] = 0

    pr[W - 1, :] = 0
    pr[:, H - 1] = 0
    pr[0, :] = 0
    pr[:, 0] = 0
    gt_temp = gdh.compute_persistence_diagram(gt, i=1)
    pr_temp = gdh.compute_persistence_diagram(pr, i=1)
    gt_betti_number = len(gt_temp)
    pr_betti_number = len(pr_temp)
    err = abs(gt_betti_number-pr_betti_number)
    return err
