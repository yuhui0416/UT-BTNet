import numpy as np
import torch

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b][c] = sdf
            assert np.min(sdf) == -1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
            assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.cpu().numpy()

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis - posdis
            sdf[boundary==1] = 0
            gt_sdf[b][c] = sdf

    return gt_sdf

def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: softmax results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:,1,...]
    dc = torch.tensor(gt_sdf[:,0,...]).cuda()
    # dc = gt_sdf[:,1,...]
    multipled = torch.einsum('bxy, bxy->bxy', pc, dc)
    bd_loss = multipled.mean()

    return bd_loss