import numpy as np
import torch
import torch.nn as nn
import pdb
from .conv_layers import BasicBlock, Bottleneck, SingleConv, MBConv, FusedMBConv, ConvNeXtBlock

def get_model(args, pretrain=False):
    
    if args.dimension == '2d':
        if args.model in ['UT-BTNet']:
            from .utbtnet import UTBTNet
            if pretrain:
                # raise ValueError('No pretrain model available')
                pass
            return UTBTNet(args.in_chan, args.classes, args.base_chan, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=nn.GELU, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop)



    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')

def get_block(name):
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'MBConv': MBConv,
        'FusedMBConv': FusedMBConv,
        'ConvNeXtBlock': ConvNeXtBlock
    }
    return block_map[name]