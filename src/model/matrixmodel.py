# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model import ops
import pdb


try:
    from model.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResBlock(nn.Module):
    def __init__(
        self, num_channels, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,**kwargs):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=bias))
            if bn: m.append(nn.BatchNorm2d(num_channels))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        initialize_weights([self.body], 0.1)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class BFN(nn.Module):
    def __init__(self, num_channels, kernel_size, reduction, n_blocks, block):
        super(BFN, self).__init__()

        branch1=[]
        branch1.append(self._make_blocks(num_channels[0], num_channels[0], kernel_size, reduction, n_blocks, block))
        branch1.append(nn.Conv2d(num_channels[0], num_channels[0], kernel_size, stride=1, padding=1, bias=True))
        branch2=[]
        branch2.append(self._make_blocks(num_channels[1], num_channels[1], kernel_size, reduction, n_blocks, block))
        branch2.append(nn.Conv2d(num_channels[1], num_channels[1], kernel_size, stride=1, padding=1, bias=True))
        branch3=[]
        branch3.append(self._make_blocks(num_channels[2], num_channels[2], kernel_size, reduction, n_blocks, block))
        branch3.append(nn.Conv2d(num_channels[2], num_channels[2], kernel_size, stride=1, padding=1, bias=True))
        self.branch1 = nn.Sequential(*branch1)
        self.branch2 = nn.Sequential(*branch2)
        self.branch3 = nn.Sequential(*branch3)
        #self.act=nn.ReLU(True)


    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) \
            for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        
        return nn.Sequential(*blocks)

    def forward(self, x):
        assert type(x) is tuple and len(x)==3
        #branch1
        res1 = x[0]
        out1 = self.branch1(x[0])
        out1 += res1

        #branch2
        res2 = x[1]
        out2 = self.branch2(x[1])
        out2 += res2

        #branch3
        res3 = x[2]
        out3 = self.branch3(x[2])
        out3 += res3

        return (out1,out2,out3)

class BFN1(nn.Module):
    def __init__(self, num_channels, kernel_size, reduction, n_blocks, block):
        super(BFN1, self).__init__()

        branch1=[]
        branch1.append(self._make_blocks(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        branch1.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        self.branch1 = nn.Sequential(*branch1)
        #self.act=nn.ReLU(True)


    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) \
            for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        
        return nn.Sequential(*blocks)

    def forward(self, x):
        #branch1
        res1 = x
        out1 = self.branch1(x)
        out1 += res1

        return out1

class BFN2(nn.Module):
    def __init__(self, num_channels, kernel_size, reduction, n_blocks, block):
        super(BFN2, self).__init__()

        branch1=[]
        branch1.append(self._make_blocks(num_channels[0], num_channels[0], kernel_size, reduction, n_blocks, block))
        branch1.append(nn.Conv2d(num_channels[0], num_channels[0], kernel_size, stride=1, padding=1, bias=True))
        branch2=[]
        branch2.append(self._make_blocks(num_channels[1], num_channels[1], kernel_size, reduction, n_blocks, block))
        branch2.append(nn.Conv2d(num_channels[1], num_channels[1], kernel_size, stride=1, padding=1, bias=True))
        self.branch1 = nn.Sequential(*branch1)
        self.branch2 = nn.Sequential(*branch2)
        #self.act=nn.ReLU(True)


    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) \
            for _ in range(n_blocks)]
        blocks.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=1, bias=True))
        
        return nn.Sequential(*blocks)

    def forward(self, x):
        assert type(x) is tuple and len(x)==2
        #branch1
        res1 = x[0]
        out1 = self.branch1(x[0])
        out1 += res1

        #branch2
        res2 = x[1]
        out2 = self.branch2(x[1])
        out2 += res2

        return (out1,out2)

class EoctResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, num_channels, stride=1, downsample=None, res_scale=1, **kwargs):
        super(EoctResBlock, self).__init__()
        self.num_channels = num_channels # (64,64,64)
        self.stride = stride
        self.downsample = downsample
        self.res_scale = res_scale
        self.conv1 = ops.EoctConv(in_channels, num_channels, stride=stride)
        self.conv2 = ops.EoctConv(num_channels, num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = ops.bn(out, self.num_channels)
        out = ops.relu(out)

        out = self.conv2(out)
        #out = ops.bn(out, self.num_channels)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        #out = out * self.res_scale + residual
        out = ops.tupleSum(out,residual)
        #pdb.set_trace()
        out = ops.relu(out)

        return out

class EoctBottleneck(nn.Module):
    def __init__(self, in_channels, num_channels, stride=1, downsample=None, res_scale=1, **kwargs):
        super(EoctBottleneck, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.downsample = downsample
        self.res_scale = res_scale
        expand = 6
        linear = 0.8
        self.conv1 = ops.EoctConv(in_channels, ops.tupleMultiply(num_channels,expand), kernel_size=1, padding=1//2)
        #self.bn1 = nn.BatchNorm2d(num_channels*expand, momentum=BN_MOMENTUM)
        self.conv2 = ops.EoctConv(ops.tupleMultiply(num_channels,expand), int(ops.tupleMultiply(num_channels,linear)), kernel_size=1, padding=1//2)
        self.conv3 = ops.EoctConv(int(ops.tupleMultiply(num_channels,linear)), num_channels, kernel_size=3, padding=kernel_size//2)
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = ops.bn(out, self.num_channels)
        out = ops.relu(out)

        out = self.conv2(out)
        #out = ops.bn(out, self.num_channels)
        
        out = self.conv3(out)
        #out = ops.bn(out, self.num_channels)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        #out = out * self.res_scale + residual
        out = ops.tupleSum(out,residual)
        out = ops.relu(out)

        return out
        

class CALayer(nn.Module):
    def __init__(self, in_channels, num_channels, reduction=16):
        super(CALayer, self).__init__()
        
        # feature channel downscale and upscale --> channel weight
        self.conv1 = ops.EoctConv(in_channels, num_channels // reduction, 1, padding=0, bias=True),
        self.conv2 = ops.EoctConv(num_channels // reduction, num_channels, 1, padding=0, bias=True),


    def forward(self, x):
    
        out = ops.avg_pool2d(x)
        
        out = self.conv1(out)
        out = ops.relu(out)
        out = self.conv2(out)
        out = ops.sigmoid(out)
        
        return x * out

class CAEoctResBlock(nn.Module):
    def __init__(self, in_channels, num_channels, reduction, bias=True, res_scale=1, **kwargs):
        super(CAEoctResBlock, self).__init__()
        self.num_channels = num_channels # [64,64,64,64]
        self.res_scale = res_scale
        self.conv1 = ops.EoctConv(in_channels, num_channels, stride=stride)
        self.conv2 = ops.EoctConv(num_channels, num_channels)
        self.caLayer = CAEctBlock(num_channels, num_channels, reduction)
        
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = ops.relu(out)
        out = self.conv2(out)
        
        out = self.caLayer(out)
        
        out = ops.tupleSum(out,res)
        #out = out * self.res_scale + res
        out = ops.relu(out)
        
        
        return out

blocks_dict = {
    'BASIC':ResBlock,
    'EctBASIC': EoctResBlock,
    'EctBOTTLENECK': EoctBottleneck,
    'CAEctBASIC':CAEoctResBlock
}

def make_model(args, parent=False):
    return MatrixModelG2(args)

class MatrixModel(nn.Module):
    def __init__(self, args):
        super(MatrixModel, self).__init__()
        
        n_groups = args.n_resgroups
        n_blocks = args.n_resblocks
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = EoctResBlock
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = ops.EoctConv(3, 64)
        
        modules_body1 = []
        modules_body1.append(self._make_blocks(64, 64, kernel_size, reduction, n_blocks, block))
        modules_body1.append(ops.EoctConv(64, (64,64), kernel_size))
        
        modules_body2 = []
        modules_body2.append(self._make_blocks((64,64), (64,64), kernel_size, reduction, n_blocks, block))
        modules_body2.append(ops.EoctConv((64,64), num_channels, kernel_size))
        
        modules_body3 = []
        modules_body3.append(self._make_blocks(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body3.append(ops.EoctConv(num_channels, 64, kernel_size))
        
        modules_tail = [
            ops._UpsampleBlock(num_channels[0], scale=scale),
            nn.Conv2d(num_channels[0], 3, kernel_size, 1, 1)]
        
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        
    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) \
            for _ in range(n_blocks)]
        blocks.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)

        res = x
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x += res
        #pdb.set_trace()

        out = self.tail(x)
        out = self.add_mean(out)
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class RERB(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        super(RERB, self).__init__()

        blocks = []
        blocks.append(self._make_blocks(in_channels, num_channels, kernel_size, reduction, n_blocks, block))
        blocks.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        self.body = nn.Sequential(*blocks)

    def _make_blocks(self, in_channels, num_channels, kernel_size, reduction, n_blocks, block):
        blocks = []
        blocks = [block(in_channels=in_channels, num_channels=num_channels, reduction=reduction) \
            for _ in range(n_blocks)]
        blocks.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        
        return nn.Sequential(*blocks)

    def forward(self, x):
        res = x
        x = self.body(x)
        x = ops.tupleSum(x,res)
        x = ops.relu(x)

        return x


class MatrixModelB(nn.Module):
    def __init__(self, args):
        super(MatrixModelB, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        self.fusion_conv1 = ops.EoctConv(num_channels, num_channels, kernel_size)
        self.fusion_conv2 = ops.EoctConv(num_channels, num_channels, kernel_size)
        self.fusion_conv3 = ops.EoctConv(num_channels, num_channels, kernel_size)
        self.conv_last = ops.EoctConv(num_channels, 64, kernel_size)
        
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()
        L1_fea = x[0]

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)
        L2_fea = x[1]

        #stage3
        x = self.stage3(x)
        out = self.stage3_conv(x)
        L3_fea = x[2]
        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        x = (L1_fea, L2_fea, L3_fea)
        res1 = x
        x = self.fusion_conv1(x)
        x = ops.tupleSum(x,res1)
        res2 = x
        x = self.fusion_conv2(x)
        x = ops.tupleSum(x,res2)
        res3 = x
        x = self.fusion_conv3(x)
        x = ops.tupleSum(x,res3)
        out = self.conv_last(x)
        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class PDF(nn.Module):
    ''' Alignment module using Pyramid, Deformable convolution and Fusion.
    with 3 pyramid levels.
    Bottom-Up.
    '''

    def __init__(self, nf=64, groups=8):
        super(PDF, self).__init__()
        # L1: level 1, original spatial size
        #self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L3_offset_conv3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L3_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv_last = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)

    def forward(self, nbr_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L1
        L1_offset = nbr_fea_l[0]
        #L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(L1_offset))
        L1_fea = self.lrelu(self.L1_dcnpack([nbr_fea_l[0], L1_offset]))
        L1_f = L1_fea
        # L2
        L2_offset = nbr_fea_l[1]
        L1_offset = self.lrelu(self.L2_offset_conv1(L1_offset))
        #L1_offset = F.interpolate(L1_offset, scale_factor=1/2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L1_offset * 2], dim=1)))
        #L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L1_fea = self.lrelu(self.L2_offset_conv3(L1_fea))
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L1_fea], dim=1)))
        L2_f = L2_fea
        # L3
        L3_offset = nbr_fea_l[2]
        #L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L2_offset = self.L3_offset_conv1(L2_offset)
        L3_offset = self.lrelu(self.L3_offset_conv2(torch.cat([L3_offset, L2_offset * 2], dim=1)))
        #L3_offset = self.lrelu(self.L3_offset_conv3(L3_offset))
        L3_fea = self.L3_dcnpack([nbr_fea_l[2], L3_offset])
        L2_fea = self.lrelu(self.L3_offset_conv3(L2_fea))
        L3_fea = self.L3_fea_conv(torch.cat([L3_fea, L2_fea], dim=1))
        # Fusion
        L3_fea = self.upsample2(L3_fea)
        L2_f = self.upsample(L2_f)
        L_fea = torch.cat([torch.cat([L1_f, L2_f], dim=1),L3_fea],dim=1)
        L_fea = self.lrelu(self.conv_last(L_fea))
        return L_fea

class PD(nn.Module):
    ''' module using Pyramid, Deformable convolution
    with 3 pyramid levels.
    Top-down.
    '''

    def __init__(self, nf=64, groups=8):
        super(PD, self).__init__()
        # L3: level 3, 1/4 spatial size
        #self.L3_offset_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        #self.L2_offset_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        #self.L1_offset_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        #self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        #self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        #self.conv_last = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)

    def forward(self, nbr_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = nbr_fea_l[2]
        #L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        L3_f = L3_fea
        # L2
        L2_offset = nbr_fea_l[1]
        #L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = self.upsample(L3_offset)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = self.upsample(L3_fea)
        #pdb.set_trace()
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        L2_f = L2_fea
        # L1
        L1_offset = nbr_fea_l[0]
        #L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = self.upsample(L2_offset)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = self.upsample(L2_fea)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        #offset = L1_fea
        #offset = self.lrelu(self.cas_offset_conv1(offset))
        #offset = self.lrelu(self.cas_offset_conv2(offset))
        #L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))
        
        #L3_f = self.upsample2(L3_f)
        #L2_f = self.upsample(L2_f)
        #L_fea = torch.cat([torch.cat([L1_fea, L2_f], dim=1),L3_f],dim=1)
        #L_fea = self.lrelu(self.conv_last(L_fea))

        return (L1_fea, L2_f, L3_f)

class MatrixModelC(nn.Module):
    def __init__(self, args):
        super(MatrixModelC, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        '''
        
        self.pd = PD()
        self.pdf = PDF()
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        self.tail1 = nn.Sequential(*modules_tail1)
        
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()
        L1_fea = x[0]

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)
        L2_fea = x[1]

        #stage3
        x = self.stage3(x)
        x = self.stage3_conv(x)
        L3_fea = x[2]
        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        #pdf-Two_way
        x = (L1_fea, L2_fea, L3_fea)
        x = self.pd(x)
        out = self.pdf(x)
        #long skip
        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelD(nn.Module):
    def __init__(self, args):
        super(MatrixModelD, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, num_channels, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)

        #stage3
        x = self.stage3(x)
        x = self.stage3_conv(x)
        out = x[0]
        #out3 = x[2]
        #pdb.set_trace()
        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail3(out3)
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelE(nn.Module):
    def __init__(self, args):
        super(MatrixModelE, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)

        #stage3
        x = self.stage3(x)
        out = self.stage3_conv(x)
        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelF(nn.Module):
    def __init__(self, args):
        super(MatrixModelF, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)

        #stage3
        x = self.stage3(x)
        out = self.stage3_conv(x)
        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class MatrixModelG(nn.Module):
    def __init__(self, args):
        super(MatrixModelG, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        self.last_conv = nn.Conv2d(64*3, 64, kernel_size, 1, 1)
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        out1 = x[0]
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)
        out2 = x[0]

        #stage3
        x = self.stage3(x)
        out = self.stage3_conv(x)
        out2 = torch.cat([out1,out2], dim=1)
        out = torch.cat([out2,out], dim=1)
        out = self.last_conv(out)

        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class MatrixModelG2(nn.Module):
    def __init__(self, args):
        super(MatrixModelG2, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        self.da = DAM_Module(64)
        #self.da_conv = nn.Conv2d(64*3, 64, 3, 1, 1)
        self.last_conv = nn.Conv2d(64*3, 64, kernel_size, 1, 1)
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        out1 = x[0].unsqueeze(1)
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)
        out2 = x[0].unsqueeze(1)

        #stage3
        x = self.stage3(x)
        out = self.stage3_conv(x).unsqueeze(1)
        out2 = torch.cat([out1,out2], dim=1)
        out = torch.cat([out2,out], dim=1)

        out = self.da(out)
        out = self.last_conv(out)

        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class MatrixModelF2(nn.Module):
    def __init__(self, args):
        super(MatrixModelF2, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        inter_channels = 64 #// 4
        self.conv5a = nn.Sequential(nn.Conv2d(64, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(64, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.pa = PAM_Module(inter_channels)
        self.ca = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        #self.conv6 = nn.Sequential(nn.Conv2d(192, 64, 1))
        #self.da_conv = nn.Conv2d(64*3, 64, 3, 1, 1)
        self.last_conv = nn.Conv2d(64*3, 64, kernel_size, 1, 1)
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)

        #stage3
        x = self.stage3(x)
        out1 = self.stage3_conv(x)

        feat1 = self.conv5a(out1)
        pa_feat = self.pa(feat1)
        pa_conv = self.conv51(pa_feat)
        #pa_output = self.conv6(pa_conv)

        feat2 = self.conv5c(out1)
        ca_feat = self.ca(feat2)
        ca_conv = self.conv52(ca_feat)
        #ca_output = self.conv7(ca_conv)

        feat_sum = torch.cat([pa_conv,ca_conv],dim=1)
        paca_output = torch.cat([feat_sum,out1],dim=1)

        out = self.last_conv(paca_output)

        
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class GAM_Module(nn.Module):
    """ Global attention module"""
    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (C*H*W) X (C*H*W)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, -1).unsqueeze(-1)
        proj_key = x.view(m_batchsize, -1).unsqueeze(-1).permute(0, 2, 1)
        #pdb.set_trace()
        energy = torch.bmm(proj_query, proj_key)
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1).unsqueeze(-1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DAM_Module(nn.Module):
    """ Deep attention module"""
    def __init__(self, in_dim):
        super(DAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class MatrixModelH(nn.Module):
    def __init__(self, args):
        super(MatrixModelH, self).__init__()
        
        num_channels = (64, 64, 64)
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        block = blocks_dict[args.block]
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = ops.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.first_conv = nn.Conv2d(3, 64, kernel_size, stride=1, padding=1, bias=True)
        
        modules_stage1 = []
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        modules_stage1.append(BFN1(64, kernel_size, reduction, 5, block))
        self.stage1 = nn.Sequential(*modules_stage1)
        self.stage1_conv = ops.EoctConv(64, (64,64), kernel_size)

        modules_stage2 = []
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        modules_stage2.append(BFN2((64,64), kernel_size, reduction, 5, block))
        self.stage2 = nn.Sequential(*modules_stage2)
        self.stage2_conv = ops.EoctConv((64,64), num_channels, kernel_size)

        modules_stage3 = []
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        modules_stage3.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage3 = nn.Sequential(*modules_stage3)
        self.stage3_conv = ops.EoctConv(num_channels, 64, kernel_size)
        
        self.pa = PAM_Module(64)
        self.pa_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.ca = CAM_Module(64)
        self.ca_conv = nn.Conv2d(64, 64, 3, 1, 1)
        '''
        modules_stage4 = []
        modules_stage4.append(BFN(num_channels, kernel_size, reduction, 5, block))
        self.stage4 = nn.Sequential(*modules_stage4)
        self.stage4_conv = ops.EoctConv(num_channels, num_channels, kernel_size)

        
        modules_body = []
        for i in range(n_groups):
            modules_body.append(RERB(num_channels, num_channels, kernel_size, reduction, n_blocks, block))
        modules_body.append(ops.EoctConv(num_channels, num_channels, kernel_size))
        '''
        
        modules_tail1 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail1 = nn.Sequential(*modules_tail1)
        '''
        modules_tail2 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail2 = nn.Sequential(*modules_tail2)
        
        modules_tail3 = [
            ops._UpsampleBlock(64, scale=scale),
            nn.Conv2d(64, 3, kernel_size, 1, 1)]
        
        #self.body = nn.Sequential(*modules_body)
        self.tail3 = nn.Sequential(*modules_tail3)
        '''
              
    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.first_conv(x)
        residual = x
        #pdb.set_trace()

        #stage1
        x = self.stage1(x)
        x = self.stage1_conv(x)
        #pdb.set_trace()

        #stage2
        x = self.stage2(x)
        x = self.stage2_conv(x)

        #stage3
        x = self.stage3(x)
        out = self.stage3_conv(x)
        
        #atten
        pa_out = self.pa(out)
        pa_out = self.pa_conv(pa_out)
        ca_out = self.ca(out)
        ca_out = self.ca_conv(ca_out)
        out = pa_out + ca_out
        #stage4
        #x = self.stage4(x)
        #x = self.stage4_conv(x)

        out += residual

        out = self.tail1(out)
        out = self.add_mean(out)

        #out2 = self.tail1(x[1])
        #out2 = self.add_mean(out2)

        #out3 = self.tail2(x[2])
        #out3 = self.add_mean(out3)
        #pdb.set_trace()
        
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))