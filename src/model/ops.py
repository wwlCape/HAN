'''EoctConv'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import pdb

BN_MOMENTUM = 0.1
class EoctConv(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True, name=None):
        super(EoctConv, self).__init__()
        self.stride = stride
        #input channels
        if type(in_channels) is tuple and len(in_channels)==3:
            in_h, in_l ,in_ll= in_channels
        elif type(in_channels) is tuple and len(in_channels)==2:
            in_h, in_l = in_channels
            in_ll = None
        else:
            in_h, in_l ,in_ll= (in_channels, None, None)
        #output channels
        if type(num_channels) is tuple and len(num_channels)==3:
            num_high, num_low, num_ll = num_channels
        elif type(num_channels) is tuple and len(num_channels)==2:
        #pdb.set_trace()
            num_high, num_low = num_channels
            num_ll = 0
        else:
            num_high, num_low, num_ll = (num_channels, 0, 0)
        self.num_high = num_high
        self.num_low = num_low
        self.num_ll = num_ll
        if in_h is not None:
            self.conv2d1 = nn.Conv2d(in_h, num_high, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_high > 0 else None
            self.conv2d2 = nn.Conv2d(in_h, num_low, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_low > 0 else None
            self.conv2d3 = nn.Conv2d(in_h, num_ll, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_ll > 0 else None
        if in_l is not None:
            self.conv2d4 = nn.Conv2d(in_l, num_low, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_low > 0 else None
            self.conv2d5 = nn.Conv2d(in_l, num_high, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_high > 0 else None
            self.conv2d6 = nn.Conv2d(in_l, num_ll, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_ll > 0 else None
        if in_ll is not None:
            self.conv2d7 = nn.Conv2d(in_ll, num_ll, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_ll > 0 else None
            self.conv2d8 = nn.Conv2d(in_ll, num_high, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_high > 0 else None
            self.conv2d9 = nn.Conv2d(in_ll, num_low, kernel_size=3, stride=1, padding=1, bias=bias) if self.num_low > 0 else None
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.pooling1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pooling2 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.constant(m.bias,0)
    def forward(self, data):
        #pdb.set_trace()
        stride = self.stride
        
        #input channels
        if type(data) is tuple and len(data)==3:
            data_h, data_l ,data_ll= data
        elif type(data) is tuple and len(data)==2:
            data_h, data_l = data
            data_ll = None
        else:
            data_h, data_l ,data_ll= (data, None, None)
        data_h2l, data_h2h, data_h2ll, data_l2l, data_l2h, data_l2ll,data_ll2ll, data_ll2h, data_ll2l= None, None, None, None, None, None, None, None, None
        
        
        if data_h is not None:
            # High -> High
            data_h = self.pooling1(data_h) if stride == 2 else data_h
            data_h2h = self.conv2d1(data_h) if self.num_high > 0 else None
            # High -> Low
            data_h2l = self.pooling1(data_h) if (self.num_low > 0) else data_h
            data_h2l = self.conv2d2(data_h2l) if self.num_low > 0 else None
            # High -> Lower
            data_h2ll = self.pooling2(data_h) if (self.num_ll > 0) else data_h
            data_h2ll = self.conv2d3(data_h2ll) if self.num_ll > 0 else None
            
        
        '''processing low frequency group'''
        if data_l is not None:
            # Low -> Low
            data_l2l = self.pooling1(data_l) if (self.num_low > 0 and stride == 2) else data_l
            data_l2l = self.conv2d4(data_l2l) if self.num_low > 0 else None
            # Low -> High
            data_l2h = self.conv2d5(data_l) if self.num_high > 0 else data_l
            data_l2h = self.upsample1(data_l2h) if (self.num_high > 0 and stride == 1) else None
            #Low -> Lower
            data_l2ll = self.pooling1(data_l) if (self.num_ll > 0) else data_l
            data_l2ll = self.conv2d6(data_l2ll) if self.num_ll > 0 else None
    
        '''processing lower frequency group'''
        if data_ll is not None:
            # Lower -> Lower
            data_ll2ll = self.pooling1(data_ll) if (self.num_ll > 0 and stride == 2) else data_ll
            data_ll2ll = self.conv2d7(data_ll2ll) if self.num_ll > 0 else None
            # Lower -> High
            data_ll2h = self.conv2d8(data_ll) if self.num_high > 0 else data_ll
            data_ll2h = self.upsample2(data_ll2h) if (self.num_high > 0 and stride == 1) else None
            #data_ll2h = upsample3(data_ll2h) if (num_high > 0 and stride == 1) else None
            #Lower -> Low
            data_ll2l = self.conv2d9(data_ll) if self.num_low > 0 else data_ll
            data_ll2l = self.upsample1(data_ll2l) if (self.num_low > 0 and stride == 1) else None
            
        '''you can force to disable the interaction paths'''
        # data_h2l = None if (data_h2h is not None) and (data_l2l is not None) else data_h2l
        # data_l2h = None if (data_h2h is not None) and (data_l2l is not None) else data_l2h

        #output = ElementWiseSum(*[(data_h2h, data_h2l, data_h2ll), (data_l2h, data_l2l, data_l2ll), (data_ll2h, data_ll2l, data_ll2ll)], name=name)
        #pdb.set_trace()
        output = (dataSum(dataSum(data_h2h, data_l2h), data_ll2h), dataSum(dataSum(data_h2l, data_l2l), data_ll2l), dataSum(dataSum(data_h2ll, data_l2ll) ,data_ll2ll))
        #output = torch.from_numpy(np.array(output))
        # squeeze output (to be backward compatible)
        if output[2] is None:
            if output[1] is None:
                return output[0]
            else:
                return output[0:2]
        elif output[1] is None:
            return output[0::2]
        else:
            return output
        
def relu(data):
    relu = nn.ReLU(inplace=True)
    if type(data) is tuple and len(data)==3:
        out = (relu(data[0]), relu(data[1]), relu(data[2]))
        return out
        
    elif type(data) is tuple and len(data)==2:
        if data[0] is None:
            out = (relu(data[1]), relu(data[2]))
            return out
        elif data[1] is None:
            out = (relu(data[0]), relu(data[2]))
            return out
        else:
            out = (relu(data[0]), relu(data[1]))
            return out
    else:
        out = relu(data)
        return out
        
def sigmoid(data):
    if type(data) is tuple and len(data)==3:
        out = (F.sigmoid(data[0]), F.sigmoid(data[1]), F.sigmoid(data[2]))
        return out
        
    elif type(data) is tuple and len(data)==2:
        if data[0] is None:
            out = (F.sigmoid(data[1]), F.sigmoid(data[2]))
            return out
        elif data[1] is None:
            out = (F.sigmoid(data[0]), F.sigmoid(data[2]))
            return out
        else:
            out = (F.sigmoid(data[0]), F.sigmoid(data[1]))
            return out
    elif type(data) is Tensor:
        out = F.sigmoid(data)
        return out

def bn(data, num_channels):
    if type(data) is tuple and len(data)==3:
        bn1 = nn.BatchNorm2d(num_channels[0], momentum=BN_MOMENTUM)
        bn2 = nn.BatchNorm2d(num_channels[1], momentum=BN_MOMENTUM)
        bn3 = nn.BatchNorm2d(num_channels[2], momentum=BN_MOMENTUM)
        out = (bn1(data[0]), bn2(data[1]), bn3(data[2]))
        return out
    elif type(data) is tuple and len(data)==2:
        bn1 = nn.BatchNorm2d(num_channels[0], momentum=BN_MOMENTUM)
        bn2 = nn.BatchNorm2d(num_channels[1], momentum=BN_MOMENTUM)
        out = (bn1(data[0]), bn2(data[1]))
        return out
    elif type(data) is Tensor:
        bn1 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        out = bn1(data)
        return out
    
def max_pool2d(data, l=(2,2)):
    if type(data) is tuple and len(data)==3:
        out = (F.max_pool2d(data[0], l), F.max_pool2d(data[1], l), F.max_pool2d(data[2], l))
        return out
        
    elif type(data) is tuple and len(data)==2:
        if data[0] is None:
            out = (F.max_pool2d(data[1], l), F.max_pool2d(data[2], l))
            return out
        elif data[1] is None:
            out = (F.max_pool2d(data[0], l), F.max_pool2d(data[2], l))
            return out
        else:
            out = (F.max_pool2d(data[0], l), F.max_pool2d(data[1], l))
            return out
    elif type(data) is Tensor:
        out = F.max_pool2d(data, l)
        return out
        
def avg_pool2d(data):
    avg_pool = nn.AdaptiveAvgPool2d(1)
    if type(data) is tuple and len(data)==3:
        out = (avg_pool(data[0]), avg_pool(data[1]), avg_pool(data[2]))
        return out
        
    elif type(data) is tuple and len(data)==2:
        if data[0] is None:
            out = (avg_pool(data[1]), avg_pool(data[2]))
            return out
        elif data[1] is None:
            out = (avg_pool(data[0]), avg_pool(data[2]))
            return out
        else:
            out = (avg_pool(data[0]), avg_pool(data[1]))
            return out
    elif type(data) is Tensor:
        out = avg_pool(data)
        return out

def dropout(data, l):
    Dropout = nn.Dropout(l)
    if type(data) is tuple and len(data)==3:
        out = (Dropout(data[0]), Dropout(data[1]), Dropout(data[2]))
        return out
        
    elif type(data) is tuple and len(data)==2:
        if data[0] is None:
            out = (Dropout(data[1]), Dropout(data[2]))
            return out
        elif data[1] is None:
            out = (Dropout(data[0]), Dropout(data[2]))
            return out
        else:
            out = (Dropout(data[0]), Dropout(data[1]))
            return out
    elif type(data) is Tensor:
        out = Dropout(data)
        return out
        
def dataSum(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        assert a.size()==b.size()
        return a+b



def tupleSum(a,b):
    out = (a[0]+b[0],a[1]+b[1],a[2]+b[2])
    return(out)
        
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        
class _UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, 
                 group=1):
        super(_UpsampleBlock, self).__init__()
        '''
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)'''
        #init_weights(self.modules)
        self.conv1 = nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group)
        self.conv2 = nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group)
        self.relu = nn.ReLU(inplace=True)
        self.pixelshuffle = nn.PixelShuffle(2)
        
    def forward(self, x):
        #out = self.body(x)
        out = self.conv1(x)
        #pdb.set_trace()
        out = self.relu(out)
        out = self.pixelshuffle(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pixelshuffle(out)
        #print(out.shape)

        return out

def tupleMultiply(a, b):
    out=[]
    assert type(b) is int
    for i in range(len(a)):
        out.append(a[i]*b)

    return tuple(out)
