from model import common
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
import math

def make_model(args, parent=False):
    return RCAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class MSCALayer(nn.Module):
    def __init__(self):
        pass

class Dis(nn.Module):
    def __init__(self, loss_type='L1', B=4):
        super(Dis, self).__init__()
        self.loss_type = loss_type
        #self.loss = torch.zeros(B)
        if self.loss_type == 'cos':
            self.dot_product, self.square_sum_x, self.square_sum_y = torch.zeros(B), torch.zeros(B), torch.zeros(B)


    def forward(self, x1, x2):

        if self.loss_type=='L1':
            return self.L1Loss(x1, x2)

        if self.loss_type=='L2':
            return self.L2Loss(x1, x2)

        if self.loss_type=='cos':
            return self.cosine_similarity(x1, x2)


    def L1Loss(self, x1, x2):

        loss = torch.sum(torch.abs(x1[:]-x2[:]), dim=1)
        return loss

    def L2Loss(self, x1, x2):

        loss = torch.sum((x1[:]-x2[:]).pow(2), dim=1)
        return loss

    def bit_product_sum(self, x, y):
        return sum([item[0] * item[1] for item in zip(x, y)])


    def cosine_similarity(self, x, y, norm=True):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"

        # method 1
        #res = torch.tensor([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        #cos = sum(res[:, 0]) / (torch.sqrt(sum(res[:, 1])) * torch.sqrt(sum(res[:, 2])))

        # method 2
        # cos = self.bit_product_sum(x, y) / (torch.sqrt(self.bit_product_sum(x, x)) * torch.sqrt(self.bit_product_sum(y, y)))

        #method 3
        dot_product, square_sum_x, square_sum_y = self.dot_product, self.square_sum_x, self.square_sum_y
        for i in range(x.size()[1]):
            dot_product[:] += x[:,i] * y[:,i]
            square_sum_x[:] += x[:,i] * x[:,i]
            square_sum_y[:] += y[:,i] * y[:,i]
        cos = dot_product / (torch.sqrt(square_sum_x) * torch.sqrt(square_sum_y))

        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

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

class SEDAM_Module(nn.Module):
    """ Deep attention module"""
    def __init__(self, in_dim):
        super(SEDAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv_du = nn.Sequential(
            nn.Conv2d(121, 11, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(11, 121, 1, padding=0, bias=True),
        )
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

        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, N, N)
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy1 = torch.zeros((m_batchsize, N, 1)).cuda()
        for i in range(N):
            energy1.data[:,i] = torch.sqrt(energy[:,i,i]).unsqueeze(1)
        energy2 = energy1.permute(0, 2, 1)
        energy = energy/energy1.expand_as(energy)
        energy = energy/energy2.expand_as(energy)
        energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, N, N)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class MSAM_Module(nn.Module):
    """MultiScale Sptial Attention"""
    def __init__(self, in_dim):
        super(MSAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.conv_du = nn.Sequential(
        #     nn.Conv2d(2304*2304, 48, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(48, 2304*2304, 1, padding=0, bias=True),
        # )
        self.conv0 = nn.Conv2d(in_dim, in_dim//16, 1, 1, 0)
        #self.conv1 = nn.Conv2d(in_dim/2, in_dim/2, 3, 1, 1)
        self.conv = nn.Conv2d(in_dim//16, in_dim//16, 3, 1, 1)
        self.atten_conv = nn.Conv2d(in_dim//16,1,1,1,0)

        self.last_conv = nn.Conv2d(in_dim//16*4, in_dim, 1, 1, 0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
    
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X HW X HW
        """
        m_batchsize, C, height, width = x.size()
        x1 = self.multi_scale(x)

        proj_query = x1.view(m_batchsize, -1, C*height*width//16)
        proj_key = x1.view(m_batchsize, -1, C*height*width//16).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        #energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, H*W, H*W)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention = self.softmax(energy_new)
        proj_value = x1.view(m_batchsize, -1, C*height*width//16)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, -1, height, width)
        out = self.last_conv(out)

        out = self.gamma*out + x
        #out = out.view(m_batchsize, -1, height, width)
        return out

    def attention(self, x):
        out = self.sigmoid(self.atten_conv(x))

        return out*x+x

    def one_scale(self, x, scale=2):
        m_batchsize, C, height, width = x.size()
        dowsample = nn.AvgPool2d(scale, stride=scale)
        upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        #pdb.set_trace()
        x = dowsample(x)
        x = self.relu(self.conv(x))
        x = upsample(x)
        x = self.attention(x)

        return x

    def multi_scale(self, x):
        x = self.relu(self.conv0(x))
        out = self.conv(x)
        out = out.unsqueeze(1)
        scale_list = [2,3,4]
        for scale in scale_list:
            x1 = self.one_scale(x ,scale).unsqueeze(1)
            #pdb.set_trace()
            out = torch.cat([out, x1], 1)

        return out

class SAM_Module(nn.Module):
    """SE Sptial Attention"""
    def __init__(self, in_dim):
        super(SAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.conv_du = nn.Sequential(
        #     nn.Conv2d(2304*2304, 48, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(48, 2304*2304, 1, padding=0, bias=True),
        # )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()
        #self.conv0 = nn.Conv2d(in_dim*4, 1, 1, 1, 0)
        self.pad1 = nn.ReplicationPad2d((0,0,1,0))
        self.pad2 = nn.ReplicationPad2d((1,0,0,0))
        self.pixel_shuffle = nn.PixelShuffle(2)
    
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X HW X HW
        """
        x,top,left = self.depixel_shuffle(x)
        m_batchsize, C, height, width = x.size()

        proj_query = x.view(m_batchsize, -1, height*width)
        proj_key = x.view(m_batchsize, -1, height*width).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        energy1 = torch.zeros((m_batchsize, height*width, 1)).cuda()
        for i in range(height*width):
            energy1.data[:,i] = torch.sqrt(energy[:,i,i]).unsqueeze(1)
        energy = energy/energy1.expand_as(energy)
        energy1 = energy1.permute(0, 2, 1)
        #energy = energy/energy1.expand_as(energy)
        energy = energy/energy1.expand_as(energy)
              #energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, H*W, H*W)
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        energy = self.softmax(energy)
        #attention = self.absmax(energy_new)
        #proj_value = x.view(m_batchsize, -1, height*width)

        out = torch.bmm(proj_query, energy.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)
        #atten = self.conv0(x)


        out = self.gamma*out + x
        out = self.pixel_shuffle(out)
        if top != 0:
            out = out[:,:,1:,:]
        if left != 0:
            out = out[:,:,:,1:]
        #out = out.view(m_batchsize, -1, height, width)
        return out

    def depixel_shuffle(self, x, upscale_factor=2):    
        batch_size, channels, height, width = x.size()
        pdb.set_trace()
        out_channels = channels * (upscale_factor ** 2)              
        top,left = 0,0
        if height%2==1:
            x = self.pad1(x)
            top=1
        if width%2==1:
            x = self.pad2(x)
            left=1

        height = math.ceil(height / upscale_factor)
        width = math.ceil(width / upscale_factor)

        x_view = x.contiguous().view(
            batch_size, channels, height, upscale_factor, width, upscale_factor)

        shuffle_out = x_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffle_out.view(batch_size, out_channels, height, width),top,left
        
    def squaremax(self, x, dim=-1):
        x_square = x.pow(2)
        x_sum = torch.sum(x_square, dim=dim, keepdim=True)
        s = x_square / x_sum
        return s
        
    def logmax(self,x):
        x_log = torch.log(x+1)
        x_sum = torch.sum(x_log, dim=-1, keepdim=True)
        s = x_log / x_sum
        return s
        
    def absmax(self,x):
        x_abs = torch.abs(x)
        x_sum = torch.sum(x_abs, dim=-1, keepdim=True)
        s = x_abs / x_sum
        return s

class SECAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(SECAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv_du = nn.Sequential(
            nn.Conv2d(4096, 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4096, 1, padding=0, bias=True),
        )
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
        pdb.set_trace()

        # proj_query = x.view(m_batchsize, C, -1)
        # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        proj_query = x.contiguous().view(m_batchsize, C, -1)
        proj_key = x.contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy1 = torch.zeros((m_batchsize, C, 1)).cuda()
        for i in range(C):
            energy1.data[:,i] = torch.sqrt(energy[:,i,i]).unsqueeze(1)
        energy2 = energy1.permute(0, 2, 1)
        energy = energy/energy1.expand_as(energy)
        energy = energy/energy2.expand_as(energy)
        energy = self.conv_du(energy.view(m_batchsize, -1, 1, 1)).view(m_batchsize, C, C)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma*out + x
        #out = out.view(m_batchsize, -1, height, width)
        return out


class LAM_Module(nn.Module):
    """ Deep attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.dis = Dis('L1')
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.energy1 =  torch.zeros((4, 11, 11)).cuda()
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
            process:
            reshape x > 2d
            任意两行层特征，求关系置信度。关系置信度定义为 距离求反
            得到置信度矩阵
            矩阵相乘，乘上尺度因子，再与输入相加
            
        """
        m_batchsize, N, C, height, width = x.size()
        energy1 = torch.zeros((4, 11, 11)).cuda()

        #energy2 = Variable(energy1,requires_grad=True)
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy2 = torch.bmm(proj_query, proj_key)
        for i in range(N):
                #a = []
            for j in range(i,N):
                #pdb.set_trace()
                #a.append(self.dis(proj_query[:][i],proj_query[:][j]))
                energy1.data[:,i,j] = self.dis(proj_query[:,i],proj_query[:,j])

                energy1.data[:,j,i] = energy1.data[:,i,j]
            #energy1.append(a)
        
        energy = energy1*energy2

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out
        
        
class GAM_Module(nn.Module):
    """ Global
    attention module"""
    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        #modules_body.append(RCMSAN())
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.ca = SECAM_Module(n_feats)
        self.sa = SAM_Module(n_feats)
        self.da = SEDAM_Module(n_feats)
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.lastc = nn.Conv2d(n_feats*3, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        #pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            #print(name)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.da(res1)
        out2 = self.last_conv(res)

        out1 = self.sa(out1)
        out3 = self.ca(out1)
        out = torch.cat([out1, out2, out3], 1)
        res = self.lastc(out)
        
        res += x
        #res = self.ga(res)

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

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