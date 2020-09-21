from model import common

import torch.nn as nn
import torch
import torch.nn.init as init
import pdb

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

class Dis(nn.Module):
    def __init__(self, loss_type='L1', batchsize=16):
        super(Dis, self).__init__()
        self.loss_type = loss_type
        #self.loss = torch.zeros(B)
        if self.loss_type == 'cos':
            self.dot_product, self.square_sum_x, self.square_sum_y = torch.zeros(batchsize).cuda(), torch.zeros(batchsize).cuda(), torch.zeros(batchsize).cuda()


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
        #pdb.set_trace()
        for i in range(x.size()[1]):
            dot_product += x[:,i] * y[:,i]
            square_sum_x += x[:,i] * x[:,i]
            square_sum_y += y[:,i] * y[:,i]
        cos = dot_product / (torch.sqrt(square_sum_x) * torch.sqrt(square_sum_y))

        return 0.5 * cos + 0

class FullConvRes(nn.Module):
    """ Full Receptive Field Conv2d Residual Block"""
    def __init__(self, out_channels=64, in_channels=64, K=9):
        super(FullConvRes, self).__init__()


        #self.dis = Dis('cos', batchsize=1)
        self.out_channels = out_channels
        self.K = K
        #self.conv = nn.Conv2d(K,K,1,1,0)
        #self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.energy1 =  torch.zeros((4, 11, 11)).cuda()
        #self.softmax  = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, K)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        init.xavier_uniform(self.weight)
        init.constant(self.bias, 0.1)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fullconv value + input feature
                attention: B X HW X 9
            process:
            reshape x > 2d
            
        """
        m_batchsize, C, height, width = x.size()
        #energy1 = torch.zeros((m_batchsize, height*width, height*width)).cuda()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        energy1 = torch.zeros((m_batchsize, height*width, 1)).cuda()
        for i in range(height*width):
            energy1.data[:,i] = torch.sqrt(energy[:,i,i]).unsqueeze(1)
        energy2 = energy1.permute(0, 2, 1)
        energy_new = energy/energy1.expand_as(energy)
        energy_new = energy_new/energy2.expand_as(energy)
        #energy_new = energy_new*energy
        #energy = self.softmax(energy)
        #energy_new = self.softmax(energy_new)

        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        #pdb.set_trace()
        energy_new = torch.sort(energy_new, dim=-1)[1].float()
        #pdb.set_trace()
        e = torch.chunk(energy_new, self.K, dim=-1)
        for i in range(self.K):
            if i == 0:
                energy_new = e[i][:,:,0].unsqueeze(2)
            else:
                energy_new = torch.cat([energy_new, e[i][:,:,0].unsqueeze(2)],dim=2)
        #energy_new = torch.stack(torch.chunk(energy_new, self.K, dim=-1)[:][:,:,0],dim=-1)
        #energy_new = energy_new[:,:,0,:].long()
        energy_new = energy_new.long()

        ReceptiveField = torch.zeros_like(energy)
        for b in range(m_batchsize):
            for t in range(height*width):
                for k in range(self.K):
                    #pdb.set_trace()
                    ReceptiveField.data[b,t,energy_new[b,t,k]] = 1
        # max5 = max((1,9))  # 取top1准确率，若取top1和top9准确率改为max((1,9))
        # max4 = max((1,4))
        # _, ReceptiveFieldIdex1 = energy_new.topk(max5, -1, True, False)
        # _, ReceptiveFieldIdex2 = (-1*energy_new).topk(max4, -1, True, False)
        # ReceptiveFieldIdex = torch.cat([ReceptiveFieldIdex1,ReceptiveFieldIdex2], 2)
        # ReceptiveFieldIdex = (self.sigmoid(self.conv(ReceptiveFieldIdex.unsqueeze(3)).squeeze(3))*height*width).int()

        #score = self.softmax(score)

        #x_in = x.view(m_batchsize,-1,height*width)
        out = torch.zeros_like(proj_query).cuda()

        for i in range(self.out_channels):
            for j in range(height*width):
                #x_in = x_in[ReceptiveField[:,j,:].unsqueeze(1).expand_as(x_in).long()]
                #pdb.set_trace()
                x_out = proj_query[ReceptiveField[:,j].unsqueeze(1).expand_as(proj_query)>0].view(m_batchsize,C,-1)
                #x_out,_ = x_in.topk(max9, -1, True, False) # The shape of x_in:B X C X 9
                x_K = torch.sum(x_out*(self.weight[i].expand_as(x_out)), dim=1)
                out.data[:,i,j] = torch.sum(x_K, dim=1)+self.bias[i]
        out = self.relu(out.view(m_batchsize,C,height,width))

        # max9 = max((1,9))
        # for i in range(self.out_channels):
        #     for j in range(height*width):
        #         #x_in = x_in[ReceptiveField[:,j,:].unsqueeze(1).expand_as(x_in).long()]
        #         pdb.set_trace()
        #         x1 = x_in*(ReceptiveField[:,j].unsqueeze(1).expand_as(x_in)) 
        #         x_out,_ = x1.topk(max9, -1, True, False) # The shape of x_out:B X C X 9
        #         x_9 = torch.sum(x_out*(self.weight[i].expand_as(x_out)), dim=1)
        #         out.data[:,i,j] = torch.sum(x_9, dim=1)+self.bias[i]
        # out = self.relu(out.view(m_batchsize,C,height,width))

        return self.gamma * out + x

class FullConvRes1(nn.Module):
    """ Full Receptive Field Conv2d Residual Block"""
    def __init__(self, out_channels=64, in_channels=64, kernel_size=3):
        super(FullConvRes1, self).__init__()


        #self.dis = Dis('cos', batchsize=1)
        self.out_channels = out_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        #self.value_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.energy1 =  torch.zeros((4, 11, 11)).cuda()
        #self.softmax  = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size*kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        init.xavier_uniform(self.weight)
        init.constant(self.bias, 0.1)
        self.relu = nn.ReLU(True)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fullconv value + input feature
                attention: B X HW X 9
            process:
            reshape x > 2d
            
        """
        m_batchsize, C, height, width = x.size()
        #energy1 = torch.zeros((m_batchsize, height*width, height*width)).cuda()
        proj_query = self.query_conv(x).view(m_batchsize, C//8, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C//8, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_key, proj_query)
        # energy1 = torch.zeros((m_batchsize, height*width, 1)).cuda()
        # for i in range(height*width):
        #     energy1.data[:,i] = torch.sqrt(energy[:,i,i]).unsqueeze(1)
        # energy2 = energy1.permute(0, 2, 1)
        # energy_new = energy/energy1.expand_as(energy)
        # energy_new = energy_new/energy2.expand_as(energy)
        #energy = self.softmax(energy)
        # energy_new = energy_new

        


        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        #pdb.set_trace()
        max9 = max((1,9))  # 取top1准确率，若取top1和top9准确率改为max((1,9))
        _, ReceptiveFieldIdex = energy.topk(max9, -1, True, False)

        proj_query = x.view(m_batchsize,-1,height*width)
        out = torch.zeros_like(proj_query).cuda()

        ReceptiveField = torch.zeros_like(energy)
        #x_out = torch.zeros_like(x_in)
        for b in range(m_batchsize):
            for t in range(height*width):
                for k in range(9):
                    #pdb.set_trace()
                    ReceptiveField.data[b,t,ReceptiveFieldIdex[b,t,k]] = 1

        for i in range(self.out_channels):
            for j in range(height*width):
                x_out = proj_query[ReceptiveField[:,j].unsqueeze(1).expand_as(proj_query)>0].view(m_batchsize,C,-1)
                #x_out,_ = x_in.topk(max9, -1, True, False) # The shape of x_in:B X C X 9
                x_K = torch.sum(x_out*(self.weight[i].expand_as(x_out)), dim=1)
                out.data[:,i,j] = torch.sum(x_K, dim=1)+self.bias[i]
        out = self.relu(out.view(m_batchsize,C,height,width))
        # for i in range(self.out_channels):
        #     for j in range(height*width):
        #         #x_in = x_in[ReceptiveField[:,j,:].unsqueeze(1).expand_as(x_in).long()]
        #         #pdb.set_trace()
        #         x_in = x_in*(ReceptiveField[:,j].unsqueeze(1).expand_as(x_in))
        #         x_out,_ = x_in.topk(max9, -1, True, False) # The shape of x_in:B X C X 9
        #         x_9 = torch.sum(x_out*(self.weight[i].expand_as(x_out)), dim=1)
        #         out.data[:,i,j] = torch.sum(x_9, dim=1)
        # out = self.relu(out.view(m_batchsize,C,height,width))

        return self.gamma * out + x

class FullConv(nn.Module):
    """ Full Receptive Field Conv2d Block"""
    def __init__(self, out_channels=64, in_channels=64, kernel_size=3):
        super(FullConv, self).__init__()


        self.dis = Dis('cos', batchsize=16)
        self.out_channels = out_channels
        #self.gamma = nn.Parameter(torch.zeros(1))
        #self.energy1 =  torch.zeros((4, 11, 11)).cuda()
        self.softmax  = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size*kernel_size)
        )
        self.relu = nn.ReLU(True)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fullconv value + input feature
                attention: B X HW X 9
            process:
            reshape x > 2d
            
        """
        m_batchsize, C, height, width = x.size()
        energy1 = torch.zeros((m_batchsize, height*width, height*width)).cuda()
        proj_query = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, C, -1)
        energy2 = torch.bmm(proj_query, proj_key)
        for i in range(height*width):
            for j in range(i,height*width):
                #pdb.set_trace()
                energy1.data[:,i,j] = self.dis(proj_query[:,i],proj_query[:,j])
                energy1.data[:,j,i] = energy1.data[:,i,j]
        
        energy = energy1*energy2

        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        maxk = max((1,9))  # 取top1准确率，若取top1和top9准确率改为max((1,9))
        top9, ReceptiveField = energy.topk(maxk, -1, True, False)
        top9 = top9*ReceptiveField
        score = self.softmax(top9)
        x_in = x.view(m_batchsize,-1,height*width)
        out = x_in
        for i in range(self.out_channels):
            for j in range(height*width):
                x_in = x_in[:,:,ReceptiveField[:,j,:]]*score[:,j,:] # The shape of x:B X C X 9
                out[:,i,j] = torch.sum(x_in*self.weight[i].expand_as(x), dim=0)
        out = self.relu(out.view(m_batchsize,C,height,width))

        return out

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
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        #modules_body.append(FullConvRes(n_feat, n_feat, kernel_size))
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
        modules_body.append(FullConvRes1())

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        #self.fcr = FullConvRes()
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

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