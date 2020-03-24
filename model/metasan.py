from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MPNCOV.python import MPNCOV
import math

def make_model(args, parent=False):
    return SAN(args)

class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        #in_channels= 64
        #inter_channels= 8
        #dimension=2
        #mode='embedded_gaussian'
        #sub_sample=sub_sample=False
        #bn_layer=bn_layer=False 

        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        #Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。

        print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2: #运行这个
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else: #运行这个
            # conv_nd = nn.Conv2d
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            #self.inter_channels = 8
            #self.in_channels= 64
            
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

            #nn.init.constant_ 例子：
            # w = torch.empty(3)
            # nn.init.constant_(w, 0.3)
            #w变为：
            # tensor([0.3000,  0.3000,  0.3000])


        self.theta = None
        self.phi = None
        self.concat_project = None
        # self.fc = nn.Linear(64,2304,bias=True)
        # self.sub_bilinear = nn.Upsample(size=(48,48),mode='bilinear')
        # self.sub_maxpool = nn.AdaptiveMaxPool2d(output_size=(48,48))

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:#运行这个
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian': #运行这个
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample: #不运行
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size,C,H,W = x.shape

        # x_sub = self.sub_bilinear(x) # bilinear downsample
        # x_sub = self.sub_maxpool(x) # maxpool downsample

        ##
        # g_x = x.view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)
        #
        # # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        # theta_x = x.view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # fc = self.fc(theta_x)
        # # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # # f = torch.matmul(theta_x, phi_x)
        # # return f
        # # f_div_C = F.softmax(fc, dim=-1)
        # return fc

        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        # return f
        f_div_C = F.softmax(f, dim=-1)
        # return f_div_C
        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        #in_channels=in_feat=64
        #inter_channels=inter_feat=8
        #sub_sample=sub_sample=False
        #bn_layer=bn_layer=False        
        
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)
        #in_channels= 64
        #inter_channels= 8
        #dimension=2
        #mode='embedded_gaussian'
        #sub_sample=sub_sample=False
        #bn_layer=bn_layer=False 






## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                # nn.Sigmoid()
                # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        _,_,h,w = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_ave



## second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8): ##两个参数的值分别是channel=64和reduction=16
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_cov = self.conv_du(cov_mat_sum)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov*x



## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        #in_feat=n_feats=64, 
        #inter_feat=n_feats//8,=8 
        #reduction=8,
        #sub_sample=False, 
        #bn_layer=False
        
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        #这里怎么会有个SOCA？？？
        self.soca=SOCA(in_feat, reduction=reduction)
        
        # nonlocal module
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))
        #in_channels=in_feat=64
        #inter_channels=inter_feat=8
        #sub_sample=sub_sample=False
        #bn_layer=bn_layer=False


        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]


        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return  nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1, dilation=2):#dilation没啥用
        super(RB, self).__init__()
        modules_body = []

        # self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma1 = 1.0
        # self.salayer = SALayer(n_feat, reduction=reduction, dilation=dilation)
        # self.salayer = SALayer2(n_feat, reduction=reduction, dilation=dilation)



        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )
        #nn.Sequential：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。

        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x

        return y

## Local-source Residual Attention Group (LSRARG)
class LSRAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        #res_scale。residual scaling：default=1。
        #n_resblocks=16
        #reduction. default=16 'number of feature maps reduction'
        #n_feats. default = 64. 'number of feature maps'难道是指特征映射通道数？
        #act = nn.ReLU(inplace=True)
        #'--res_scale'。residual scaling。default=1。

        super(LSRAG, self).__init__()
        ##
        self.rcab= nn.ModuleList([RB(conv, n_feat, kernel_size, reduction, \
                                       bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in range(n_resblocks)])
        self.soca = (SOCA(n_feat,reduction=reduction))#两个参数的值分别是 64 和 16
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks
        ##
        # modules_body = []
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        # for i in range(n_resblocks):
        #     modules_body.append(RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1))
        # modules_body.append(SOCA(n_feat,reduction=reduction))
        # # modules_body.append(Nonlocal_CA(in_feat=n_feat, inter_feat=n_feat//8, reduction =reduction, sub_sample=False, bn_layer=False))
        # modules_body.append(conv(n_feat, n_feat, kernel_size))
        # self.body = nn.Sequential(*modules_body)
        ##

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        # batch_size,C,H,W = x.shape
        # y_pre = self.body(x)
        # y_pre = y_pre + x
        # return y_pre

        ## share-source skip connection

        for i,l in enumerate(self.rcab):
            # x = l(x) + self.gamma*residual
            x = l(x)
        x = self.soca(x)
        x = self.conv_last(x)

        x = x + residual

        return x
        ##

## Second-order Channel Attention Network (SAN)
class SAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SAN, self).__init__()

        self.args = args
        self.scale_idx = 0

        n_resgroups = args.n_resgroups #default = 20。'number of residual groups'
        n_resblocks = args.n_resblocks #default = 16。'number of residual blocks'
        n_feats = args.n_feats #default = 64. 'number of feature maps'

        kernel_size = 3 
        reduction = args.reduction #default=16 'number of feature maps reduction'
        
        print("args.scale[0]:")
        print(args.scale[0])

        scale = args.scale[0]
        act = nn.ReLU(inplace=True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.soca= SOCA(n_feats, reduction=reduction)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        ## share-source skip connection share-source跳过连接，怎么操作的？？？

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        #含义是将一个固定不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
        #所以经过类型转换这个self.gamma变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        #使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # self.gamma = 0.2

        self.n_resgroups = n_resgroups #20
        self.RG = nn.ModuleList([LSRAG(conv, n_feats, kernel_size, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)])
        #res_scale。residual scaling：default=1。
        #n_resblocks=16
        #reduction. default=16 'number of feature maps reduction'
        #n_feats. default = 64. 'number of feature maps'难道是指特征映射通道数？
        #act = nn.ReLU(inplace=True)
        #'--res_scale'。residual scaling。default=1。


        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]
        # modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats//8, reduction=8,sub_sample=False, bn_layer=False)
        #in_feat=n_feats=64, 
        #inter_feat=n_feats//8,=8 
        #reduction=8,
        #sub_sample=False, 
        #bn_layer=False

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)

        # self.tail = nn.Sequential(*modules_tail) #这里注释了

        ## position to weight
        self.P2W = Pos2Weight(inC=n_feats)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    def repeat_weight(self, weight, scale, inw,inh):
        k = int(math.sqrt(weight.size(0)))
        outw  =inw * scale
        outh = inh * scale
        weight = weight.view(k, k, -1)
        scale_w = (outw+k-1) // k
        scale_h = (outh + k - 1) // k
        weight = torch.cat([weight] * scale_h, 0)
        weight = torch.cat([weight] * scale_w, 1)

        weight = weight[0:outh,0:outw,:]

        return weight

    def forward(self, x, pos_mat):
        x = self.sub_mean(x)
        x = self.head(x)

        ## add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # res = self.RG(xx)
        # res = res + xx

        ## share-source residual gruop
        #Due to our share-source skip connections, the abundant lowfrequency information can be bypassed.

        for i,l in enumerate(self.RG):
            xx = l(xx) + self.gamma*residual
            # xx = self.gamma*xx + residual
        # body part
        # res = self.body(xx)
        ##
        ## add nonlocal
        res = self.non_local(xx)
        ##
        # res = self.soca(res)
        # res += x
        res = res + x

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        up_x = self.repeat_x(res)     ### the output is (N*r*r,inC,inH,inW)

        # N*r^2 x [inC * kH * kW] x [inH * inW]
        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)
        local_weight = self.repeat_weight(local_weight,scale_int,x.size(2),x.size(3))
        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = self.add_mean(out)
        ###
        return out 

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]
