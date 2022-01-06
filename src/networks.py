import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict



class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)





class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm,
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm,
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class SqueezeResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False, reduction=16):
        super(SqueezeResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm,
            nn.BatchNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm,
            nn.BatchNorm2d(dim),
        )


        self.linear1 = nn.Linear(dim, dim//reduction)
        self.linear2 = nn.Linear(dim//reduction, dim)


    def forward(self, x):
        x1 = torch.mean(torch.mean(x, dim=3), dim=2)
        x1 = self.linear1(x1)
        x1 = F.relu(x1)
        x1 = self.linear2(x1)
        x1 = torch.sigmoid(x1)
        x1 = x1.unsqueeze(2).unsqueeze(3)


        out = x + x1 * self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out








class Encoder(BaseNetwork):
    def __init__(self, base_channel_nums, init_weights=True):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential( nn.ReflectionPad2d(2),
            nn.Conv2d(3, base_channel_nums, 5, 1),)

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_channel_nums, base_channel_nums * 2, 3, 2),
            nn.BatchNorm2d(base_channel_nums * 2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_channel_nums * 2, base_channel_nums * 2, 3, 1),
            nn.BatchNorm2d(base_channel_nums * 2),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_channel_nums * 2, base_channel_nums * 4, 3, 2),
            nn.BatchNorm2d(base_channel_nums * 4),
            nn.ReLU(),
        )


        # blocks = []
        #
        # for i in range(3):
        #     blocks.append(ResnetBlock(base_channel_nums*4))
        #
        # self.resblocks = nn.Sequential(*blocks)



        if init_weights:
            self.init_weights('kaiming')

    def forward(self, x):
        #x = self.net(x)
        #x = self.resblocks(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)


        return x4, [x1,x2,x3,x4]

#
# class Decoder(BaseNetwork):
#     def __init__(self, base_channel_nums, init_weights=True):
#         super(Decoder, self).__init__()
#
#
#
#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=base_channel_nums*4, out_channels=base_channel_nums*2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(base_channel_nums*2),
#             nn.ReLU(True),
#         )
#
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=base_channel_nums * 2, out_channels=base_channel_nums * 2, kernel_size=4, padding=1,
#                                stride=2),
#             nn.BatchNorm2d(base_channel_nums*2),
#             nn.ReLU(True)
#         )
#
#         self.conv = nn.Sequential(
#             nn.ReflectionPad2d(2),
#             nn.Conv2d(in_channels=base_channel_nums*2, out_channels=3, kernel_size=5),
#         )
#
#         if init_weights:
#             self.init_weights('kaiming')
#
#     def forward(self, input):
#         deconv1 = self.deconv1(input)
#         deconv2 = self.deconv2(deconv1)
#         result = self.conv(deconv2)
#         result = (torch.tanh(result)+1)/2
#         return result, [deconv1,deconv2]


class Decoder(BaseNetwork):
    def __init__(self, base_channel_nums, out_channels=3, init_weights=True):
        super(Decoder, self).__init__()

        #self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
        self.up1 = pixelshuffle_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, act_type='relu', norm_type='batch')

        self.conv1 = conv_block(base_channel_nums*2, out_nc=base_channel_nums*2, kernel_size=3, act_type='relu', norm_type='batch')

        #self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')

        self.up2 = pixelshuffle_block(in_nc=base_channel_nums * 2, out_nc=base_channel_nums * 1, act_type='relu',
                                      norm_type='batch')

        self.conv2 = conv_block(base_channel_nums * 1, out_nc= out_channels, kernel_size=3, act_type=None ,norm_type=None)

        if init_weights:
            self.init_weights('kaiming')

    def forward(self, input):
        deconv1 = self.up1(input)
        conv1 = self.conv1(deconv1)
        deconv2 = self.up2(conv1)
        result = self.conv2(deconv2)
        result = (torch.tanh(result)+1)/2
        return result, [deconv1,deconv2]



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







class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBNRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class DNCNN(BaseNetwork):
    def __init__(self, num_features, num_layers, in_channels=3, out_channels=3):
        super(DNCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        blocks = []
        for i in range(num_layers - 2):
            blocks.append(ConvBNRelu(num_features))

        self.mid_blocks = nn.Sequential(*blocks)

        self.conv_final = nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mid_blocks(x)
        x = self.conv_final(x)

        return x


class GSBlock(nn.Module):
    def __init__(self, num_features, num_layers, is_first=False):
        super(GSBlock, self).__init__()

        blocks = []
        blocks.append(ConvBNRelu(in_channels=num_features+1 if is_first else 2*num_features+1, out_channels=num_features))
        for i in range(num_layers-1):
            blocks.append(ConvBNRelu(in_channels=num_features, out_channels=num_features))
        self.gaussian_blocks = nn.Sequential(*blocks)

        blocks = []
        blocks.append(ConvBNRelu(in_channels=num_features + 1 if is_first else 2*num_features+1, out_channels=num_features))
        for i in range(num_layers - 1):
            blocks.append(ConvBNRelu(in_channels= num_features, out_channels=num_features))
        self.sobel_blocks = nn.Sequential(*blocks)

        blocks = []
        blocks.append(ConvBNRelu(in_channels=3*num_features, out_channels=num_features))
        for i in range(num_layers):
            blocks.append(ConvBNRelu(in_channels=num_features, out_channels=num_features))
        self.main_blocks = nn.Sequential(*blocks)

        self.sobelkernel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand(1, 1, -1,
                                                                                                     -1).cuda()
        self.sobelkernel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).expand(1, 1, -1,
                                                                                                     -1).cuda()
        self.is_first = is_first

    def forward(self, x, x_sobel, x_gauss, x_sobel_pre=None, x_gauss_pre=None):
        if self.is_first:
            f_sobel = self.sobel_blocks(torch.cat([x,x_sobel], dim=1))
            f_gauss = self.gaussian_blocks(torch.cat([x,x_gauss], dim=1))

        else:
            f_sobel = self.sobel_blocks(torch.cat([x,x_sobel,x_sobel_pre], dim=1))
            f_gauss = self.gaussian_blocks(torch.cat([x,x_gauss, x_gauss_pre], dim=1))

        f_main = self.main_blocks(torch.cat([x,f_sobel, f_gauss], dim=1))

        return f_main, f_sobel, f_gauss


class FeatureProcessNet(BaseNetwork):
    def __init__(self, num_features, num_blocks, norm_type=None, act_type='leakyrelu'):
        super(FeatureProcessNet, self).__init__()

        self.block_num = num_blocks
        # self.grad_b_conv = conv_block(num_features, num_features, kernel_size=3, norm_type=None, act_type=None)

        # blocks = []
        # for i in range(num_blocks):
        #     blocks.append(nn.Sequential(
        #         conv_block(2 * num_features, num_features, kernel_size=3, norm_type=None, act_type=None),
        #         RRDB(num_features, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
        #              norm_type=norm_type, act_type=act_type, mode='CNA')
        #     ))
        # self.grad_blocks = nn.ModuleList(blocks)

        blocks = []
        for i in range(num_blocks):
            blocks.append(nn.Sequential(
                ContextBlock(input_channel=num_features, output_channel=num_features // 4),
                ResNetBlock(in_nc=num_features, mid_nc=num_features, out_nc=num_features, act_type=act_type, norm_type=norm_type),
                ResNetBlock(in_nc=num_features, mid_nc=num_features, out_nc=num_features, act_type=act_type, norm_type=norm_type),
            ))

        # self.tran_last_conv = conv_block(num_features, num_features, kernel_size=3, norm_type=None, act_type=None)
        self.main_blocks = nn.ModuleList(blocks)

        # self.merge_block_1 = conv_block(in_nc=2 * num_features, out_nc=num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        # self.merge_block_2 = ResNetBlock(in_nc=num_features, mid_nc=num_features, out_nc=num_features, act_type=act_type, norm_type=norm_type)

        # self.tran_merge_block = conv_block(in_nc= num_features, out_nc=1, kernel_size=1, act_type=None)

        #
        # self.b_concat_1 = conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        # self.b_block_1 = RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
        #                         norm_type=norm_type, act_type=act_type, mode='CNA')
        #
        # self.b_concat_2 = conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        # self.b_block_2 = RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
        #                         norm_type=norm_type, act_type=act_type, mode='CNA')
        #
        # self.b_concat_3 = conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        # self.b_block_3 = RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
        #                         norm_type=norm_type, act_type=act_type, mode='CNA')
        #
        # self.b_concat_4 = conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        # self.b_block_4 = RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
        #                         norm_type=norm_type, act_type=act_type, mode='CNA')
        #
        # self.main_muldilated_1 = ContextBlock(input_channel=nf, output_channel=nf//4 )
        # self.resblock_1_1 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        # self.resblock_1_2 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        #
        # self.main_muldilated_2 = ContextBlock(input_channel=nf, output_channel=nf//4 )
        # self.resblock_2_1 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        # self.resblock_2_2 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        #
        # self.main_muldilated_3 = ContextBlock(input_channel=nf, output_channel=nf//4 )
        # self.resblock_3_1 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        # self.resblock_3_2 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        #
        # self.main_muldilated_4 = ContextBlock(input_channel=nf, output_channel=nf//4 )
        # self.resblock_4_1 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        # self.resblock_4_2 = ResNetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, act_type=act_type, norm_type=norm_type)
        #


    def forward(self, f_main):
        #f_sob = self.sobel_filter.apply_sobel_filter_keepc(f_main)
        # f_tran = self.grad_b_conv(f_main)
        # f_main_p = f_main
        f_main_r = f_main
        for i in range(self.block_num):
            f_main_r = self.main_blocks[i](f_main_r)
            #f_tran = self.grad_blocks[i](torch.cat([f_main_p, f_tran], dim=1))
        # f_tran = self.tran_last_conv(f_tran)
        #
        # f_main = f_main + f_main_p
        #
        # f_main = self.merge_block_1(torch.cat([f_main, f_tran],dim=1))
        # f_main = self.merge_block_2(f_main)
        #
        # tran_map = self.tran_merge_block(f_tran)
        # tran_map = (torch.tanh(tran_map)+1)/2

        return f_main + f_main_r

# class Decoder2(BaseNetwork):
#     def __init__(self, base_channel_nums, out_channels=3, init_weights=True):
#         super(Decoder2, self).__init__()
#
#         #self.deconv1 = deconv_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
#         self.up1 = pixelshuffle_block(in_nc=base_channel_nums*4, out_nc=base_channel_nums*2, act_type='relu', norm_type='batch')
#
#         self.conv1 = conv_block(base_channel_nums*2, out_nc=base_channel_nums*2, kernel_size=3, act_type='relu', norm_type='batch')
#
#         #self.deconv2 = deconv_block(in_nc=base_channel_nums*2, out_nc=base_channel_nums*1, kernel_size=4, padding=1, stride=2, act_type='relu', norm_type='batch')
#
#         self.up2 = pixelshuffle_block(in_nc=base_channel_nums * 2, out_nc=base_channel_nums * 1, act_type='relu',
#                                       norm_type='batch')
#
#         self.conv2 = conv_block(base_channel_nums * 1, out_nc= out_channels, kernel_size=3, act_type=None ,norm_type=None)
#
#         if init_weights:
#             self.init_weights('kaiming')
#
#     def forward(self, input):
#         deconv1 = self.up1(input)
#         conv1 = self.conv1(deconv1)
#         deconv2 = self.up2(conv1)
#         result = self.conv2(deconv2)
#         result = (torch.tanh(result)+1)/2
#         return result, [deconv1,deconv2]
def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)




class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()

        self.sobelkernel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).expand(1, 1, -1,
                                                                                                     -1).cuda()
        self.sobelkernel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).expand(1, 1, -1,
                                                                                                     -1).cuda()

    def apply_sobel_filter(self, input):
        n,c,h,w = input.shape

        sobel_x = torch.abs(torch.nn.functional.conv2d(input, self.sobelkernel_x.expand(1,c,-1,-1), padding=1))
        sobel_y = torch.abs(torch.nn.functional.conv2d(input, self.sobelkernel_y.expand(1,c,-1,-1), padding=1))

        return sobel_x + sobel_y


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=2):
        super(GaussianFilter, self).__init__()

        self.gaussian_kernel = self.cal_kernel(kernel_size=kernel_size, sigma=sigma).expand(1,1,-1,-1).cuda()


    def apply_gaussian_filter(self, x):
        # cal gaussian filter of N C H W in cuda
        n,c,h,w = x.shape
        gaussian = torch.nn.functional.conv2d(x,self.gaussian_kernel.expand(1,c,-1,-1),padding=self.gaussian_kernel.shape[2]//2)

        return gaussian

    def cal_gaussian_kernel_at_ij(self, i, j, sigma):
        return (1. / (2 * math.pi * pow(sigma, 2))) * math.exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2)))

    def cal_kernel(self, kernel_size=3, sigma=1.):
        kernel = torch.ones((kernel_size, kernel_size)).float()
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = self.cal_gaussian_kernel_at_ij(-(kernel_size // 2) + j, (kernel_size // 2) - i, sigma=sigma)

        kernel = kernel / torch.sum(kernel)
        return kernel



class DNCNN_FeatureProcessNet(BaseNetwork):
    def __init__(self, num_features, num_layers, in_channels=3, out_channels=3):
        super(DNCNN_FeatureProcessNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, padding=1,
                               bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        blocks = []
        for i in range(num_layers - 2):
            blocks.append(ConvBNRelu(num_features, out_channels=num_features))

        self.mid_blocks = nn.Sequential(*blocks)

        self.conv_final = nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=3, padding=1,
                                    bias=False)

    def forward(self, x):
        n = self.conv1(x)
        n = self.relu1(n)
        n = self.mid_blocks(n)
        n = self.conv_final(n)

        return x + n


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_4C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_4C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_4C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding



class ResidualDenseBlock_4C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)

        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv4 = conv_block(nc + 3 * gc, nc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4.mul(0.2) + x




def pixelshuffle_block(in_nc, out_nc, bias=True, norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)

    c = nn.Conv2d(in_nc, 2*in_nc, kernel_size=3, stride=1, padding=1, bias=bias)
    ps = nn.PixelShuffle(upscale_factor=2)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(c, ps, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, ps,c)


