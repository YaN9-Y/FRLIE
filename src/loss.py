import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd as autograd
from functools import reduce
from math import exp


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class DecoderStyleLoss(nn.Module):
    def __init__(self):
        super(DecoderStyleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y, masks):
        assert len(x) == len(y)
        style_loss = 0.

        for i in range(len(x)):
            H,W = x[i].shape[2:4]
            scaled_masks = torch.nn.functional.interpolate(masks,[H,W])
            style_loss += self.criterion(self.compute_gram(x[i]*scaled_masks),self.compute_gram(y[i]*scaled_masks))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            #'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            #'relu3_2': relu3_2,
            #'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            #'relu4_2': relu4_2,
            #'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            #'relu5_3': relu5_3,
            #'relu5_4': relu5_4,
        }
        return out

# class VGG19(nn.Module):
#     def __init__(self, pool='max'):
#         super(VGG19, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         if pool == 'max':
#             self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
#         elif pool == 'avg':
#             self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         out = {}
#         out['r11'] = F.relu(self.conv1_1(x))
#         out['r12'] = F.relu(self.conv1_2(out['r11']))
#         out['p1'] = self.pool1(out['r12'])
#         out['r21'] = F.relu(self.conv2_1(out['p1']))
#         out['r22'] = F.relu(self.conv2_2(out['r21']))
#         out['p2'] = self.pool2(out['r22'])
#         out['r31'] = F.relu(self.conv3_1(out['p2']))
#         out['r32'] = F.relu(self.conv3_2(out['r31']))
#         out['r33'] = F.relu(self.conv3_3(out['r32']))
#         out['r34'] = F.relu(self.conv3_4(out['r33']))
#         out['p3'] = self.pool3(out['r34'])
#         out['r41'] = F.relu(self.conv4_1(out['p3']))
#         out['r42'] = F.relu(self.conv4_2(out['r41']))
#         out['r43'] = F.relu(self.conv4_3(out['r42']))
#         out['r44'] = F.relu(self.conv4_4(out['r43']))
#         out['p4'] = self.pool4(out['r44'])
#         out['r51'] = F.relu(self.conv5_1(out['p4']))
#         out['r52'] = F.relu(self.conv5_2(out['r51']))
#         out['r53'] = F.relu(self.conv5_3(out['r52']))
#         out['r54'] = F.relu(self.conv5_4(out['r53']))
#         out['p5'] = self.pool5(out['r54'])
#         return out
#


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        self.weight = torch.tensor([[[[0,1,0],[1, -4 ,1], [0 ,1 ,0]]]]).float().cuda().expand(1,256,3,3)
        self.criterion = nn.L1Loss()
    def __call__(self, x, y):
        return self.criterion(F.conv2d(x,self.weight,padding=1), F.conv2d(y,self.weight,padding=1)) / 256


class DecoderFeatureLoss(nn.Module):
    def __init__(self):
        super(DecoderFeatureLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def __call__(self, x, y, weight=[1.0,1.0,1.0,1.0,1.0]): #convconv64_256, deconv64_128_128,conv128_128,deconv128_256_64,conv256_3
        assert len(weight) == 5 and len(x) == len(y)
        decoder_feature_loss = 0.
        for i in range(len(x)):
            decoder_feature_loss += weight[i] * self.criterion(x[i],y[i])

        return decoder_feature_loss



############################### DMFN Part ####################################3

class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -input.mean()
        return {'g_loss': g_loss, 'd_loss': d_loss}


def gradient_penalty(xin, yout, mask=None):
    gradients = autograd.grad(yout, xin, create_graph=True,
                              grad_outputs=torch.ones(yout.size()).cuda(), retain_graph=True, only_inputs=True)[0]
    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def random_interpolate(gt, pred):
    batch_size = gt.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    # alpha = alpha.expand(gt.size()).cuda()
    interpolated = gt * alpha + pred * (1 - alpha)
    return interpolated

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG19FeatLayer()
        for k, v in self.featlayer.named_parameters():
            v.requires_grad = False
        self.self_guided_layers = ['relu1_1', 'relu2_1']
        self.feat_vgg_layers = ['relu{}_1'.format(x + 1) for x in range(5)]
        self.lambda_loss = 25
        self.gamma_loss = 1
        self.align_loss, self.guided_loss, self.fm_vgg_loss = None, None, None
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.coord_y, self.coord_x = torch.meshgrid(torch.arange(-1, 1, 1 / 16), torch.arange(-1, 1, 1 / 16))
        self.coord_y, self.coord_x = self.coord_y.cuda(), self.coord_x.cuda()

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def calc_align_loss(self, gen, tar):
        def sum_u_v(x):
            area = x.shape[-2] * x.shape[-1]
            return torch.sum(x.view(-1, area), -1) + 1e-7

        sum_gen = sum_u_v(gen)
        sum_tar = sum_u_v(tar)
        c_u_k = sum_u_v(self.coord_x * tar) / sum_tar
        c_v_k = sum_u_v(self.coord_y * tar) / sum_tar
        c_u_k_p = sum_u_v(self.coord_x * gen) / sum_gen
        c_v_k_p = sum_u_v(self.coord_y * gen) / sum_gen
        out = F.mse_loss(torch.stack([c_u_k, c_v_k], -1), torch.stack([c_u_k_p, c_v_k_p], -1), reduction='mean')
        return out

    def forward(self, gen, tar, mask_guidance, weight_fn):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        guided_loss_list = []
        mask_guidance = mask_guidance.unsqueeze(1)
        for layer in self.self_guided_layers:
            guided_loss_list += [F.l1_loss(gen_vgg_feats[layer] * mask_guidance, tar_vgg_feats[layer] * mask_guidance, reduction='sum') * weight_fn(tar_vgg_feats[layer])]
            mask_guidance = self.avg_pool(mask_guidance)
        self.guided_loss = reduce(lambda x, y: x + y, guided_loss_list)

        content_loss_list = [F.l1_loss(gen_vgg_feats[layer], tar_vgg_feats[layer], reduction='sum') * weight_fn(tar_vgg_feats[layer]) for layer in self.feat_vgg_layers]
        self.fm_vgg_loss = reduce(lambda x, y: x + y, content_loss_list)

        self.align_loss = self.calc_align_loss(gen_vgg_feats['relu4_1'], tar_vgg_feats['relu4_1'])

        return self.align_loss, self.guided_loss, self.fm_vgg_loss
        return self.gamma_loss * self.align_loss + self.lambda_loss * (self.guided_loss + self.fm_vgg_loss)



class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            x = layer(x)
            if isinstance(layer, nn.ReLU) and ri == 1:
                out[name] = x
                if ci == 5:
                    break
        # print([x for x in out])
        return out

class RaganLoss(nn.Module):
    def __init__(self, config):
        super(RaganLoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss().to(config.DEVICE)
        self.zeros = torch.zeros((config.BATCH_SIZE, 1)).to(config.DEVICE)
        self.ones = torch.ones((config.BATCH_SIZE, 1)).to(config.DEVICE)

    def Dra(self, x1, x2):
        return x1 - torch.mean(x2)

    def forward(self, x_real, x_fake, type):
        assert type in ['adv', 'dis']
        if type == 'dis':
            return (self.BCELoss(self.Dra(x_real, x_fake), self.ones) + self.BCELoss(self.Dra(x_fake, x_real), self.zeros)) / 2
        else:
            return (self.BCELoss(self.Dra(x_real, x_fake), self.zeros) + self.BCELoss(self.Dra(x_fake, x_real), self.ones)) / 2



#############  ssim loss ##########33
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

## change to no L:
    # C1 = (0.01 * L) ** 2
    # C2 = (0.03 * L) ** 2

    C1 = (0.01 ) ** 2
    C2 = (0.03 ) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

    def __call__(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x):
        height = x.size()[2]
        width = x.size()[3]
        tv_h = torch.div(torch.sum(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])),(x.size()[0]*x.size()[1]*(height-1)*width))
        tv_w = torch.div(torch.sum(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])),(x.size()[0]*x.size()[1]*(height)*(width-1)))
        return tv_w + tv_h
    
    
class HueLoss(nn.Module):
    def __init__(self):
        super(HueLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, img1, img2):
        img_h1 = rgb2hsv(img1)[:,0].unsqueeze(1)
        img_h2 = rgb2hsv(img2)[:,0].unsqueeze(1)
        hueloss = 180-torch.abs(180-torch.abs(img_h1-img_h2))
        hueloss = torch.mean(hueloss) / 360
        return hueloss


def rgb2hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)
