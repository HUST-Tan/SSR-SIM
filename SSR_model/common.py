import torch
import math
from math import pi
from torch import nn
import torch.nn.functional as F

from utils.option import sir_parse2dict
from utils.tools import my_meshgrid

device = torch.device('cuda')


# ----------------------------------------
# generate pattern in raw data size - for two stream raw denoise network
# ----------------------------------------
def generate_pattern(raw_input_data, para_data, json_path, modamp=0.5, center_ratio=0.5, pattern_size='raw'):  # pattern formed in raw data size
    assert modamp == 0.5
    assert center_ratio == 0.5
    result_list = []
    for idx_bs in range(raw_input_data.shape[0]):
        para = para_data[idx_bs]
        Nw, Nh = raw_input_data[idx_bs].shape[-1], raw_input_data[idx_bs].shape[-2]
        k0 = para[0:6].reshape(3, 2)
        if len(para) == 13:  # 2beams
            phase_list = para[9:12]
        elif len(para) == 19:  # 3beams
            phase_list = para[12:18].reshape(3, 2)
        else:
            raise NotImplementedError
        k0_angle = torch.atan2(k0[:, 1], k0[:, 0])
        k0_mag = torch.sqrt(torch.sum(torch.square(k0), 1))
        opt = sir_parse2dict(json_path[idx_bs])
        ndirs, nphases = opt['num_orientation'], opt['num_phase']
        if pattern_size == 'raw':
            w = 2 * pi * opt['width_space_sampling'] * torch.arange(-Nw // 2, Nw // 2, 1, device=device)
            h = 2 * pi * opt['height_space_sampling'] * torch.arange(-Nh // 2, Nh // 2, 1, device=device)
        elif pattern_size == 'sim':
            w = pi * opt['width_space_sampling'] * torch.arange(-Nw, Nw, 1, device=device)
            h = pi * opt['height_space_sampling'] * torch.arange(-Nh, Nh, 1, device=device)
        else:
            raise NotImplementedError
        [H, W] = my_meshgrid(h, w)
        gen_raw_list = []
        for idx_ori in range(ndirs):
            for idx_pha in range(nphases):
                angle = k0_angle[idx_ori]
                mag = k0_mag[idx_ori]
                Irtest = torch.cos(angle) * W + torch.sin(angle) * H
                if len(para) == 13:
                    patternOriPha = 1.0 + 2 * modamp * torch.cos(mag * Irtest - phase_list[idx_ori] + idx_pha * 2 * pi / nphases)
                else:  # generate radial pattern (axial pattern is bound to OTF)
                    center_ratio = center_ratio
                    """
                    Updates: 2022-1014 change center_ratio from constant (1.0) into a tunable var (default 0.5).

                                              [kxL kxC kxR]
                    | exp ( 1j * [dx, dy, dz] [kyL kyC kyR] ) | ** 2
                                              [kzL kzC kzR]
                    where
                        kxL =   2 * pi * exNA * cos(alpha) / exWave
                        kyL =   2 * pi * exNA * sin(alpha) / exWave
                        kzL =   2 * pi * nimm / exWave * cos(theta)
                        kxR = - 2 * pi * exNA * cos(alpha) / exWave
                        kyR = - 2 * pi * exNA * sin(alpha) / exWave
                        kzR =   2 * pi * nimm / exWave * cos(argsin(exNA / nimm))
                        kxC =   0
                        kyC =   0
                        kzC =   2 * pi * nimm / exWave

                    where alpha is attitude angle. Consider SLM focus position only, and using |exp[j*alpha, c*j*theta, j*beta]| = 
                          1 + 1 + c**2 + 2c * (cos(theta-alpha) + cos(theta-beta)) + 2 * cos(alpha-beta), we have: 
                    """
                    patternOriPha = 1.0 + 1.0 + center_ratio ** 2 + \
                                    2.0 * torch.cos(2 * mag * Irtest - phase_list[idx_ori, 1] + idx_pha * 4 * pi / nphases) + \
                                    4.0 * center_ratio * torch.cos(1 * mag * Irtest - phase_list[idx_ori, 0] + idx_pha * 2 * pi / nphases)
                    patternOriPha /= (1 + 1 + center_ratio ** 2)
                gen_raw_list.append(patternOriPha)
        pattern = torch.stack(gen_raw_list)
        pattern[pattern < 0] = 0
        pattern /= pattern.max()
        result_list.append(pattern)  # [C, H, W]
    return torch.stack(result_list).to(raw_input_data.dtype)  # [B, C, H, W]


# ----------------------------------------
# PixelUnShuffle
# ----------------------------------------
class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        S = self.upscale_factor
        B, C, H, W = x.shape
        assert H % S == 0 and W % S == 0
        input_view = x.contiguous().view(B, C, H // S, S, W // S, S)
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(B, C * S ** 2, H // S, W // S)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class PixelUnShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle3d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        S = self.upscale_factor
        B, C, D, H, W = x.shape
        assert H % S == 0 and W % S == 0
        input_view = x.contiguous().view(B, C, D, H // S, S, W // S, S)
        unshuffle_out = input_view.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        return unshuffle_out.view(B, C * S ** 2, D, H // S, W // S)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        S = self.upscale_factor
        B, C, D, H, W = x.shape
        assert C % (S ** 2) == 0
        input_view = x.contiguous().view(B, C // (S ** 2), S, S, D, H, W)
        shuffle_out = input_view.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return shuffle_out.view(B, C // (S ** 2), D, H * S, W * S)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# ----------------------------------------
#  *调制注意力机制*
# ----------------------------------------
class Modulation(nn.Module):  # 3
    def __init__(self, n_feat, nc=9, scale=2, bias=True, att_op='mul', skip_op='noskip', sigm_act=True):  # scale = p.shape[-1] / x.shape[-1]
        super(Modulation, self).__init__()
        self.scale = scale
        self.att_op = att_op
        self.skip_op = skip_op
        self.sigm_act = sigm_act
        if self.scale == 1:  # raw-sam
            self.conv2 = nn.Conv2d(n_feat, nc, kernel_size=(3, 3), padding=(1, 1), bias=bias, stride=(1, 1))
            self.conv3 = nn.Conv2d(nc, n_feat, kernel_size=(3, 3), padding=(1, 1), bias=bias, stride=(1, 1))
        elif self.scale in [2, 3]:  # 必须用Shuffle或者转置卷积做上采样，不能直接插值
            # 48->12 12->9
            self.conv2 = nn.Sequential(nn.PixelShuffle(self.scale), nn.Conv2d(n_feat // (self.scale ** 2), nc, kernel_size=(3, 3), padding=(1, 1), bias=bias, stride=(1, 1)))
            # 9->36 36->48
            self.conv3 = nn.Sequential(PixelUnShuffle(self.scale), nn.Conv2d(nc * self.scale ** 2, n_feat, kernel_size=(3, 3), padding=(1, 1), bias=bias, stride=(1, 1)))
        else:
            raise NotImplementedError

    def forward(self, x, p):
        """
        经测试att_op/skip_op/sigm_act对结果的影响均很小
        """
        x1 = self.conv2(x)
        if self.att_op == 'mul':
            x2 = x1 * p
        elif self.att_op == 'plus':
            x2 = x1 + p
        else:
            raise NotImplementedError
        x3 = self.conv3(x2)
        if self.sigm_act:
            x3 = torch.sigmoid(x3)
        if self.skip_op == 'mul':
            x3 = x3 * x
        elif self.skip_op == 'plus':
            x3 = x3 + x
        elif self.skip_op == 'noskip':
            x3 = x3
        else:
            raise NotImplementedError
        return x3


class Modulation_3D(nn.Module):  # 3
    def __init__(self, n_feat, nc=15, scale=1, conv=True, bias=False, att_op='mul', skip_op='mul', sigm_act=True, group=False):  # scale = p.shape[-1] / x.shape[-1]
        super(Modulation_3D, self).__init__()
        self.scale = scale
        self.att_op = att_op
        self.skip_op = skip_op
        self.sigm_act = sigm_act
        if self.scale == 1:  # raw-sam
            if conv:
                if group:
                    self.conv2 = nn.Conv3d(n_feat, nc, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias, groups=nc)
                    self.conv3 = nn.Conv3d(nc, n_feat, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias, groups=n_feat)
                else:
                    self.conv2 = nn.Conv3d(n_feat, nc, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias)
                    self.conv3 = nn.Conv3d(nc, n_feat, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias)
            else:
                self.conv2 = nn.Identity()
                self.conv3 = nn.Identity()
        elif self.scale in [2]:  # x2太费显存
            raise NotImplementedError

    def forward(self, x, p):
        p_3d = torch.tile(p.unsqueeze(2), [1, 1, x.shape[2], 1, 1])
        x1 = self.conv2(x)
        if self.att_op == 'mul':
            x2 = x1 * p_3d
        elif self.att_op == 'plus':
            x2 = x1 + p_3d
        else:
            raise NotImplementedError
        x3 = self.conv3(x2)
        if self.sigm_act:
            x3 = torch.sigmoid(x3)
        if self.skip_op == 'mul':
            x3 = x3 * x
        elif self.skip_op == 'plus':
            x3 = x3 + x
        elif self.skip_op in ['noskip', 'no_skip', 'no-skip']:
            x3 = x3
        else:
            raise NotImplementedError
        return x3


# ----------------------------------------
#          <Upsample3D>
# ----------------------------------------
class Upsample_3D(nn.Module):
    def __init__(self, scale_factor=(1, 2, 2)):
        super(Upsample_3D, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor[0] == 1:
            B, C, D, H, W = x.shape
            x = x.reshape(B, C * D, H, W)
            x = F.interpolate(x, scale_factor=self.scale_factor[-2:], mode='bilinear', align_corners=False)
            x = x.reshape(B, C, D, int(self.scale_factor[-2] * H), int(self.scale_factor[-1] * W))
        else:
            raise NotImplementedError
        return x


# ----------------------------------------
#          <init>
# ----------------------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.")

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: # (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
