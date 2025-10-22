import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange
import numbers


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), proposed in Jie Liu et al. "Residual feature aggregation network for image super-resolution", CVPR2020

    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes are deleted.
    """

    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = 16
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class ESA_3D(nn.Module):
    def __init__(self, n_feats):
        super(ESA_3D, self).__init__()
        f = 16
        self.conv1 = nn.Conv3d(n_feats, f, kernel_size=(1, 1, 1))
        self.conv_f = nn.Conv3d(f, f, kernel_size=(1, 1, 1))
        self.conv2 = nn.Conv3d(f, f, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.conv3 = nn.Conv3d(f, f, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(f, n_feats, kernel_size=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, _, H, W = x.shape
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool3d(c1, kernel_size=(1, 7, 7), stride=(1, 3, 3))
        c3 = self.conv3(v_max)
        # c3 = F.interpolate(c3, (x.size(2), x.size(3), x.size(4)), mode='trilinear', align_corners=False) # mode
        B_, C_, D_, H_, W_ = c3.shape
        c3 = c3.reshape(B_, C_ * D_, H_, W_)
        c3 = F.interpolate(c3, (H, W), mode='bilinear', align_corners=False)
        c3 = c3.reshape(B_, C_, D_, H, W)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """
    proposed by ByteESR Team in NTIRE 2022 Challenge on Efficient Super-Resolution
    """

    def __init__(self, n_feats, bias=True):
        super(RLFB, self).__init__()
        self.RB = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, (3, 3), (1, 1), (1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(n_feats, n_feats, (3, 3), (1, 1), (1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
            nn.Conv2d(n_feats, n_feats, (3, 3), (1, 1), (1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
        )
        self.C = nn.Conv2d(n_feats, n_feats, (1, 1), (1, 1), (0, 0), bias=bias)
        self.ESA = ESA(n_feats)

    def forward(self, x):
        res = self.RB(x)
        res = res + x
        res = self.C(res)
        res = self.ESA(res)
        return res


class RLFB_3D(nn.Module):
    """
    proposed by ByteESR Team in NTIRE 2022 Challenge on Efficient Super-Resolution
    """

    def __init__(self, n_feats, bias=True):
        super(RLFB_3D, self).__init__()
        self.RB_3D = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
            nn.Conv3d(n_feats, n_feats, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
            nn.Conv3d(n_feats, n_feats, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=bias),
            nn.LeakyReLU(0.05, True),
        )
        self.C_3D = nn.Conv3d(n_feats, n_feats, (1, 1, 1), (1, 1, 1), (0, 0, 0), bias=bias)
        self.ESA_3D = ESA_3D(n_feats)

    def forward(self, x):
        res = self.RB_3D(x)
        res = res + x
        res = self.C_3D(res)
        res = self.ESA_3D(res)
        return res


def to_3d(x):  # from_4d_to_3d
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):  # from_3d_to_4d
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def from_5d_to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')


def from_3d_to_5d(x, d, h, w):
    return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm_3D(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_3D, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return from_3d_to_5d(self.body(from_5d_to_3d(x)), d, h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b head c (h w)] \times [b head (h w) c] = [b head c c]
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # [b head c c] \times [b head c (h w)] = [b head c (h w)]

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention_3D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_3d = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv_3d = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=dim * 3, bias=bias)
        self.project_out_3d = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)

    def forward(self, x):
        b, c, d, h, w = x.shape

        qkv = self.qkv_dwconv_3d(self.qkv_3d(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, h=h, w=w)

        out = self.project_out_3d(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock_3D(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(TransformerBlock_3D, self).__init__()

        self.norm1_3d = LayerNorm_3D(dim, LayerNorm_type)
        self.attn_3d = Attention_3D(dim, num_heads, bias)
        self.norm2_3d = LayerNorm_3D(dim, LayerNorm_type)
        self.ffn_3d = GFeedForward_3D(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn_3d(self.norm1_3d(x))
        x = x + self.ffn_3d(self.norm2_3d(x))

        return x


## Gated-Dconv Feed-Forward Network (GDFN)
class GFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GFeedForward_3D(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward_3D, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in_3d = nn.Conv3d(dim, hidden_features * 2, kernel_size=(1, 1, 1), bias=bias)

        self.dwconv_3d = nn.Conv3d(hidden_features * 2, hidden_features * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=hidden_features * 2, bias=bias)

        self.project_out_3d = nn.Conv3d(hidden_features, dim, kernel_size=(1, 1, 1), bias=bias)

    def forward(self, x):
        x = self.project_in_3d(x)
        x1, x2 = self.dwconv_3d(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out_3d(x)
        return x


if __name__ == '__main__':
    m = TransformerBlock_3D(32).cuda()
    # x_ = torch.randn(1,32,9,128,128).cuda()
    # print(m(x_).shape)
