import torch.nn as nn
import torch
import time

from SSR_model.common import generate_pattern, Modulation_3D, Upsample_3D
from SSR_model.base_phct import RLFB_3D, ESA_3D, TransformerBlock_3D, WithBias_LayerNorm


class MakeGroup(nn.Module):
    def __init__(self, n_feats, group='conv3'):
        super(MakeGroup, self).__init__()
        block_list = []
        for s in group.lower().split('+'):
            num = int(s[:s.find('*')]) if s.find('*') >= 0 else 1
            if num >= 1:
                ss = s[s.find('*') + 1:]

                if ss == 'conv1':
                    assert num == 1
                    block_list.append(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)))
                elif ss == 'conv3':
                    assert num == 1
                    block_list.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)))

                elif ss == 'esa':
                    block_list += [ESA_3D(n_feats) for _ in range(num)]
                elif ss == 'rlfb':
                    block_list += [RLFB_3D(n_feats) for _ in range(num)]

                elif ss == 'et':
                    block_list += [TransformerBlock_3D(n_feats, 8, 2.66, False, WithBias_LayerNorm) for _ in range(num)]

                else:
                    print(ss)
                    raise NotImplementedError

        self.body = nn.Sequential(*block_list)

    def forward(self, x):
        x = self.body(x)
        return x


class PHCT_3D_V1(nn.Module):
    def __init__(self, in_nc, out_nc, para1, para2, para3, para4, scale=2):
        super(PHCT_3D_V1, self).__init__()

        before_modulation_attention = para1

        after_modulation_attention = para2

        if not para3:
            print('ablation mode - not using modulation')
            time.sleep(2)
            self.use_modulation = False
        else:
            self.use_modulation = True

        n_feats = para4

        # define body module
        self.head = nn.Conv3d(in_nc, n_feats, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.body_before = MakeGroup(n_feats, before_modulation_attention)

        # define head module
        self.att = Modulation_3D(n_feats, scale=1, conv=True, bias=False, att_op='mul', skip_op='mul', sigm_act=True, group=False)

        # define body module
        self.body_after = MakeGroup(n_feats, after_modulation_attention)

        # define tail module
        if scale == 1:
            self.tail = nn.Conv3d(n_feats, out_nc, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        elif scale == 2:
            self.tail = nn.Sequential(
                nn.Conv3d(n_feats, n_feats, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                Upsample_3D(scale_factor=(1, 2, 2)),
                nn.Conv3d(n_feats, out_nc, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )
        else:
            raise NotImplementedError

    def forward(self, x, para, json):

        x = self.head(x)

        res = self.body_before(x)

        if self.use_modulation:
            praw = generate_pattern(res, para, json, center_ratio=0.5, pattern_size='raw')
            res = self.att(res, praw)

        res = self.body_after(res)
        res += x
        res = self.tail(res)
        return res

    def forward_with_pattern(self, x, praw=None, psim=None):

        x = self.head(x)

        res = self.body_before(x)

        if self.use_modulation:
            res = self.att(res, praw)

        res = self.body_after(res)
        res += x
        res = self.tail(res)
        return res


class PHCT_3D_V2(nn.Module):
    # 去掉了body1，把调制放最前面
    def __init__(self, in_nc, out_nc, para1, para2, para3, para4, scale=2):
        super(PHCT_3D_V2, self).__init__()

        before_modulation_attention = para1

        after_modulation_attention = para2

        if not para3:
            print('ablation mode - not using modulation')
            time.sleep(2)
            self.use_modulation = False
        else:
            self.use_modulation = True

        n_feats = para4

        # define head module
        self.att = Modulation_3D(in_nc, scale=1, conv=True, bias=False, att_op='mul', skip_op='mul', sigm_act=True, group=True)

        # define body module
        self.head = nn.Conv3d(2 * in_nc, n_feats, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        # define body module
        self.body = MakeGroup(n_feats, after_modulation_attention)

        # define tail module
        if scale == 1:
            self.tail = nn.Conv3d(n_feats, out_nc, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        elif scale == 2:
            self.tail = nn.Sequential(
                nn.Conv3d(n_feats, n_feats, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                Upsample_3D(scale_factor=(1, 2, 2)),
                nn.Conv3d(n_feats, out_nc, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )
        else:
            raise NotImplementedError

    def forward(self, x, para, json):

        if self.use_modulation:
            praw = generate_pattern(x, para, json, center_ratio=0.5, pattern_size='raw')
            px = self.att(x, praw)
            x = torch.cat((x, px), dim=1)

        x = self.head(x)

        res = self.body(x)
        res += x
        res = self.tail(res)
        return res

    def forward_with_pattern(self, x, praw=None, psim=None):

        if self.use_modulation:
            px = self.att(x, praw)
            x = torch.cat((x, px), dim=1)

        x = self.head(x)

        res = self.body(x)
        res += x
        res = self.tail(res)
        return res


class PHCT_3D_V3(nn.Module):
    # 去掉att中的卷积，变为纯调制
    # NFeat=48, Seq2='3*rlfb+et+conv3'
    # paras: 0.803707 M
    # GFLOPs: 1638.189146112
    # [4090] fps 5.15492048059308 Hz
    # [4090] time 193.98941337014548 ms
    def __init__(self, in_nc, out_nc, para1, para2, para3, para4, scale=2):
        super(PHCT_3D_V3, self).__init__()

        before_modulation_attention = para1

        after_modulation_attention = para2

        if not para3:
            print('ablation mode - not using modulation')
            time.sleep(2)
            self.use_modulation = False
        else:
            self.use_modulation = True

        n_feats = para4

        # define head module
        self.att = Modulation_3D(in_nc, scale=1, conv=False, bias=False, att_op='mul', skip_op='noskip', sigm_act=False, group=None)

        # define body module
        self.head = nn.Conv3d(2 * in_nc, n_feats, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        # define body module
        self.body = MakeGroup(n_feats, after_modulation_attention)

        # define tail module
        if scale == 1:
            self.tail = nn.Conv3d(n_feats, out_nc, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        elif scale == 2:
            self.tail = nn.Sequential(
                nn.Conv3d(n_feats, n_feats, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                Upsample_3D(scale_factor=(1, 2, 2)),
                nn.Conv3d(n_feats, out_nc, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            )
        else:
            raise NotImplementedError

    def forward(self, x, para, json):

        if self.use_modulation:
            praw = generate_pattern(x, para, json, center_ratio=0.5, pattern_size='raw')
            px = self.att(x, praw)
            x = torch.cat((x, px), dim=1)
        else:
            x = torch.cat((x, x), dim=1)

        x = self.head(x)

        res = self.body(x)
        res += x
        res = self.tail(res)
        return res

    # def forward(self, x):
    #
    #     para, json = None, None
    #
    #     if self.use_modulation:
    #         praw = generate_pattern(x, para, json, center_ratio=0.5, pattern_size='raw')
    #         px = self.att(x, praw)
    #         x = torch.cat((x, px), dim=1)
    #     else:
    #         x = torch.cat((x, x), dim=1)
    #
    #     x = self.head(x)
    #
    #     res = self.body(x)
    #     res += x
    #     res = self.tail(res)
    #     return res

    def forward_with_pattern(self, x, praw=None, psim=None):

        if self.use_modulation:
            px = self.att(x, praw)
            x = torch.cat((x, px), dim=1)

        x = self.head(x)

        res = self.body(x)
        res += x
        res = self.tail(res)
        return res


PHCT_3D = PHCT_3D_V3

if __name__ == '__main__':

    from thop import profile

    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True

    model = PHCT_3D(15, 1, None, '3*rlfb+et+conv3', False, 48, scale=2)
    model = model.to(device)
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False

    N = 100
    a = torch.randn((N, 1, 15, 8, 512, 512), dtype=torch.float32, device=device)  # 1um * 31um * 31um

    flops, params = profile(model, inputs=(a[0],))
    print("paras: {} M, GFlops: {}".format(params / 10 ** 6, flops / 10 ** 9))

    # with torch.no_grad():
    #     for idx in range(N):
    #         b = model(a[idx], None, None)
    #
    # with torch.no_grad():
    #     torch.cuda.synchronize()
    #     start = time.perf_counter()
    #     for idx in range(N):
    #         b = model(a[idx], None, None)
    #     torch.cuda.synchronize()
    #     end = time.perf_counter()
    #     print('time: {} ms'.format(1000 * (end - start) / N))
    #     print('fps: {} Hz'.format(N / (end - start)))
