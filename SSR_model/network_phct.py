import torch.nn as nn
import torch
import time

from SSR_model.common import generate_pattern, Modulation

from SSR_model.base_phct import RLFB, ESA, TransformerBlock, WithBias_LayerNorm

class MakeGroup(nn.Module):
    def __init__(self, n_feats, group='conv3'):
        super(MakeGroup, self).__init__()
        block_list = []
        for s in group.lower().split('+'):
            num = int(s[:s.find('*')]) if s.find('*') >= 0 else 1
            if num >= 1:
                ss = s[s.find('*')+1:]

                if ss == 'conv1':
                    assert num == 1
                    block_list.append(nn.Conv2d(n_feats, n_feats, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

                elif ss == 'conv3':
                    assert num == 1
                    block_list.append(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

                elif ss == 'esa':
                    block_list += [ESA(n_feats) for _ in range(num)]

                elif ss == 'rlfb':
                    block_list += [RLFB(n_feats) for _ in range(num)]

                elif ss == 'et':
                    block_list += [TransformerBlock(n_feats, 8, 2.66, False, WithBias_LayerNorm) for _ in range(num)]

                else:
                    print(ss)
                    raise NotImplementedError

        self.body = nn.Sequential(*block_list)

    def forward(self, x):
        x = self.body(x)
        return x

class PHCT(nn.Module):
    # NFeat=48, Seq1=Seq2='RLFB+ET+Conv3'
    # paras: 0.333677 M
    # GFLOPs: 85.47889664
    # [4090] fps 59.84015453671644 Hz
    # [4090] time 16.711186789907515 ms
    def __init__(self, in_nc, out_nc, para1, para2, para3, para4, scale=2):
        super(PHCT, self).__init__()

        before_modulation_attention = para1

        after_modulation_attention = para2

        self.modulation = para3 # True
        if not para3:
            print('ablation mode - not using modulation')
            time.sleep(2)

        n_feats = para4 if para4 is not None else 48

        # define head module
        self.head = nn.Conv2d(in_nc, n_feats, (3,3), (1,1), (1,1), bias=True)

        # define body module
        self.body1 = MakeGroup(n_feats, before_modulation_attention)
        if self.modulation: self.att = Modulation(n_feats, nc=in_nc, scale=2)
        self.body2 = MakeGroup(n_feats, after_modulation_attention)

        # define tail module
        if scale == 1:
            modules_tail = [nn.Conv2d(n_feats, n_feats, (3,3), (1,1), (1,1), bias=True), nn.Conv2d(n_feats, out_nc, (3,3), (1,1), (1,1), bias=True)]
        elif scale == 2:
            modules_tail = [nn.Conv2d(n_feats, n_feats * 4, (3,3), (1,1), (1,1), bias=True), nn.PixelShuffle(2), nn.Conv2d(n_feats, out_nc, (3,3), (1,1), (1,1), bias=True)]
        else:
            raise NotImplementedError
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, para, json):

        if self.modulation:
            # praw = generate_pattern(x, para, json, modamp=0.5, pattern_size='raw')
            psim = generate_pattern(x, para, json, modamp=0.5, pattern_size='sim')

        x = self.head(x)

        res = self.body1(x)
        if self.modulation: res = self.att(res, psim)
        res = self.body2(res)
        res += x

        res = self.tail(res)

        return res

    # def forward(self, x):
    #
    #     para, json = None, None
    #
    #     if self.modulation:
    #         # praw = generate_pattern(x, para, json, modamp=0.5, pattern_size='raw')
    #         psim = generate_pattern(x, para, json, modamp=0.5, pattern_size='sim')
    #
    #     x = self.head(x)
    #
    #     res = self.body1(x)
    #     if self.modulation: res = self.att(res, psim)
    #     res = self.body2(res)
    #     res += x
    #
    #     res = self.tail(res)
    #
    #     return res

    def forward_with_pattern(self, x, praw=None, psim=None):

        x = self.head(x)

        res = self.body1(x)
        if self.modulation: res = self.att(res, psim)
        res = self.body2(res)
        res += x

        res = self.tail(res)

        return res


if __name__ == '__main__':

    from thop import profile
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True

    model = PHCT(9, 1, 'rlfb+et+conv3', 'rlfb+et+conv3', False, 48, scale=2)
    model = model.to(device)
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False

    N = 1
    a = torch.randn((N, 1, 9, 512, 512), dtype=torch.float32, device=device)

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