import functools
from torch.nn import init


def define_G(opt):
    if opt['data'] in ['2d-sim']:

        # --------------------------------------------
        # 2d reconstruction
        # --------------------------------------------
        if opt['net_G'] == 'phct':
            from SSR_model.network_phct import PHCT
            netG = PHCT(in_nc=opt['net_channel_in'], out_nc=opt['net_channel_out'], scale=opt['net_scale'], para1=opt['net_G_para1'], para2=opt['net_G_para2'], para3=opt['net_G_para3'],
                        para4=opt['net_G_para4'])

        else:
            print(opt['net_G'])
            raise NotImplementedError

    else:
        raise NotImplementedError

    if opt['is_train']:
        if opt['init_type']:
            init_weights(netG, init_type=opt['init_type'], gain=opt['init_gain'])
            opt['outinfo'].info('Initialization method [{:s}], bn [{:s}], gain is [{:.2f}]\n'.format(opt['init_type'], 'uniform', opt['init_gain']))
        else:
            opt['outinfo'].info('Warning! Do nothing in init_weights  |  The initialization method shoule be involved in network init code\n')
    return netG


# --------------------------------------------
# weights initialization
# --------------------------------------------
def init_weights(net, init_type='xavier_uniform', gain=0.2):
    def init_fn(m, init_type='kaiming_normal', gain=0.2):
        classname = m.__class__.__name__
        if classname in ['Conv2d', 'Conv3d', 'Linear']:  # and classname.find('_selfinit') == -1: # if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if init_type in ['default', 'none']:
                pass
            elif init_type == 'normal_default':
                init.normal_(m.weight.data, 0, 0.1)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_leaky_relu':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)
            elif init_type == 'kaiming_uniform_leaky_relu':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:
            if m.affine:
                init.uniform_(m.weight.data, 0.1, 1.0)
                init.constant_(m.bias.data, 0.0)

    if init_type not in ['default', 'none']:
        fn = functools.partial(init_fn, init_type=init_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
