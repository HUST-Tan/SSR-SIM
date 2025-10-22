import os
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d
import scipy.interpolate as si
from math import ceil, sqrt

from utils.option import product_of_tuple_elements
from utils.mrc import ReadMRC
from utils.tools import my_meshgrid
from utils.pytorch_interp1 import interp1

from SIR_core.function_for_NN import conventional_reconstruction_in_neural_network
from SSR_model.common import generate_pattern

device = torch.device('cuda')


class ModelBase(object):
    def __init__(self, opt):
        self.opt = opt  # opt
        self.save_dir = os.path.join(opt['save_path'], 'model')  # save models
        self.device = torch.device('cuda')  # single gpu | distributed training is not implemented
        self.is_train = opt['is_train']  # training or not
        self.schedulers = {}  # schedulers
        self.factor = 1.33  # 1 + num_phase / num_phase # it is a MYSTERY in the original program

        # ------------------------------------
        # data
        # ------------------------------------
        self.para_data = None
        self.json_path = None
        self.wf_input_data = None
        self.wf_target_data = None
        self.wf_infer_data = None
        self.raw_input_data = None
        self.raw_target_data = None
        self.raw_infer_data = None
        self.sim_input_data = None
        self.sim_target_data = None
        self.sim_infer_data = None
        # self.raw_input_pattern = None
        self.raw_target_pattern = None
        self.raw_infer_pattern = None
        self.predemod_pattern = None
        self.predemod_patternpsf = None

        self.OTF_TRAIN = None

        # ------------------------------------
        # log
        # ------------------------------------
        self.log_dict = None

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    @staticmethod
    def save_network(save_dir, network, network_label, iter_label):
        save_filename = '{:0>6}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    @staticmethod
    def load_network(load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self, net_list):
        return "\n".join(self.describe_network(net) for net in net_list)

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self, net_list):
        return "\n".join(self.describe_params(net) for net in net_list)

    # ----------------------------------------
    # learning rate during training
    # ----------------------------------------
    def current_learning_rate(self):
        dict_out = {}
        for key in self.schedulers.keys():
            dict_out[key] = self.schedulers[key].get_last_lr()[0]
        return dict_out

    def update_learning_rate(self, net):
        self.schedulers[net].step()

    # ----------------------------------------
    # cal loss for <tensor> or <list of tensor>
    # ----------------------------------------
    @staticmethod
    def cal_loss(infer_data_list, target_data, lossfn):
        if isinstance(infer_data_list, list):  # progressive learning
            if isinstance(target_data, list):  # sim-raw
                factor = 4 / target_data[1].shape[1]
                infer_data = infer_data_list[0]
                G_loss = lossfn(infer_data_list[0], target_data[0]) + factor * lossfn(infer_data_list[1], target_data[1])
                G_loss /= len(infer_data_list)
            else:  # sim-sim
                infer_data = infer_data_list[0]
                G_loss = lossfn(infer_data_list[0], target_data) + lossfn(infer_data_list[1], target_data)
                G_loss /= len(infer_data_list)
        else:
            infer_data = infer_data_list
            G_loss = lossfn(infer_data_list, target_data)
        return G_loss, infer_data

    # ----------------------------------------
    # do reconstruction
    # ----------------------------------------
    @staticmethod
    def conv_rec(input_data, para_data, num_channels_in, opt_class):
        # num_channels_in: OPC
        in_shape = input_data.shape
        if len(in_shape) == 3:  # 2D [CHW]
            T = 1
            O, P, C = num_channels_in
            D = 1
            _, H, W = in_shape
            input_data = input_data.reshape(T, O, P, C, D, H, W)  # [ChannelHW] -> [TOPCDHW]
        elif len(in_shape) == 4:  # 3D  [CDHW]
            if in_shape[0] == 15:
                T = 1
                O, P, C = num_channels_in
                _, D, H, W = in_shape
                input_data = input_data.reshape(T, O, P, C, D, H, W)  # [ChannelDHW] -> [TOPCDHW]
            elif in_shape[0] == 9:
                _, T, H, W = in_shape
                O, P, C = num_channels_in
                D = 1
                input_data = input_data.permute(1, 0, 2, 3).reshape(T, O, P, C, D, H, W)  # [ChannelTHW] -> [TChannelHW] -> [TOPCDHW]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return conventional_reconstruction_in_neural_network(input_data, para_data, opt_class)

    @staticmethod
    def tensor2format(img, out_shape):
        img = img.float()
        T = 1
        (O, P, C) = out_shape
        (H, W) = (img.shape[-2], img.shape[-1])
        D = product_of_tuple_elements(img.shape) // product_of_tuple_elements((T, O, P, C, H, W))
        return img.reshape(T, O, P, C, D, H, W)

    # ----------------------------------------
    # generate pattern in raw data size - for two stream raw denoise network
    # ----------------------------------------
    @staticmethod
    def _generate_pattern(raw_input_data, para_data, json_path, modamp=0.5, center_ratio=0.5, pattern_size='raw'):  # pattern formed in raw data size
        return generate_pattern(raw_input_data, para_data, json_path, modamp, center_ratio, pattern_size)

    # ----------------------------------------
    # obtain OTF
    # ----------------------------------------
    def set_otf_while_training(self, opt, Nx, Ny, Nz=1, state='test'):  # for 2d only
        if state == 'train' and self.OTF_TRAIN is not None:
            # Assume training data use the same OTF
            return self.OTF_TRAIN
        else:
            if self.opt['data'] == '2d-sim':

                try:
                    rm_OTF = ReadMRC(opt['otf_path'], is_SIM_rawdata=False)
                except FileNotFoundError:
                    rm_OTF = ReadMRC(os.path.join(*opt['otf_path'].split('\\')[1:]), is_SIM_rawdata=False)
                rawOTF = rm_OTF.get_total_data_as_mat(convert_to_tensor=False).squeeze()
                nxotf, dkrotf = rm_OTF.opt['num_pixel_width'], rm_OTF.opt['height_space_sampling']
                diagdist = int(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 2)
                OTF = np.real(rawOTF)
                x = np.arange(0, nxotf * dkrotf, dkrotf)
                dkx = 1 / (Nx * opt['width_space_sampling'])
                dky = 1 / (Ny * opt['height_space_sampling'])
                dkr = min(dkx, dky)
                xi = np.arange(0, (nxotf - 1) * dkrotf, dkr)
                interp = scipy_interp1d(x, OTF, kind='slinear')
                OTF = interp(xi)
                sizeOTF = len(OTF)
                prol_OTF = np.zeros((diagdist * 2))
                prol_OTF[0:sizeOTF] = OTF
                OTF = prol_OTF
                kxx = dkx * np.arange(-Nx / 2, Nx / 2, 1)
                kyy = dky * np.arange(-Ny / 2, Ny / 2, 1)
                [dX, dY] = np.meshgrid(kxx, kyy)
                rdist = np.sqrt(np.square(dX) + np.square(dY))
                otflen = len(OTF)
                x = np.arange(0, otflen * dkr, dkr)
                interp = scipy_interp1d(x, OTF, kind='slinear')
                OTF = interp(rdist)
                OTF = torch.from_numpy(OTF).to(device).float()
                OTF = OTF / OTF.max()
                return OTF

            elif self.opt['data'] == 'single-slice-sim':

                otf3d_class = ReadMRC(os.path.join(os.getcwd(), opt['otf_path']), is_SIM_rawdata=False)
                otf3d_data = otf3d_class.get_total_data_as_mat(convert_to_tensor=False).squeeze().transpose(2, 1, 0)
                nyotf = otf3d_class.opt['num_pixel_height']  # this dim is in fact depth pixel
                nzotf = otf3d_class.opt['num_pixel_depth']  # this dim is in fact order
                dkrotf = otf3d_class.opt['height_space_sampling']  # space sampling saved in mrc file is in fact freq sampling w.r.t. image
                rawOTF = np.fft.fftshift(otf3d_data, axes=0)
                if rawOTF.shape[0] % 2 == 1:  # odd number
                    rawOTF_mid = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(torch.from_numpy(rawOTF), dim=[0]), dim=[0]), dim=[0])[rawOTF.shape[0] // 2].numpy()
                else:
                    rawOTF_mid1 = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(torch.from_numpy(rawOTF), dim=[0]), dim=[0]), dim=[0])[rawOTF.shape[0] // 2]
                    rawOTF_mid2 = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(torch.from_numpy(rawOTF), dim=[0]), dim=[0]), dim=[0])[rawOTF.shape[0] // 2 - 1]
                    rawOTF_mid1_absMax = torch.abs(rawOTF_mid1).max()
                    rawOTF_mid2_absMax = torch.abs(rawOTF_mid2).max()
                    if rawOTF_mid1_absMax > rawOTF_mid2_absMax:
                        rawOTF_mid = rawOTF_mid1.numpy()
                    else:
                        rawOTF_mid = rawOTF_mid2.numpy()
                x = np.arange(0, nyotf * dkrotf, dkrotf)

                dkx = 1 / (Nx * opt['width_space_sampling'])
                dky = 1 / (Ny * opt['height_space_sampling'])
                dkr = min(1 / (Nx * opt['width_space_sampling']), 1 / (Ny * opt['height_space_sampling']))

                xi = np.arange(0, (nyotf - 1) * dkrotf, dkr)
                otfsingleslice = np.zeros((xi.shape[0], nzotf), np.complex64)
                for ii in range(nzotf):
                    otfsingleslice[:, ii] = si.interp1d(x, np.real(rawOTF_mid[:, ii]), kind='cubic', copy=False, bounds_error=False, fill_value=0)(xi) + \
                                            1j * si.interp1d(x, np.imag(rawOTF_mid[:, ii]), kind='cubic', copy=False, bounds_error=False, fill_value=0)(xi)  # quadratic
                diagdist = ceil(sqrt((opt['num_pixel_width'] / 2) ** 2 + (opt['num_pixel_height'] / 2) ** 2) + 1)
                otfsingleslice = torch.from_numpy(otfsingleslice.astype(np.complex64)).to(device).unsqueeze(0)
                otfsingleslice_afterpad = torch.zeros((otfsingleslice.shape[0], diagdist + ceil(1.26 / 0.488 / dkr) * (opt['num_order'] - 1) * 4, otfsingleslice.shape[2]),
                                                      dtype=otfsingleslice.dtype, device=device)
                otfsingleslice_afterpad[:, :otfsingleslice.shape[1], :] = otfsingleslice
                otfsingleslice_afterpad /= torch.abs(otfsingleslice_afterpad).max()
                otf = otfsingleslice_afterpad.permute(2, 1, 0)
                OTF = []
                for i in range(3):
                    OTFtemp = otf[i, :, 0].clone()
                    kxx = dkx * torch.arange(-Nx / 2, Nx / 2, 1, device=device)
                    kyy = dky * torch.arange(-Ny / 2, Ny / 2, 1, device=device)
                    [dY, dX] = my_meshgrid(kyy, kxx)
                    rdist = torch.sqrt(torch.square(dX) + torch.square(dY))
                    x = torch.arange(0, len(OTFtemp), 1, device=device) * dkr
                    OTFtemp = interp1(x, OTFtemp, rdist)
                    OTFtemp /= torch.abs(OTFtemp).max()
                    OTF.append(OTFtemp)
                return OTF

            elif self.opt['data'] == '3d-sim':

                otf_class = ReadMRC(os.path.join(os.getcwd(), opt['otf_path']), is_SIM_rawdata=False)
                otf_data = otf_class.get_total_data_as_mat(convert_to_tensor=False).squeeze().transpose(2, 1, 0)
                nxotf = otf_class.opt['num_pixel_width']  # this dim is in fact radial pixel
                nyotf = otf_class.opt['num_pixel_height']  # this dim is in fact depth pixel
                nzotf = otf_class.opt['num_pixel_depth']  # this dim is in fact order
                dkzotf = otf_class.opt['width_space_sampling']  # space sampling saved in mrc file is in fact freq sampling w.r.t. image
                dkrotf = otf_class.opt['height_space_sampling']  # space sampling saved in mrc file is in fact freq sampling w.r.t. image
                rawOTF = np.fft.fftshift(otf_data, axes=0)
                z = np.arange(0, nxotf * dkzotf, dkzotf) - (nxotf - 1) * dkzotf / 2
                x = np.arange(0, nyotf * dkrotf, dkrotf)

                nz = Nz
                dkx = 1 / (Nx * opt['width_space_sampling'])
                dky = 1 / (Ny * opt['height_space_sampling'])
                dkz = 1 / (Nz * opt['depth_space_sampling'])
                dkr = min(1 / (Nx * opt['width_space_sampling']), 1 / (Ny * opt['height_space_sampling']))
                zi = np.arange(0, nz * dkz, dkz) - (nz - 1) * dkz / 2
                xi = np.arange(0, (nyotf - 1) * dkrotf, dkr)
                otf3d = np.zeros((zi.shape[0], xi.shape[0], nzotf), np.complex64)
                for ii in range(nzotf):
                    otf3d[:, :, ii] = si.interp2d(x, z, np.real(rawOTF)[:, :, ii], kind='cubic', copy=False, bounds_error=False, fill_value=0)(xi, zi) + \
                                      1j * si.interp2d(x, z, np.imag(rawOTF)[:, :, ii], kind='cubic', copy=False, bounds_error=False, fill_value=0)(xi, zi)
                # diagdist = ceil(sqrt((opt['num_pixel_width'] / 2) ** 2 + (opt['num_pixel_height'] / 2) ** 2) + 1)
                diagdist = ceil(sqrt((Nx / 2) ** 2 + (Ny / 2) ** 2) + 1)
                otf3d = torch.from_numpy(otf3d.astype(np.complex64)).to(device)
                otf_afterpad = torch.zeros((otf3d.shape[0], diagdist + ceil(1.26 / 0.488 / dkr) * (opt['num_order'] - 1) * 4, otf3d.shape[2]), dtype=otf3d.dtype, device=device)
                otf_afterpad[:, :otf3d.shape[1], :] = otf3d
                otf_afterpad /= torch.abs(otf_afterpad).max()
                otf = otf_afterpad.permute(2, 1, 0)

                OTF = []
                for i in range(3):
                    OTFtemp = []
                    for zi in range(otf.shape[2]):
                        OTFtempzi = otf[i, :, zi].clone()
                        kxx = dkx * torch.arange(-Nx / 2, Nx / 2, 1, device=device)
                        kyy = dky * torch.arange(-Ny / 2, Ny / 2, 1, device=device)
                        [dY, dX] = my_meshgrid(kyy, kxx)
                        rdist = torch.sqrt(torch.square(dX) + torch.square(dY))
                        x = torch.arange(0, len(OTFtempzi), 1, device=device) * dkr
                        OTFtempzi = interp1(x, OTFtempzi, rdist)
                        OTFtemp.append(OTFtempzi)
                    OTFtemp = torch.stack(OTFtemp)  # D, H, W
                    OTFtemp /= torch.abs(OTFtemp).max()
                    OTF.append(OTFtemp)
                return OTF

            else:

                raise RuntimeError
