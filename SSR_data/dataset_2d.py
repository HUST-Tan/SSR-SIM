import random
import numpy as np
import torch.utils.data as data
import math

from SSR_data.common import get_npys_paths, read_npy_and_check, dictsingle2tensor, get_json_paths, rot_datadict, \
    rot_datadict_para, get_dataset, get_datadict_from_datarootdict, \
    single2tensor

ALLOW_NUM = 9999


def read_para(path):
    assert path[-3:] == 'npy', 'only .npy data are allowed while training'
    # pre-process npy files
    img_np = np.load(path)
    assert img_np.ndim == 1, '{}'.format(img_np)
    assert img_np.shape[0] in [13, 19], '{}'.format(img_np)
    return img_np


class Dataset_2d(data.Dataset):

    def __init__(self, opt, phase):
        super(Dataset_2d, self).__init__()
        self.opt = opt
        self.phase = phase
        self.dataroot = get_dataset(opt['dataset_path'], opt['dataset_use'], phase)
        if opt['phase_noise_ud_range'] is None:
            self.phase_noise_ud_range = 0
        else:
            self.phase_noise_ud_range = opt['phase_noise_ud_range']
            print('apply phase noise uniform distribution range {}'.format(self.phase_noise_ud_range))
        if phase == 'train':
            if self.opt['need_para']:
                self.paths_para = get_npys_paths(opt['para_path']['train_PARA'])
            self.paths_json = get_json_paths(opt['json_path']['train_JSON'])
        elif phase == 'val':
            if self.opt['need_para']:
                self.paths_para = get_npys_paths(opt['para_path']['val_PARA'])
            self.paths_json = get_json_paths(opt['json_path']['val_JSON'])
        else:
            raise NotImplementedError
        self.if_random_angle_rotate_aug = True if opt['dataaug_random_angle_rotate'] is None else opt['dataaug_random_angle_rotate']
        if self.if_random_angle_rotate_aug:
            self.rotate_ratio = 0.5 if opt['dataaug_random_angle_rotate_ratio'] is None else opt['dataaug_random_angle_rotate_ratio']

        self.read_whole_dataset_toCPU = self.opt['read_whole_dataset_toCPU']  # read all data and save in CPU memory
        self.read_whole_dataset_toGPU = self.opt['read_whole_dataset_toGPU']  # read all data and save in GPU memory
        self.read_whole_dataset = self.read_whole_dataset_toCPU or self.read_whole_dataset_toGPU
        if phase == 'train' and self.read_whole_dataset:
            self.dict_data_list = []
            length = len(self.dataroot[list(self.dataroot.keys())[0]])
            if length > ALLOW_NUM:
                if length >= 2 * ALLOW_NUM:
                    factor = math.ceil(len(self.dataroot[list(self.dataroot.keys())[0]]) / ALLOW_NUM)
                    dataset_list = list(range(length))[:length // factor * factor:factor]
                else:
                    dataset_list = random.sample(list(range(length)), ALLOW_NUM)
                    # dataset_list.sort()
            else:
                dataset_list = list(range(length))
            if self.opt['need_para']:
                temp_paths_para = []
            temp_paths_json = []
            for idx in dataset_list:
                if self.read_whole_dataset_toGPU:
                    print('read whole dataset into GPU memory, current {}, total {}'.format(idx + 1, length))
                else:
                    print('read whole dataset into CPU memory, current {}, total {}'.format(idx + 1, length))
                self.dict_data_list.append(
                    get_datadict_from_datarootdict(self.dataroot, idx=idx, ncr=self.opt['raw_shape_OPC'],
                                                   ncs=self.opt['sim_shape_OPC'],
                                                   ncw=self.opt['wf_shape_OPC'],
                                                   convert_to_CUDA=self.read_whole_dataset_toGPU)
                )
                if self.opt['need_para']:
                    temp_paths_para.append(self.paths_para[idx])
                temp_paths_json.append(self.paths_json[idx])
            if self.opt['need_para']:
                self.paths_para = temp_paths_para
            self.paths_json = temp_paths_json
        else:
            pass

    def __getitem__(self, index):
        # ------------------------------------
        # get input image, get para, get json
        # ------------------------------------
        if self.phase == 'train' and self.read_whole_dataset:
            # dict_data = deepcopy(self.dict_data_list[index])
            dict_data = self.dict_data_list[index].copy()
        else:
            dict_data = get_datadict_from_datarootdict(self.dataroot, idx=index, ncr=self.opt['raw_shape_OPC'],
                                                       ncs=self.opt['sim_shape_OPC'], ncw=self.opt['wf_shape_OPC'])
        if self.opt['need_para']: para = read_para(self.paths_para[index])
        json_path = self.paths_json[index]
        if self.phase == 'val':
            input_path = self.dataroot[list(self.dataroot.keys())[0]][index]

        S = self.opt['sim_scale']

        if self.phase == 'train':
            # --------------------------------
            # augmentation - prepare para
            # --------------------------------
            H, W = 0, 0
            ps = self.opt['in_patch_size']
            for key in dict_data.keys():
                if key.lower().find('raw') >= 0:
                    H, W = dict_data[key].shape[-2], dict_data[key].shape[-1]
                    break
            assert H * W > 0

            # --------------------------------
            # augmentation - random rotate - randomly crop the patch
            # --------------------------------
            rotate_flag = random.random()
            if self.if_random_angle_rotate_aug and H == W and 0.3 * H - 0.5 * ps > 0 and rotate_flag < self.rotate_ratio:  # 0.3 belongs to [0.214, 0.354]
                rotate_angle = 360 * random.random()
                if self.opt['need_para']:
                    dict_data, para = rot_datadict_para(dict_data, para, rotate_angle)
                else:
                    dict_data = rot_datadict(dict_data, rotate_angle)
                dx, dy = np.random.randint(-math.floor(0.3 * H - 0.5 * ps), math.floor(0.3 * H - 0.5 * ps) + 1), \
                         np.random.randint(-math.floor(0.3 * H - 0.5 * ps), math.floor(0.3 * H - 0.5 * ps) + 1)
                for key in dict_data.keys():
                    if key.lower().find('sim') >= 0:
                        assert key.lower().find('raw') == -1 and key.lower().find('wf') == -1
                        dict_data[key] = dict_data[key][...,
                                         (H - ps) // 2 * S + S * dx:(H - ps) // 2 * S + S * dx + S * ps,
                                         (W - ps) // 2 * S + S * dy:(W - ps) // 2 * S + S * dy + S * ps]
                    elif key.lower().find('raw') >= 0 or key.lower().find('wf') >= 0:
                        dict_data[key] = dict_data[key][...,
                                         (H - ps) // 2 + dx:(H - ps) // 2 + dx + ps,
                                         (W - ps) // 2 + dy:(W - ps) // 2 + dy + ps]
                    elif "predemodpattern" in key.lower() and "psf" not in key.lower():
                        dict_data[key] = dict_data[key][...,
                                         (H - ps) // 2 + dx:(H - ps) // 2 + dx + ps,
                                         (W - ps) // 2 + dy:(W - ps) // 2 + dy + ps]
                    elif "predemodpatternpsf" in key.lower():
                        pass
                    else:
                        raise NotImplementedError
                if self.opt['need_para']:
                    n_ori = 3
                    if len(para) == 13:  # 6 + 3 + 3 + 1
                        for idx_ori in range(n_ori):
                            if self.phase_noise_ud_range > 0:
                                para[9 + idx_ori] += 2 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[
                                    2 * idx_ori] * dy) + self.phase_noise_ud_range * (np.random.rand() - 0.5)
                            else:
                                para[9 + idx_ori] += 2 * np.pi * para[-1] * (
                                        - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                    elif len(para) == 19:  # 6 + 6 + 6 + 1
                        for idx_ori in range(n_ori):
                            if self.phase_noise_ud_range > 0:
                                noise = self.phase_noise_ud_range * (np.random.rand() - 0.5)
                                para[12 + 2 * idx_ori] += 2 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy) + noise / 2
                                para[13 + 2 * idx_ori] += 4 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy) + noise
                            else:
                                para[12 + 2 * idx_ori] += 2 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                                para[13 + 2 * idx_ori] += 4 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                    else:
                        raise NotImplementedError
            else:
                if self.opt['90degree'] is not None and self.opt['90degree'] is True:
                    rotate_angle = 90 * random.randint(0, 3)
                    if self.opt['need_para']:
                        dict_data, para = rot_datadict_para(dict_data, para, rotate_angle)
                    else:
                        dict_data = rot_datadict(dict_data, rotate_angle)
                if self.opt['train_border'] is not None:
                    # 这个参数专门为了特殊空间采样率的数据设置，为了防止取到边缘图像，其它时候取0-10均可
                    # 当H=ps时，该参数设置没有意义
                    dis = self.opt['train_border']
                else:
                    dis = 5
                if (H - ps) // 2 - dis > 0 + dis and (W - ps) // 2 - dis > 0:
                    dx, dy = np.random.randint(-(H - ps) // 2 + dis, (H - ps) // 2 + 1 - dis), np.random.randint(
                        -(W - ps) // 2 + dis, (W - ps) // 2 + 1 - dis)
                else:
                    dx, dy = 0, 0
                for key in dict_data.keys():
                    if key.lower().find('sim') >= 0:
                        assert key.lower().find('raw') == -1 and key.lower().find('wf') == -1
                        dict_data[key] = dict_data[key][...,
                                         (H - ps) // 2 * S + S * dx:(H - ps) // 2 * S + S * dx + S * ps,
                                         (W - ps) // 2 * S + S * dy:(W - ps) // 2 * S + S * dy + S * ps]
                    elif key.lower().find('raw') >= 0 or key.lower().find('wf') >= 0:
                        dict_data[key] = dict_data[key][...,
                                         (H - ps) // 2 + dx:(H - ps) // 2 + dx + ps,
                                         (W - ps) // 2 + dy:(W - ps) // 2 + dy + ps]
                    elif "predemodpattern" in key.lower() and "psf" not in key.lower():
                        dict_data[key] = dict_data[key][...,
                                         (H - ps) // 2 + dx:(H - ps) // 2 + dx + ps,
                                         (W - ps) // 2 + dy:(W - ps) // 2 + dy + ps]
                    elif "predemodpatternpsf" in key.lower():
                        pass
                    else:
                        raise NotImplementedError
                if self.opt['need_para']:
                    n_ori = 3
                    if len(para) == 13:
                        for idx_ori in range(n_ori):
                            if self.phase_noise_ud_range > 0:
                                para[9 + idx_ori] += 2 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[
                                    2 * idx_ori] * dy) + self.phase_noise_ud_range * (np.random.rand() - 0.5)
                            else:
                                para[9 + idx_ori] += 2 * np.pi * para[-1] * (- para[2 * idx_ori + 1] * dx - para[
                                    2 * idx_ori] * dy) + self.phase_noise_ud_range
                    elif len(para) == 19:
                        for idx_ori in range(n_ori):
                            if self.phase_noise_ud_range > 0:
                                noise = self.phase_noise_ud_range * (np.random.rand() - 0.5)
                                para[12 + 2 * idx_ori] += 2 * np.pi * para[-1] * (
                                        - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy) + noise / 2
                                para[13 + 2 * idx_ori] += 4 * np.pi * para[-1] * (
                                        - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy) + noise
                            else:
                                para[12 + 2 * idx_ori] += 2 * np.pi * para[-1] * (
                                        - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                                para[13 + 2 * idx_ori] += 4 * np.pi * para[-1] * (
                                        - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                    else:
                        raise NotImplementedError

            dict_data = dictsingle2tensor(dict_data)
            if self.opt['need_para']:
                para = single2tensor(para)
                dict_data.update({'para': para, 'json_path': json_path})
            else:
                dict_data.update({'json_path': json_path})

            return dict_data

        elif self.phase == 'val':

            # --------------------------------
            # augmentation - prepare para
            # --------------------------------
            H, W = 0, 0
            for key in dict_data.keys():
                if key.lower().find('raw') >= 0:
                    H, W = dict_data[key].shape[-2], dict_data[key].shape[-1]
                    break
            assert H * W > 0

            if H > 512 and W > 512:  # cuda out of memory
                H_ps, W_ps = 512, 512
                dx, dy = 0, 0
                for key in dict_data.keys():
                    if key.lower().find('sim') >= 0:
                        assert key.lower().find('raw') == -1 and key.lower().find('wf') == -1
                        dict_data[key] = dict_data[key][...,
                                         (H - H_ps) // 2 * S + S * dx:(H - H_ps) // 2 * S + S * dx + S * H_ps,
                                         (W - W_ps) // 2 * S + S * dy:(W - W_ps) // 2 * S + S * dy + S * W_ps]
                    elif key.lower().find('raw') >= 0 or key.lower().find('wf') >= 0:
                        dict_data[key] = dict_data[key][...,
                                         (H - H_ps) // 2 + dx:(H - H_ps) // 2 + dx + H_ps,
                                         (W - W_ps) // 2 + dy:(W - W_ps) // 2 + dy + W_ps]
                    elif "predemodpattern" in key.lower() and "psf" not in key.lower():
                        dict_data[key] = dict_data[key][...,
                                         (H - H_ps) // 2 + dx:(H - H_ps) // 2 + dx + H_ps,
                                         (W - W_ps) // 2 + dy:(W - W_ps) // 2 + dy + W_ps]
                    elif "predemodpatternpsf" in key.lower():
                        pass
                    else:
                        raise NotImplementedError
            else:
                factor = 2 ** 5
                H_ps, W_ps = H // factor * factor, W // factor * factor
                dx, dy = 0, 0
                # H_ps, W_ps = 384, 384
                # dx, dy = random.randint(-10,10), random.randint(-10,10)
                for key in dict_data.keys():
                    if key.lower().find('sim') >= 0:
                        assert key.lower().find('raw') == -1 and key.lower().find('wf') == -1
                        dict_data[key] = dict_data[key][...,
                                         (H - H_ps) // 2 * S + S * dx:(H - H_ps) // 2 * S + S * dx + S * H_ps,
                                         (W - W_ps) // 2 * S + S * dy:(W - W_ps) // 2 * S + S * dy + S * W_ps]
                    elif key.lower().find('raw') >= 0 or key.lower().find('wf') >= 0:
                        dict_data[key] = dict_data[key][...,
                                         (H - H_ps) // 2 + dx:(H - H_ps) // 2 + dx + H_ps,
                                         (W - W_ps) // 2 + dy:(W - W_ps) // 2 + dy + W_ps]
                    elif "predemodpattern" in key.lower() and "psf" not in key.lower():
                        dict_data[key] = dict_data[key][...,
                                         (H - H_ps) // 2 + dx:(H - H_ps) // 2 + dx + H_ps,
                                         (W - W_ps) // 2 + dy:(W - W_ps) // 2 + dy + W_ps]
                    elif "predemodpatternpsf" in key.lower():
                        pass
                    else:
                        raise NotImplementedError

            dict_data = dictsingle2tensor(dict_data)

            if self.opt['need_para']:
                n_ori = 3
                if len(para) == 13:
                    for idx_ori in range(n_ori):
                        para[9 + idx_ori] += 2 * np.pi * para[-1] * (
                                - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                elif len(para) == 19:
                    for idx_ori in range(n_ori):
                        para[12 + 2 * idx_ori] += 2 * np.pi * para[-1] * (
                                - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                        para[13 + 2 * idx_ori] += 4 * np.pi * para[-1] * (
                                - para[2 * idx_ori + 1] * dx - para[2 * idx_ori] * dy)
                else:
                    raise NotImplementedError
                para = single2tensor(para)
                dict_data.update({'para': para, 'json_path': json_path, 'input_path': input_path})
            else:
                dict_data.update({'json_path': json_path, 'input_path': input_path})

            return dict_data

        else:
            raise NotImplementedError

    def __len__(self):
        if self.phase == 'train' and self.read_whole_dataset and len(self.dataroot[list(self.dataroot.keys())[0]]) > ALLOW_NUM:
            if len(self.dataroot[list(self.dataroot.keys())[0]]) >= 2 * ALLOW_NUM:
                factor = math.ceil(len(self.dataroot[list(self.dataroot.keys())[0]]) / ALLOW_NUM)
                return len(self.dataroot[list(self.dataroot.keys())[0]]) // factor
            else:
                return ALLOW_NUM
        else:
            return len(self.dataroot[list(self.dataroot.keys())[0]])
