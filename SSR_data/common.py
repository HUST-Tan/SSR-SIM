import os
import numpy as np
import torch
from math import pi
from cv2 import getRotationMatrix2D, warpAffine
from utils.option import dict2nonedict
import torchvision.transforms.functional as TVTF

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff','.TIF','.TIFF']
NPY_EXTENSIONS = ['.npy', '.NPY']
JSON_EXTENSIONS = ['.json', '.JSON']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)
def is_json_file(filename):
    return any(filename.endswith(extension) for extension in JSON_EXTENSIONS)

def _get_paths_from_npys(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_npy_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image(para) file'.format(path)
    return images

def _get_paths_from_jsons(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_json_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image(para) file'.format(path)
    return images

def get_npys_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_npys(dataroot))
    return paths

def get_json_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_jsons(dataroot))
    return paths

def read_npy_and_check(path, assert_data_shape):
    assert path[-3:] == 'npy', 'only .npy data are allowed while training'
    # pre-process npy files
    img_np = np.load(path)  # TOPCDHW  |  C is always 1  |  for 2d data D is 1, for 3d data D > 1
    (T, O, P, C, D, H, W) = img_np.shape
    assert (O, P, C) == assert_data_shape, 'assert train/test data shape {}, it should be {}'.format(img_np.shape[:4], assert_data_shape)
    if D == 1 and T == 1: # CHW
        return img_np.reshape(T * O * P * C * D, H, W)
    elif D != 1 and T == 1: # CDHW
        return img_np.reshape(T * O * P * C, D, H, W)
    elif D == 1 and T != 1: # TOPCDHW -> OP,T,H,W
        return np.transpose(img_np.reshape(T, O * P * C * D, H, W), (1, 0, 2, 3))
    else:
        raise NotImplementedError

def single2tensor(img):
    if isinstance(img, torch.Tensor):
        return img
    else:
        return torch.from_numpy(np.ascontiguousarray(img)).float()

def dictsingle2tensor(img_dict):
    for key in img_dict.keys():
        img_dict[key] = single2tensor(img_dict[key])
    return img_dict

def rot_datadict_para(data_dict, para, angle):
    # for idx in range(len(data_list)):
    #     data_list[idx] = rot_data(data_list[idx], angle)
    for key in data_dict.keys():
        data_dict[key] = rot_data(data_dict[key], angle)
    para = rot_para(para, angle)
    return data_dict, para

def rot_datadict(data_dict, angle):
    # for idx in range(len(data_list)):
    #     data_list[idx] = rot_data(data_list[idx], angle)
    for key in data_dict.keys():
        data_dict[key] = rot_data(data_dict[key], angle)
    return data_dict

def rot_para(para, angle):
    n_ori = 3
    for idx_ori in range(n_ori):
        para[2 * idx_ori + 1], para[2 * idx_ori] = rot_coor(para[2 * idx_ori + 1], para[2 * idx_ori], angle)
    return para

def rot_coor(x1, x0, ang):
    x_angle = np.arctan2(x1, x0)
    x_length = np.sqrt(x0 ** 2 + x1 ** 2)
    theta = x_angle - ang * 2.0 * pi / 360.0
    x1 = x_length * np.sin(theta)
    x0 = x_length * np.cos(theta)
    return x1, x0

def rot_data(x, angle):
    assert x.shape[-1] % 2 == 0 and x.shape[-2] % 2 == 0
    if isinstance(x, torch.Tensor):
        if len(x.shape) >= 5:
            x_shape = x.shape
            return TVTF.rotate(x.squeeze(), angle, center=(x.shape[-2] / 2.0 + 0.5, x.shape[-1] / 2.0 + 0.5), interpolation=TVTF.InterpolationMode.BILINEAR).reshape(x_shape)
        else:
            return TVTF.rotate(x, angle, center=(x.shape[-2] / 2.0 + 0.5, x.shape[-1] / 2.0 + 0.5), interpolation=TVTF.InterpolationMode.BILINEAR)
    else:
        if len(x.shape) == 3:
            c, rows, cols = x.shape
            M_raw = getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)
            dst_raw = x.copy()
            for idx in range(c):
                dst_raw[idx, ...] = warpAffine(x[idx, ...], M_raw, (cols, rows))
        elif len(x.shape) == 4:
            c, d, rows, cols = x.shape
            M_raw = getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)
            dst_raw = x.copy()
            for idx_c in range(c):
                for idx_d in range(d):
                    dst_raw[idx_c, idx_d, ...] = warpAffine(x[idx_c, idx_d, ...], M_raw, (cols, rows))
        elif len(x.shape) == 5:
            b, c, d, rows, cols = x.shape
            M_raw = getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)
            dst_raw = x.copy()
            for idx_b in range(b):
                for idx_c in range(c):
                    for idx_d in range(d):
                        dst_raw[idx_b, idx_c, idx_d, ...] = warpAffine(x[idx_b, idx_c, idx_d, ...], M_raw, (cols, rows))
        else:
            raise NotImplementedError
        return dst_raw

def get_dataset(dict_path, dict_key, phase='all'):
    dict_new = dict2nonedict({})
    for key, value in dict_key.items():
        if value:
            if phase == 'all':
                dict_new[key] = get_npys_paths(dict_path[key])
            elif phase == 'train':
                if key[:5] == 'train':
                    dict_new[key] = get_npys_paths(dict_path[key])
            elif phase == 'val':
                if key[:3] == 'val':
                    dict_new[key] = get_npys_paths(dict_path[key])
            else:
                raise NotImplementedError
    return dict_new

def get_datadict_from_datarootdict(datarootdict, idx, ncr=None, ncs=None, ncw=None, convert_to_CUDA=False):
    datadict = dict2nonedict({})

    for key, value in datarootdict.items():
        if key.lower().find('wf') >= 0:
            if convert_to_CUDA:
                datadict[key] = torch.from_numpy(np.ascontiguousarray(read_npy_and_check(value[idx], ncw))).float().to(torch.device('cuda'))
            else:
                datadict[key] = read_npy_and_check(value[idx], ncw)
        elif key.lower().find('raw') >= 0:
            if convert_to_CUDA:
                datadict[key] = torch.from_numpy(np.ascontiguousarray(read_npy_and_check(value[idx], ncr))).float().to(torch.device('cuda'))
            else:
                datadict[key] = read_npy_and_check(value[idx], ncr)
        elif key.lower().find('sim') >= 0:
            if convert_to_CUDA:
                datadict[key] = torch.from_numpy(np.ascontiguousarray(read_npy_and_check(value[idx], ncs))).float().to(torch.device('cuda'))
            else:
                datadict[key] = read_npy_and_check(value[idx], ncs)
        elif key.lower().find('pattern') >= 0 or key.lower().find('psf') >= 0:
            if convert_to_CUDA:
                datadict[key] = torch.from_numpy(np.ascontiguousarray(np.load(value[idx]))).float().to(torch.device('cuda'))
            else:
                datadict[key] = np.load(value[idx])
        else:
            raise NotImplementedError

    return datadict