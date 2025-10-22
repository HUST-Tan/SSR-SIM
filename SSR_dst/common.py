from utils.mrc import ReadMRC
import numpy as np
import torch


def ReadMrcImage_SSR(path):
    rm = ReadMRC(path)
    if rm.opt['num_step'] == 2:
        data = rm.get_total_data_as_mat(convert_to_tensor=False)
        assert len(data.shape) == 8
        return np.array(data[:, 0, ...]).astype(np.float32), np.array(data[:, 1, ...]).astype(np.float32)
    elif rm.opt['num_step'] == 1:
        data = rm.get_total_data_as_mat(convert_to_tensor=False)
        assert len(data.shape) == 7
        return np.array(data[0:data.shape[0] // 2 * 2:2]).astype(np.float32), np.array(
            data[1:data.shape[0] // 2 * 2:2]).astype(np.float32)
    else:
        raise NotImplementedError

def ReadMrcImage(path):
    rm = ReadMRC(path)
    data = rm.get_total_data_as_mat(convert_to_tensor=False).astype(np.float32)
    return np.array(data)


def ParaProcess(npmat1, npmat2, sampling_rate):
    # npmat1 is k0 and of shape [n_ori, 2]. (0 and +1 orders for L-SIM and NL-SIM, 0 and +2 orders for Single-Slice-SIM).
    # npmat2 is modamp (complex).
    # sampling_rate is just sampling_rate.
    # size = n_ori*2 + n_ori + n_ori + 1
    return np.concatenate(
        (npmat1.flatten(), np.abs(npmat2).flatten(), np.angle(npmat2).flatten(), np.array([sampling_rate])), axis=0)


def json2para(json_dict, outinfo=None):
    # json_dict: json保存的字典
    # outinfo：输出信息
    k0 = np.array(json_dict['k0'])
    phaseAng = np.array(json_dict['phase'])
    if json_dict['if_force_modamp']:
        if len(json_dict['force_modamp']) == 1:  # 3phase
            modamp = np.array(json_dict['force_modamp'][0])
            pha = modamp * np.cos(phaseAng) + 1j * modamp * np.sin(phaseAng)
        elif len(json_dict['force_modamp']) == 2:  # 5phase
            modamp = np.array([
                [json_dict['force_modamp'][0], json_dict['force_modamp'][1]],
                [json_dict['force_modamp'][0], json_dict['force_modamp'][1]],
                [json_dict['force_modamp'][0], json_dict['force_modamp'][1]],
            ])
            pha = modamp * np.cos(phaseAng) + 1j * modamp * np.sin(phaseAng)
    else:
        modamp = np.array(json_dict['modamp'])
        pha = modamp * np.cos(phaseAng) + 1j * modamp * np.sin(phaseAng)

    para = ParaProcess(k0, pha, json_dict['height_space_sampling'])
    if outinfo is not None:
        outinfo.info('pattern para [{}]\n'.format(", ".join(map(lambda x: str(round(x, 4)), para.tolist()))))
    return para


def raw_weighted_np(x):
    """
    :param x: numpy array [T, O, P, C, D, H ,W]
    :return: scale in dim [O]
    """
    if len(x.shape) == 7:
        (T, O, P, C, D, H, W) = x.shape
        for idx_T in range(T):
            for idx_C in range(C):
                for idx_D in range(D):
                    OPHW = torch.from_numpy(x[idx_T, :, :, idx_C, idx_D, :, :].copy())
                    OPHW = OPHW * torch.mean(OPHW[0, ...]) / torch.mean(OPHW, dim=[1, 2, 3], keepdim=True)
                    x[idx_T, :, :, idx_C, idx_D, :, :] = OPHW.numpy()
    elif len(x.shape) == 8:
        (T, S, O, P, C, D, H, W) = x.shape
        for idx_T in range(T):
            for idx_S in range(S):
                for idx_C in range(C):
                    for idx_D in range(D):
                        OPHW = torch.from_numpy(x[idx_T, idx_S, :, :, idx_C, idx_D, :, :].copy())
                        OPHW = OPHW * torch.mean(OPHW[0, ...]) / torch.mean(OPHW, dim=[1, 2, 3], keepdim=True)
                        x[idx_T, idx_S, :, :, idx_C, idx_D, :, :] = OPHW.numpy()
    else:
        raise NotImplementedError
    return x


def raw_weighted_torch(x):
    """
    :param x: torch Tensor [T, O, P, C, D, H ,W]
    :return: scale in dim [O]
    """
    if len(x.shape) == 7:
        (T, O, P, C, D, H, W) = x.shape
        for idx_T in range(T):
            for idx_C in range(C):
                for idx_D in range(D):
                    OPHW = x[idx_T, :, :, idx_C, idx_D, :, :].clone()
                    OPHW = OPHW * torch.mean(OPHW[0, ...]) / torch.mean(OPHW, dim=[1, 2, 3], keepdim=True)
                    x[idx_T, :, :, idx_C, idx_D, :, :] = OPHW
    elif len(x.shape) == 8:
        (T, S, O, P, C, D, H, W) = x.shape
        for idx_T in range(T):
            for idx_S in range(S):
                for idx_C in range(C):
                    for idx_D in range(D):
                        OPHW = x[idx_T, idx_S, :, :, idx_C, idx_D, :, :].clone()
                        OPHW = OPHW * torch.mean(OPHW[0, ...]) / torch.mean(OPHW, dim=[1, 2, 3], keepdim=True)
                        x[idx_T, idx_S, :, :, idx_C, idx_D, :, :] = OPHW
    else:
        raise NotImplementedError

    return x