import numpy as np
import math
import torch
from cv2 import GaussianBlur, BORDER_REPLICATE
from utils.pytorch_msssim import ms_ssim, ssim

device = torch.device('cuda')


# --------------------------------------------
# Calculate the average number of photons from raw data
# --------------------------------------------
def F_cal_average_photon(x, Gaussian_sigma=5.0, ratio=0.2, conversion_factor=0.6026, mean_axis=None, norm_type='mean1000p'):  # default Gaussian_sigma=5.0 which is too large
    # (0) get the average image (wide-field)
    assert x.ndim == 7
    assert norm_type in ['mean1000p', 'maxmin', 'maxmin1000p', 'max', 'none']
    T, O, P, C, D, H, W = x.shape
    if D == 1:
        x = np.mean(x, axis=(0, 1, 2, 3, 4))  # [TOPCD]HW
    else:
        x = np.mean(x, axis=(0, 1, 2, 3))  # [TOPC]DHW
        x = np.max(x, 0)  # Max Intensity Projection
        # 3D-SIM数据最好还是先按照层之间的bleach比率来纠正数据，再计算MIP，不然最大层会偏向远离玻片的那边
    assert x.ndim == 2
    # (1) subtracting the average background of the camera [done]
    # x -= 100 # 在送入该函数之前就做了
    # (2) using a Gaussian LPF (sigma=5) to blur the original image, ksize is set to 9 as following qiao's code
    x_blur = GaussianBlur(src=x, ksize=(9, 9), sigmaX=Gaussian_sigma, sigmaY=Gaussian_sigma, borderType=BORDER_REPLICATE)
    # Gaussian_sigma：5.0为默认值，(由于模板是9*9, Gaussian_sigma实际上达不到5.0，最多能到3.0~4.0，也足够去噪了)
    # (3-1) performing the percentile-normalization on the filtered image
    temp_ = x_blur.flatten()
    if norm_type == 'maxmin':
        # 最大最小，略微受椒盐噪声影响
        max_ = temp_.max()
        min_ = temp_.min()
    elif norm_type == 'maxmin1000p':
        # 这个归一化是DFCAN文章里的归一化，略微受成像视野影响：min_在大视野下通常接近零，在小视野下往往具有一个不小的值
        max_ = np.min(temp_[np.argpartition(temp_, -temp_.shape[0] // 1000)[-temp_.shape[0] // 1000:]])
        min_ = np.max(temp_[np.argpartition(temp_, temp_.shape[0] // 1000)[:temp_.shape[0] // 1000]])
    elif norm_type == 'mean1000p':
        # 这个归一化更为平和，在bioSR数据集上，前三种norm方式算出的光子数不超过1%，即无差异
        max_ = np.mean(temp_[np.argpartition(temp_, -temp_.shape[0] // 1000)[-temp_.shape[0] // 1000:]])
        min_ = np.mean(temp_[np.argpartition(temp_, temp_.shape[0] // 1000)[:temp_.shape[0] // 1000]])
    elif norm_type == 'none':
        # 啥都不做
        max_ = 1.0
        min_ = 0.0
    elif norm_type == 'max':
        # 相当于啥都不做，后两种norm方式算出的光子数相同，并与前三种norm方式算出的光子数相差不超过3%
        max_ = temp_.max()
        min_ = 0.0
    else:
        raise NotImplementedError
    x_blur = (x_blur - min_) / (max_ - min_)
    # (3-2) extracting the feature-only regions of the normalized image with threshold default 0.2
    x = x[x_blur > np.max(x_blur) * ratio]
    # ratio：阈值，0.2为默认值，适合F-Actin或CCPs这种TIRF图像；实际上，GI-SIM和3D-SIM取更大一点（例如~0.4）更合适；该工作均取0.2
    # (4) calculating the average sCMOS count of the thresholded image
    average_sCMOS_count = np.mean(x)
    # (5) converting the sCMOS count into the photon count by a conversion factor of 0.6026 photons per sCMOS count, which is measured via Hamamatsu's protocol
    average_photon_count = average_sCMOS_count * conversion_factor
    return average_photon_count


def F_cal_average_sCMOS_count(x, Gaussian_sigma=5.0, ratio=0.2, mean_axis=None):
    return F_cal_average_photon(x, Gaussian_sigma=Gaussian_sigma, ratio=ratio, conversion_factor=1.0, mean_axis=mean_axis)



# --------------------------------------------
# evaluation index MS-SSIM
# --------------------------------------------
def calculate_index(img1, img2, border=0, need_res=False, peak=None):
    """
    batchsize of img1 (and img2) should be 1
    NumChannel of img1 (and img2) should be 1
        - this script is for calcuting indexes of HW-shape data, DHW-shape data, THW-shape data, and TDHW-shape data,
          but not Bxxxx-shape data or Cxxxx-shape data or BCxxxx-shape data
        - this script is not for calcuting indexes of multi-channel data, and also, this script is not for calcuting indexes of rDL raw data
    """
    # dict_out = {'nrmse': calculate_nrmse(img1, img2, border=border),
    #             'psnr': calculate_psnr_peak(img1, img2, border=border),
    #             'ssim': calculate_ssim_peak(img1, img2, border=border),
    #             'msssim': calculate_ms_ssim_peak(img1, img2, border=border)}
    # return dict_out
    list_out = [calculate_nrmse(img1, img2, border=border, peak=peak),
                calculate_psnr_peak(img1, img2, border=border, peak=peak),
                calculate_ssim_peak(img1, img2, border=border, peak=peak),
                calculate_ms_ssim_peak(img1, img2, border=border, peak=peak),
                ]
    assert need_res is False, 'need res'
    return list_out


def calculate_ms_ssim_peak(img1, img2, border=32, peak=None):
    # for N-dim data (N>2), ssim is calculated slice-by-slice
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
    img1 = img1.to(device)
    img2 = img2.to(device)

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    img1_stack = img1.reshape(-1, h, w)
    img2_stack = img2.reshape(-1, h, w)
    msssim_sum = 0
    msssim_num = 0
    if peak is None: peak = torch.max(img1_stack)
    for idx in range(img1_stack.shape[0]):
        img1temp = img1_stack[idx, border:h - border, border:w - border]
        img2temp = img2_stack[idx, border:h - border, border:w - border]
        # if peak is None:
        #     ms_ssim_val = ms_ssim(img1temp.unsqueeze(0).unsqueeze(0), img2temp.unsqueeze(0).unsqueeze(0), data_range=torch.max(img1temp), size_average=True).cpu().numpy().astype(np.float32)
        # else:
        #     ms_ssim_val = ms_ssim(img1temp.unsqueeze(0).unsqueeze(0), img2temp.unsqueeze(0).unsqueeze(0), data_range=peak, size_average=True).cpu().numpy().astype(np.float32)
        ms_ssim_val = ms_ssim(img1temp.unsqueeze(0).unsqueeze(0), img2temp.unsqueeze(0).unsqueeze(0), data_range=peak, size_average=True).cpu().numpy().astype(np.float32)
        msssim_sum += ms_ssim_val
        msssim_num += 1

    return msssim_sum / msssim_num


def calculate_ssim_peak(img1, img2, border=32, peak=None):
    # for N-dim data (N>2), ssim is calculated slice-by-slice
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
    img1 = img1.to(device)
    img2 = img2.to(device)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    img1_stack = img1.reshape(-1, h, w)
    img2_stack = img2.reshape(-1, h, w)
    ssim_sum = 0
    ssim_num = 0
    if peak is None: peak = torch.max(img1_stack)
    for idx in range(img1_stack.shape[0]):
        img1temp = img1_stack[idx, border:h - border, border:w - border]
        img2temp = img2_stack[idx, border:h - border, border:w - border]
        # if peak is None:
        #     ssim_val = ssim(img1temp.unsqueeze(0).unsqueeze(0), img2temp.unsqueeze(0).unsqueeze(0), data_range=torch.max(img1temp), size_average=True).cpu().numpy().astype(np.float32)
        # else:
        #     ssim_val = ssim(img1temp.unsqueeze(0).unsqueeze(0), img2temp.unsqueeze(0).unsqueeze(0), data_range=peak, size_average=True).cpu().numpy().astype(np.float32)
        ssim_val = ssim(img1temp.unsqueeze(0).unsqueeze(0), img2temp.unsqueeze(0).unsqueeze(0), data_range=peak, size_average=True).cpu().numpy().astype(np.float32)
        ssim_sum += ssim_val
        ssim_num += 1

    return ssim_sum / ssim_num


def calculate_nrmse(img1, img2, border=32, peak=None, down=None):
    # for any dimension data
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    if isinstance(img1, torch.Tensor):
        img1 = img1.to(device).to(torch.float64)
        img2 = img2.to(device).to(torch.float64)
    else:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
    img1 = img1[..., border:h - border, border:w - border]
    img2 = img2[..., border:h - border, border:w - border]
    mse = ((img1 - img2) ** 2).mean()
    if peak is None: peak = img1.max()
    if peak < img1.min(): return float('inf')
    if down is None: down = img1.min()
    result = math.sqrt(mse) / (peak - down)
    if isinstance(result, torch.Tensor):
        return result.cpu().numpy()
    else:
        return result


def calculate_psnr_peak(img1, img2, border=32, peak=None):
    # for any dimension data
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[-2:]
    if isinstance(img1, torch.Tensor):
        img1 = img1.to(device).to(torch.float64)
        img2 = img2.to(device).to(torch.float64)
    else:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
    img1 = img1[..., border:h - border, border:w - border]
    img2 = img2[..., border:h - border, border:w - border]
    mse = ((img1 - img2) ** 2).mean()
    if peak is None: peak = img1.max()
    if mse == 0: return float('inf')
    result = 20 * math.log10(peak / math.sqrt(mse))
    if isinstance(result, torch.Tensor):
        return result.cpu().numpy()
    else:
        return result


def first_p999(x):
    # to norm data
    if isinstance(x, torch.Tensor):
        temp_ = x.flatten()
        a = torch.topk(temp_, temp_.shape[0] // 1000, sorted=False)[0].min()
    else:
        temp_ = x.flatten().astype(np.float32)
        a = np.min(temp_[np.argpartition(temp_, -temp_.shape[0] // 1000)[-temp_.shape[0] // 1000:]])
    return a


def norm(x):
    # to norm data
    return x / first_p999(x)


def xxnorm(x):
    # pytorch code pf xxnorm in qiao's paper <-> not used
    if isinstance(x, torch.Tensor):
        temp_ = x.flatten()
        first1000percentile = torch.topk(temp_, temp_.shape[0] // 1000, sorted=False)[0].min()
        last1000percentile = torch.topk(temp_, temp_.shape[0] // 1000, sorted=False, largest=False)[0].max()
    else:
        temp_ = x.flatten().astype(np.float32)
        first1000percentile = np.min(temp_[np.argpartition(temp_, -temp_.shape[0] // 1000)[-temp_.shape[0] // 1000:]])
        last1000percentile = np.max(temp_[np.argpartition(temp_, temp_.shape[0] // 1000)[:temp_.shape[0] // 1000]])
    x = (x - last1000percentile) / (first1000percentile - last1000percentile)
    return x


# ------------------------------------
# apodize2d
# ------------------------------------
def apodize2d(x, num_apodize=10):
    """
    :param x: input raw images [..., H, W]
    :return: apodized raw images: avoid artifacts due to the native periodic padding bahavior in fft operation
    """

    H_, W_ = x.shape[-2:]

    slice_apodize = list(range(num_apodize - 1, -1, -1))

    imageUp = x[..., :num_apodize, :]
    imageDown = x[..., -num_apodize:, :]

    diff = (imageDown[..., slice_apodize, :] - imageUp) / 2

    fact = 1 - torch.sin((torch.arange(0, num_apodize, device=x.device) + 0.5) / num_apodize * math.pi / 2)

    factor = diff * torch.tile(fact.unsqueeze(1), (1, W_))

    x[..., :num_apodize, :] = imageUp + factor
    x[..., -num_apodize:, :] = imageDown - factor[..., slice_apodize, :]

    imageLeft = x[..., :num_apodize]
    imageRight = x[..., -num_apodize:]

    diff = (imageRight[..., slice_apodize] - imageLeft) / 2

    factor = diff * torch.tile(fact.unsqueeze(0), (H_, 1))

    x[..., :num_apodize] = imageLeft + factor
    x[..., -num_apodize:] = imageRight - factor[..., slice_apodize]

    return x
