import time
from functools import wraps
from contextlib import contextmanager
import datetime
from functools import reduce
import numpy as np
import random
import torch
import os
import platform

def my_meshgrid(x, y):
    if int(torch.__version__.split('.')[0]) >= 2 or (int(torch.__version__.split('.')[0]) >= 1 and int(torch.__version__.split('.')[1]) >= 10):
        return torch.meshgrid(x, y, indexing='ij')
    else:
        return torch.meshgrid(x, y)

# --------------------------------------------
# time this
# --------------------------------------------
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper

# --------------------------------------------
# timeblock
# --------------------------------------------
@contextmanager
def timeblock(label):
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        end = time.perf_counter()
        print('{} : {}'.format(label, end - start))

@contextmanager
def codeblock(label=None):
    yield


# --------------------------------------------
# imshow
# --------------------------------------------
# def sp_imshow(x, cmap='gray'):
#     if isinstance(x, torch.Tensor): x = x.cpu().numpy()
#     x = x.squeeze()
#     if x.dtype in [np.complex64, np.complex128]: x = np.real(x)
#     plt.imshow(x, cmap=cmap)
#     plt.show()
#
# def fq_imshow(x, cmap='gray'):
#     if isinstance(x, torch.Tensor): x = x.cpu().numpy()
#     x = x.squeeze()
#     x = np.log(np.abs(x))
#     plt.imshow(x, cmap=cmap)
#     plt.show()
#
# def sp_fft2d_imshow(x):
#     if not isinstance(x, torch.Tensor): raise NotImplementedError("input in sp_fft2d_imshow must be tensor")
#     x = torch.fft.ifftshift(x, [-1,-2])
#     x = torch.fft.fft2(x)
#     x = torch.fft.fftshift(x, [-1,-2])
#     fq_imshow(x)
#
# def fq_ifft2d_imshow(x):
#     if not isinstance(x, torch.Tensor): raise NotImplementedError("input in sp_fft2d_imshow must be tensor")
#     x = torch.fft.ifftshift(x, [-1,-2])
#     x = torch.fft.fft2(x)
#     x = torch.fft.fftshift(x, [-1,-2])
#     sp_imshow(x)


# --------------------------------------------
# output in txt
# --------------------------------------------
class WriteOutputTxt:
    def __init__(self, filename, encoding=None, if_print=True, if_use=True):
        self.if_use = if_use
        if if_use:
            # check
            assert filename[-4:] == '.txt'
            self.filename = filename
            self.encoding = encoding
            self.if_print = if_print
            with open(filename, mode="w", encoding=encoding) as f:
                f.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')

    def info(self, content):
        try:
            if self.if_use:
                if self.if_print: print(content)
                with open(self.filename, mode="a", encoding=self.encoding) as f:
                    f.write(content)
        except FileNotFoundError:
            pass

# --------------------------------------------
# dir tool
# --------------------------------------------
def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_with_time(path):
    path += '_' + get_timestamp()
    os.makedirs(path)
    return path


# --------------------------------------------
# seed
# --------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------
# get_model
# --------------------------------------------
def get_model(path, model, device=torch.device('cuda')):
    model.load_state_dict(torch.load(path), strict=True)
    model = model.to(device)
    model.eval()

    for _, v in model.named_parameters():
        v.requires_grad = False

    return model