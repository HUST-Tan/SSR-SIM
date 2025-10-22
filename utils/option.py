import struct
from collections import OrderedDict
import json
import os
import platform
from utils.tools import get_timestamp, mkdir
from functools import reduce

# ----------------------------------------
# json -> dict
# ----------------------------------------
def sir_parse2dict(opt_path, ordered_dict_in=None):
    """
    :param opt_path: [str] path of option saved in json file
    :return: dict
    """
    # remove comments starting with '//'
    json_str = ''
    if os.path.exists(opt_path):
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
    else:
        temppath = os.path.join(*opt_path.split('\\')[1:])
        if os.path.exists(temppath):
            with open(temppath, 'r') as f:
                for line in f:
                    line = line.split('//')[0] + '\n'
                    json_str += line
        else:
            print(opt_path)
            raise FileNotFoundError

    if platform.system().lower() in ['windows']:
        pass
    else:
        # only for research
        json_str = json_str.replace('\\\\', '/')
        for disksignal in ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
            json_str = json_str.replace(disksignal+':', '/home/ljh/')
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt = dict2nonedict(opt)
    if ordered_dict_in is None:
        return opt
    else:
        ordered_dict_out = ordered_dict_in.copy()
        ordered_dict_out.update(opt)
        return ordered_dict_out

# if __name__ == '__main__':
#     sir_parse2dict(r'K:\AIA_SIM_TestData\data_Lifeact_COS7_mEmerald_HSNR_bioSR_all\20211120_101810\TIRF-488-10ms_cam1_step1_001.json')

# --------------------------------------------
# tuple/list product
# --------------------------------------------
def product_of_tuple_elements(tup):
    return reduce(lambda x,y:x*y, tup)

# ----------------------------------------
# json opt -> [dict] opt
# ----------------------------------------
def AI_parse(opt_path, is_train=True):

    opt = AI_parse_read(opt_path)
    opt = AI_parse_process(opt, is_train)

    return opt

def AI_parse_read(opt_path):
    # remove comments starting with '//'
    json_str = ''
    try:
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
    except FileNotFoundError:
        try:
            with open(os.path.join(*opt_path.split('\\')[1:]), 'r') as f:
                for line in f:
                    line = line.split('//')[0] + '\n'
                    json_str += line
        except TypeError:
            with open(os.path.join(*opt_path.split('/')[1:]), 'r') as f:
                for line in f:
                    line = line.split('//')[0] + '\n'
                    json_str += line

    # initialize opt
    if platform.system().lower() in ['windows']:
        pass
    else:
        # only for research
        json_str = json_str.replace('\\\\', '/')
        for disksignal in ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
            json_str = json_str.replace(disksignal+':', '/home/ljh/')
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt = dict2nonedict(opt)
    opt['opt_path'] = opt_path
    return opt

def AI_parse_process(opt, is_train=True):

    # ----------------------------------------
    # CHECK
    # ----------------------------------------
    # assert supervised
    assert opt['supervise'] in [
        'full-supervised', 'full-supervised-val', # using high-snr data as target
        'self-supervised', 'self-supervised-val', # using low-snr data itself (using neighbouring tps) as target
    ]
    if opt['supervise'] == 'full-supervised-val': opt['supervise'] = 'full-supervised'

    # assert imaging model [wf means non-sim]
    assert opt['data'] in ['2d-sim','3d-sim']

    assert opt['model'] in ["reconstruction"]

    # ----------------------------------------
    #
    # ----------------------------------------

    if platform.system().lower() in ['windows']:
        pass
    else:
        temp = opt['dataset_root']
        temp = temp.replace('\\\\', '/')
        temp = temp.replace('\\', '/')
        for disksignal in ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
            temp = temp.replace(disksignal+':', '/home/ljh/')
        opt['dataset_root'] = temp

    opt['is_train'] = is_train

    # path -> expanduser
    opt['dataset_root'] = os.path.expanduser(opt['dataset_root'])
    if opt['pretrained_netG']: opt['pretrained_netG'] = os.path.expanduser(opt['pretrained_netG'])

    # path for model / valimages / output_txt
    opt['save_path'] = os.path.join(opt['dataset_root'], '{}_{}'.format(opt['supervise'], opt['model']))
    if opt['onetps-algorithm'] is not None: opt['save_path'] += '_{}'.format(opt['onetps-algorithm'])
    opt['save_path'] += '_{}'.format(opt['net_G'])
    if opt['net_tail'] is not None: opt['save_path'] += '{}'.format(opt['net_tail'])
    if os.path.exists(opt['save_path']):
        opt['save_path'] += '_{}'.format(get_timestamp())

    mkdir(opt['save_path'])
    mkdir(os.path.join(opt['save_path'], 'model'))
    mkdir(os.path.join(opt['save_path'], 'image'))

    # ----------------------------------------
    # IF need gt in validation
    # IF need para
    # ----------------------------------------
    opt['val_need_gt'] = True if opt['supervise'] in ['full-supervised', 'self-supervised-val', 'onetps-self-supervised-val'] else False
    opt['need_para'] = True

    # ----------------------------------------
    # dataset path
    # ----------------------------------------
    dataset_path = {
        "train_RAW_LSNR_1": os.path.join(opt['dataset_root'], 'train_Raw_LSNR_1'),
        "train_RAW_LSNR_2": os.path.join(opt['dataset_root'], 'train_Raw_LSNR_2'),
        "train_RAW_HSNR": os.path.join(opt['dataset_root'], 'train_Raw_HSNR'),
        "train_SIM_LSNR_1": os.path.join(opt['dataset_root'], 'train_SIM_LSNR_1'),
        "train_SIM_LSNR_2": os.path.join(opt['dataset_root'], 'train_SIM_LSNR_2'),
        "train_SIM_HSNR": os.path.join(opt['dataset_root'], 'train_SIM_HSNR'),
        "train_WF_LSNR_1": os.path.join(opt['dataset_root'], 'train_WF_LSNR_1'),
        "train_WF_LSNR_2": os.path.join(opt['dataset_root'], 'train_WF_LSNR_2'),
        "train_WF_HSNR": os.path.join(opt['dataset_root'], 'train_WF_HSNR'),
        "train_PreDeModPattern": os.path.join(opt['dataset_root'], 'train_PreDeModPattern'),
        "train_PreDeModPatternPSF": os.path.join(opt['dataset_root'], 'train_PreDeModPatternPSF'),

        "val_RAW_LSNR_1": os.path.join(opt['dataset_root'], 'val_Raw_LSNR_1'),
        "val_RAW_LSNR_2": os.path.join(opt['dataset_root'], 'val_Raw_LSNR_2'),
        "val_RAW_HSNR": os.path.join(opt['dataset_root'], 'val_Raw_HSNR'),
        "val_SIM_LSNR_1": os.path.join(opt['dataset_root'], 'val_SIM_LSNR_1'),
        "val_SIM_LSNR_2": os.path.join(opt['dataset_root'], 'val_SIM_LSNR_2'),
        "val_SIM_HSNR": os.path.join(opt['dataset_root'], 'val_SIM_HSNR'),
        "val_WF_LSNR_1": os.path.join(opt['dataset_root'], 'val_WF_LSNR_1'),
        "val_WF_LSNR_2": os.path.join(opt['dataset_root'], 'val_WF_LSNR_2'),
        "val_WF_HSNR": os.path.join(opt['dataset_root'], 'val_WF_HSNR'),
        "val_PreDeModPattern": os.path.join(opt['dataset_root'], 'val_PreDeModPattern'),
        "val_PreDeModPatternPSF": os.path.join(opt['dataset_root'], 'val_PreDeModPatternPSF'),
    }

    para_path = {
        "train_PARA": os.path.join(opt['dataset_root'], 'train_Para'),
        "val_PARA": os.path.join(opt['dataset_root'], 'val_Para'),
    }

    json_path = {
        "train_JSON": os.path.join(opt['dataset_root'], 'train_Json'),
        "val_JSON": os.path.join(opt['dataset_root'], 'val_Json'),
    }

    # ----------------------------------------
    # set dataset used
    # ----------------------------------------
    dataset_use = {
        "train_RAW_LSNR_1": False,
        "train_RAW_LSNR_1_pattern": False,
        "train_RAW_LSNR_2": False,
        "train_RAW_HSNR": False,
        "train_SIM_LSNR_1": False,
        "train_SIM_LSNR_2": False,
        "train_SIM_HSNR": False,
        "train_WF_LSNR_1": False,
        "train_WF_LSNR_2": False,
        "train_WF_HSNR": False,
        "train_PreDeModPattern": False,
        "train_PreDeModPatternPSF": False,

        "val_RAW_LSNR_1": False,
        "val_RAW_LSNR_2": False,
        "val_RAW_HSNR": False,
        "val_SIM_LSNR_1": False,
        "val_SIM_LSNR_2": False,
        "val_SIM_HSNR": False,
        "val_WF_LSNR_1": False,
        "val_WF_LSNR_2": False,
        "val_WF_HSNR": False,
        "val_PreDeModPattern": False,
        "val_PreDeModPatternPSF": False,
    }


    # ----------------------------------------
    # task
    # ----------------------------------------
    opt['raw_scale'] = 1
    if opt['data'] in ['2d-sim']:
        opt['sim_scale'] = 2
        opt['raw_shape_OPC'] = (3, 3, 1)  # OPC
        opt['sim_shape_OPC'] = (1, 1, 1)  # OPC
    elif opt['data'] == '3d-sim':
        opt['sim_scale'] = 2
        opt['raw_shape_OPC'] = (3, 5, 1)  # OPC
        opt['sim_shape_OPC'] = (1, 1, 1)  # OPC
    else:
        raise NotImplementedError

    # ----------------------------------------
    # model and sup
    # ----------------------------------------
    opt['net_scale'] = opt['sim_scale'] // opt['raw_scale']
    opt['net_channel_in'] = opt['raw_shape_OPC'][0] * opt['raw_shape_OPC'][1]
    opt['net_channel_out'] = opt['sim_shape_OPC'][0] * opt['sim_shape_OPC'][1]
    if opt['supervise'] in ["full-supervised"]:
        dataset_use['train_RAW_LSNR_1'] = True
        dataset_use['train_SIM_HSNR'] = True
        dataset_use['val_RAW_LSNR_1'] = True
        dataset_use['val_SIM_HSNR'] = True
    elif opt['supervise'] in ["self-supervised-val"]:
        dataset_use['train_RAW_LSNR_1'] = True
        dataset_use['train_SIM_LSNR_2'] = True
        dataset_use['val_RAW_LSNR_1'] = True
        dataset_use['val_SIM_HSNR'] = True
    elif opt['supervise'] in ["self-supervised"]:
        dataset_use['train_RAW_LSNR_1'] = True
        dataset_use['train_SIM_LSNR_2'] = True
        dataset_use['val_RAW_LSNR_1'] = True
    elif opt['supervise'] in ["onetps-self-supervised-val"]:
        dataset_use['train_RAW_LSNR_1'] = True
        dataset_use['train_SIM_LSNR_1'] = True
        dataset_use['val_RAW_LSNR_1'] = True
        dataset_use['val_SIM_HSNR'] = True
    elif opt['supervise'] in ["onetps-self-supervised"]:
        dataset_use['train_RAW_LSNR_1'] = True
        dataset_use['train_SIM_LSNR_1'] = True
        dataset_use['val_RAW_LSNR_1'] = True
    else:
        raise NotImplementedError

    opt['dataset_path'] = dataset_path
    opt['dataset_use'] = dataset_use
    opt['para_path'] = para_path
    opt['json_path'] = json_path

    # ----------------------------------------
    # close multi-thread (multi-process) if the
    #   dataset havs been loaded in CUDA memory
    # ----------------------------------------
    if opt['read_whole_dataset_toGPU']:
        opt['dataloader_num_workers'] = 0

    # ----------------------------------------
    # save only PSNR-best models in simu
    # save all model in practice
    # ----------------------------------------
    # if opt['supervise'] in ['full-supervised', 'self-supervised-val', 'onetps-self-supervised-val']:
    #     opt['num_iter_interval_save'] = 1000000
    # else:
    #     opt['num_iter_interval_save'] = min(opt['num_iter_interval_save'], opt['num_iter_interval_val']*2)

    return opt

# --------------------------------------------
# process null in json / dict
# --------------------------------------------
def dict2nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict2nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict2nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
class NoneDict(dict):
    def __missing__(self, key):
        return None

# --------------------------------------------
# dict to string for logger by recursion
# --------------------------------------------
def dict2str(opt, indent_l=1):
    msg = ''
    for key, vaule in opt.items():
        if isinstance(vaule, dict):
            msg += ' ' * (indent_l * 2) + key + ':{\n'
            msg += dict2str(vaule, indent_l + 1)
            msg += ' ' * (indent_l * 2) + '}\n'
        else:
            msg += ' ' * (indent_l * 2) + key + ': ' + str(vaule) + '\n'
    return msg

# --------------------------------------------
# print dict structurally
# --------------------------------------------
def print_dict(opt):
    print(dict2str(opt))


# --------------------------------------------
# save json
# --------------------------------------------
def save_json(opt, dump_path):
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)