"""
Download LifeAct (Zero-shot SSR) dataset:
from https://drive.google.com/file/d/1VMe_RfTZg3eaEyo8XUrgNZau775pvE88/view?usp=drive_link,
and unzip them to
path="home/LifeAct" (Linux) or path=r"K:\LifeAct" (Windows)

Pre-training models are provided in https://drive.google.com/file/d/10FHPORONhF7lXUd5VB2ilV39b0hapHWx/view?usp=drive_link
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.split(os.getcwd())[0])
device = torch.device('cuda')

from SIR_core.common import post_process_file

from utils.option import mkdir, save_json, AI_parse_read, AI_parse_process, sir_parse2dict, get_timestamp
from utils.image_process import F_cal_average_photon
from utils.tools import mkdir_with_time, WriteOutputTxt
from utils.mrc import make_sr_header, ReadMRC, WriteMRC, make_sr_header_assign_scale

from SSR_dst.common import ReadMrcImage_SSR, raw_weighted_np, raw_weighted_torch, json2para
from SSR_logic.main_train_plain import main
from SSR_model.network_phct import PHCT
from SSR_model.common import generate_pattern


def sirecon_noisy_raw_data_for_training(path: str):
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if fname == "seq1_TIRF-SIM488_GreenCh-DL.mrc":
                post_process_file(file=os.path.join(dirpath, fname), imaging_device='MultiSIM002', ifprint=True)


def convert_to_training_dataset(dataset_folder: str):
    min_allowed_photon = 10  # [10/25/50] [TIRF/HighNAGI/LowNAGI]

    # ----------------------------------------
    #            <check the number of cells available>
    # ----------------------------------------
    list_raw_path, list_sim_path, list_json_path = [], [], []
    for dirpath, _, fnames in sorted(os.walk(dataset_folder)):
        for fname in sorted(fnames):
            if fname == "seq1_TIRF-SIM488_GreenCh-DL.mrc":
                raw_path = os.path.join(dirpath, fname)
                sim_path = os.path.join(dirpath, fname[:-4] + '_SIM.mrc')
                json_path = os.path.join(dirpath, fname[:-4] + '_output', fname[:-4] + '.json')
                assert os.path.exists(raw_path) and os.path.exists(sim_path) and os.path.exists(json_path)
                list_raw_path.append(raw_path)
                list_sim_path.append(sim_path)
                list_json_path.append(json_path)

    print(len(list_raw_path))

    num_of_cell = len(list_raw_path)

    # ----------------------------------------
    #            <mkdir and config-txt>
    # ----------------------------------------
    training_data_folder = os.path.join(os.path.split(dataset_folder)[0], os.path.split(dataset_folder)[1] + '_DST')
    if not os.path.exists(training_data_folder):
        mkdir(training_data_folder)
    else:
        training_data_folder = mkdir_with_time(os.path.join(os.path.split(dataset_folder)[0], os.path.split(dataset_folder)[1] + '_DST'))

    for s_1 in ['train']:
        for s_2 in ['Raw_LSNR_1', 'SIM_LSNR_1', 'Raw_LSNR_2', 'SIM_LSNR_2', 'Para', 'Json']:
            mkdir(os.path.join(training_data_folder, s_1 + '_' + s_2))
    for s_1 in ['val']:
        for s_2 in ['Raw_LSNR_1', 'SIM_LSNR_1', 'Para', 'Json']:
            mkdir(os.path.join(training_data_folder, s_1 + '_' + s_2))

    outinfo = WriteOutputTxt(os.path.join(training_data_folder, 'config.txt'))
    outinfo.info('list_raw_path: {}\n'.format(list_raw_path))
    outinfo.info('num_of_cell: {:d}\n'.format(num_of_cell))
    outinfo.info('min_allowed_photon: {: .1f}\n'.format(min_allowed_photon))

    # ----------------------------------------
    #            <Process and Save>
    # ----------------------------------------
    outinfo.info('\n-------- Processing --------')
    training_patch_num = 0
    val_patch_num = 0

    for idx in range(num_of_cell):

        outinfo.info('\non processing cell/sequence No.{:d}'.format(idx))

        json_path = list_json_path[idx]
        json_dict = sir_parse2dict(json_path)  # return a class

        outinfo.info('use all the slices')

        raw_lsnr_1, raw_lsnr_2 = ReadMrcImage_SSR(list_raw_path[idx])
        raw_lsnr_1 = raw_lsnr_1.astype(np.float32) - json_dict['camera_background']
        raw_lsnr_2 = raw_lsnr_2.astype(np.float32) - json_dict['camera_background']
        raw_lsnr_1, raw_lsnr_2 = raw_weighted_np(raw_lsnr_1), raw_weighted_np(raw_lsnr_2)
        sim_lsnr_1, sim_lsnr_2 = ReadMrcImage_SSR(list_sim_path[idx])
        sim_lsnr_1 = sim_lsnr_1.astype(np.float32)
        sim_lsnr_2 = sim_lsnr_2.astype(np.float32)

        para = json2para(json_dict, outinfo)

        average_photon_1 = F_cal_average_photon(raw_lsnr_1, mean_axis=(0, 1, 2, 3, 4))
        average_photon_2 = F_cal_average_photon(raw_lsnr_2, mean_axis=(0, 1, 2, 3, 4))

        if average_photon_2 < min_allowed_photon or average_photon_1 < min_allowed_photon:
            continue

        mean2 = np.mean(raw_lsnr_2)
        mean1 = np.mean(raw_lsnr_1)

        if mean2 < 1. or mean1 < 1.:
            continue

        intensity_div_2 = mean2 / mean1
        raw_lsnr_2 /= intensity_div_2
        sim_lsnr_2 /= intensity_div_2

        # this_max = max(first_p999(raw_lsnr_1), first_p999(raw_lsnr_2))
        this_max = max(raw_lsnr_1.max(), raw_lsnr_2.max())

        raw_lsnr_1 /= this_max
        raw_lsnr_2 /= this_max
        sim_lsnr_1 /= this_max
        sim_lsnr_2 /= this_max
        outinfo.info('cell No.{:d} normalized. raw L2/L1 ratio {: .3f}. max raw intensity {: .1f}\n'.format(idx, intensity_div_2, this_max))

        # train
        outinfo.info('training_patch_num {}. average_photon_1 {}\n'.format(training_patch_num, average_photon_1))
        outinfo.info('training_patch_num {}. average_photon_2 {}\n'.format(training_patch_num, average_photon_2))

        np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), raw_lsnr_2)
        np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), sim_lsnr_2)
        np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), raw_lsnr_1)
        np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), sim_lsnr_1)
        np.save(os.path.join(training_data_folder, 'train_Para', '{:0>5}.npy'.format(training_patch_num)), para)
        save_json(json_dict, os.path.join(training_data_folder, 'train_Json', '{:0>5}.json'.format(training_patch_num)))
        training_patch_num += 1

        np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), raw_lsnr_1)
        np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), sim_lsnr_1)
        np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), raw_lsnr_2)
        np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), sim_lsnr_2)
        np.save(os.path.join(training_data_folder, 'train_Para', '{:0>5}.npy'.format(training_patch_num)), para)
        save_json(json_dict, os.path.join(training_data_folder, 'train_Json', '{:0>5}.json'.format(training_patch_num)))
        training_patch_num += 1

        if num_of_cell < 5 or idx % (num_of_cell // 5) == 0:  # val

            # raw_lsnr_1 = raw_lsnr_1[..., 512:-512, 512:-512]
            # sim_lsnr_1 = sim_lsnr_1[..., 1024:-1024, 1024:-1024]

            outinfo.info('val_patch_num {}. average_photon_1 {}\n'.format(val_patch_num, average_photon_1))
            np.save(os.path.join(training_data_folder, 'val_Raw_LSNR_1', '{:0>5}.npy'.format(val_patch_num)), raw_lsnr_1)
            np.save(os.path.join(training_data_folder, 'val_SIM_LSNR_1', '{:0>5}.npy'.format(val_patch_num)), sim_lsnr_1)
            np.save(os.path.join(training_data_folder, 'val_Para', '{:0>5}.npy'.format(val_patch_num)), para)
            save_json(json_dict, os.path.join(training_data_folder, 'val_Json', '{:0>5}.json'.format(val_patch_num)))
            val_patch_num += 1


def train(dataset_root: str):
    # opt = AI_parse_read(os.path.join(os.path.split(os.getcwd())[0], 'SSR_options/train_recon2d_phct.json'))

    if os.path.exists(os.path.join(os.path.split(os.getcwd())[0], 'SSR_options/train_recon2d_phct.json')):
        opt = AI_parse_read(os.path.join(os.path.split(os.getcwd())[0], 'SSR_options/train_recon2d_phct.json'))
    elif os.path.exists(os.path.join(os.getcwd(), 'SSR_options/train_recon2d_phct.json')):
        opt = AI_parse_read(os.path.join(os.getcwd(), 'SSR_options/train_recon2d_phct.json'))
    elif os.path.exists(os.path.join(os.getcwd(), '../SSR_options/train_recon2d_phct.json')):
        opt = AI_parse_read(os.path.join(os.getcwd(), '../SSR_options/train_recon2d_phct.json'))
    else:
        raise FileNotFoundError('')

    opt['dataset_root'] = dataset_root

    opt['supervise'] = "self-supervised"

    opt['dataaug_random_angle_rotate'] = True
    opt['read_whole_dataset_toGPU'] = True
    opt['read_whole_dataset_toCPU'] = True

    opt['num_iter_interval_val'] = 10000
    opt['num_iter_interval_save'] = 10000

    opt = AI_parse_process(opt)
    main(json_path=None, opt=opt)


def refine(dataset_root: str, pretrained_netG: str):
    # opt = AI_parse_read(os.path.join(os.path.split(os.getcwd())[0], 'SSR_options/train_recon2d_phct.json'))

    if os.path.exists(os.path.join(os.path.split(os.getcwd())[0], 'SSR_options/train_recon2d_phct.json')):
        opt = AI_parse_read(os.path.join(os.path.split(os.getcwd())[0], 'SSR_options/train_recon2d_phct.json'))
    elif os.path.exists(os.path.join(os.getcwd(), 'SSR_options/train_recon2d_phct.json')):
        opt = AI_parse_read(os.path.join(os.getcwd(), 'SSR_options/train_recon2d_phct.json'))
    elif os.path.exists(os.path.join(os.getcwd(), '../SSR_options/train_recon2d_phct.json')):
        opt = AI_parse_read(os.path.join(os.getcwd(), '../SSR_options/train_recon2d_phct.json'))
    else:
        raise FileNotFoundError('')

    opt['supervise'] = "self-supervised"
    opt['tail'] = 'PL'

    opt['dataset_root'] = dataset_root
    opt['pretrained_netG'] = pretrained_netG

    assert opt['dataset_root'] is not None and len(opt['dataset_root']) > 1
    assert opt['pretrained_netG'] is not None and len(opt['pretrained_netG']) > 1

    # transfer learning
    opt['dataloader_batch_size'] = 1
    opt['in_patch_size'] = 1536
    opt['G_optimizer_lr'] = 1e-5
    opt['G_scheduler_IterNum'] = 20000

    opt['dataaug_random_angle_rotate'] = False
    opt['read_whole_dataset_toGPU'] = True
    opt['read_whole_dataset_toCPU'] = True

    opt = AI_parse_process(opt)
    main(json_path=None, opt=opt)


def test(modelPath: str, path: str):
    path_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if fname == "seq1_TIRF-SIM488_GreenCh-DL.mrc":
                path_list.append(os.path.join(dirpath, fname))

    model = PHCT(9, 1, scale=2, para1='RLFB+ET+Conv3', para2='RLFB+ET+Conv3', para3=True, para4=48)
    model.load_state_dict(torch.load(modelPath), strict=True)
    model = model.to(device)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False

    for file in path_list:
        rm = ReadMRC(file)
        header = make_sr_header_assign_scale(rm.header.copy(), rm.opt, [2, 2, 1])
        header[57] = 1 * 65536 + 1  # step=1
        header = tuple(header)

        wm = WriteMRC(file[:-4] + '_SSR-SIM.mrc', header, compress='uint16norm')

        json_path = os.path.join(file[:-4] + '_output', os.path.split(file)[1][:-4] + '.json')
        json_dict = sir_parse2dict(json_path)  # return a class
        para = json2para(json_dict, None)

        para = torch.from_numpy(para).to(device).squeeze().unsqueeze(0)

        rm_data = rm.get_total_data_as_mat(convert_to_tensor=True, convert_to_float32=True).to(device) - json_dict['camera_background']
        rm_data_step0 = rm_data[:, 0]
        rm_data_step1 = rm_data[:, 1]

        rm_data_step0 = raw_weighted_torch(rm_data_step0)
        rm_data_step1 = raw_weighted_torch(rm_data_step1)

        data_shape = rm_data_step0.shape
        T, O, P, C, D, H, W = data_shape
        out_shape = [T, 1, 1, C, D, 2 * H, 2 * W]

        max_g = rm_data_step0.max()
        data_step0 = rm_data_step0.reshape(T, O * P, H, W) / max_g
        psim = generate_pattern(data_step0, para, [json_path], pattern_size='sim').to(device)
        with torch.no_grad():
            result_step0 = model.forward_with_pattern(data_step0, None, psim) * max_g

        max_g = rm_data_step1.max()
        data_step1 = rm_data_step1.reshape(T, O * P, H, W) / max_g
        psim = generate_pattern(data_step1, para, [json_path], pattern_size='sim').to(device)
        with torch.no_grad():
            result_step1 = model.forward_with_pattern(data_step1, None, psim) * max_g

        result = ((result_step0 + result_step1) / 2).reshape(*out_shape)

        wm.write_data_append(result)


if __name__ == '__main__':

    # ----------------------------------------
    # [Train SSR-SIM Model > DataSet PreProcessing]
    # ----------------------------------------
    sirecon_noisy_raw_data_for_training(r'K:\LifeAct')

    # # ----------------------------------------
    # # [Train SSR-SIM Model > DataSet Organization]
    # # ----------------------------------------
    convert_to_training_dataset(r'K:\LifeAct')

    # # ----------------------------------------
    # # [Train SSR-SIM Model]
    # # ----------------------------------------
    train(r'K:\LifeAct_DST')
    # # If the created file contains a timestamp, e.g., LifeAct_DST_20200531, please add a timestamp likewise

    # # ----------------------------------------
    # # [Refine SSR-SIM Model]
    # # ----------------------------------------
    # refine(r'K:\LifeAct_DST', r'E:\LifeAct_DST\self-supervised_reconstruction_phct\model\latest_G.pth')
    # # If the created file contains a timestamp, e.g., LifeAct_DST_20200531, please add a timestamp likewise

    # # ----------------------------------------
    # # [Test SSR-SIM Model -> Model Forward]
    # # ----------------------------------------
    # test(r'K:\LifeAct_DST\self-supervised_reconstruction_phct_pl\model\latest_G.pth', r'K:\LifeAct')
    # # test('PreTrained Model for LifeAct DataSet\latest_G.pth', r'K:\LifeAct')
