"""
Download BioSR dataset: (1) "Microtubules.zip" (2) "CCPs.zip" (3) "F-actin.zip"
from https://figshare.com/articles/dataset/BioSR/13264793, and unzip them to
path="home/BioSR" (Linux) or path=r"K:\BioSR" (Windows)
"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.split(os.getcwd())[0])

import torch
import numpy as np

device = torch.device('cuda')

from utils.option import dict2str, sir_parse2dict, save_json, AI_parse_read, AI_parse_process
from utils.tools import mkdir, WriteOutputTxt, mkdir_with_time
from utils.image_process import F_cal_average_photon
from utils.mrc import make_sr_header, ReadMRC, WriteMRC, make_sr_header_assign_scale

from SIR_core.base import make_otf_2d
from SIR_core.pe import guess_k0, force_modamp, save_wave_vector, SIMEstimate2D
from SIR_core.r2 import SIMReconstr2D

from SSR_dst.common import ReadMrcImage_SSR, ReadMrcImage, raw_weighted_np, ParaProcess, raw_weighted_torch
from SSR_logic.main_train_plain import main

from SSR_model.common import generate_pattern
from SSR_model.network_phct import PHCT

def generate_noisy_raw_data_for_training(in_folder: str, out_folder: str, sample: str, snr: str):
    conversion_factor = 0.6026
    camera_background = 100
    gaussian_noise_std = 10

    if sample == 'Microtubules':
        inpath = os.path.join(in_folder, 'Microtubules')
        outpath = os.path.join(out_folder, 'Microtubules_' + snr)
    elif sample == 'F-actin':
        inpath = os.path.join(in_folder, 'F-actin')
        outpath = os.path.join(out_folder, 'F-actin_' + snr)
    elif sample == 'CCPs':
        inpath = os.path.join(in_folder, 'CCPs')
        outpath = os.path.join(out_folder, 'CCPs_' + snr)
    else:
        raise RuntimeError

    mkdir(outpath)

    if snr == 'hsnr':
        photon_list = [180, 240, 300, 360, 420, 480, 540, 600]
    elif snr == 'lsnr':
        if sample == 'Microtubules' or sample == 'CCPs':
            photon_list = [25, 35, 45, 55, 70, 85, 100, 120]
        elif sample == 'F-actin':
            photon_list = [60, 80, 100, 120, 140, 160, 180, 200]
        else:
            raise RuntimeError
    else:
        raise RuntimeError

    for idx in range(1, 50 + 1):

        print(idx)

        rm = ReadMRC(os.path.join(inpath, 'Cell_{:0>3}'.format(idx), 'RawSIMData_gt.mrc'))

        raw = rm.get_total_data_as_mat(convert_to_tensor=False)

        mkdir(os.path.join(outpath, 'Cell_{:0>2}'.format(idx)))

        raw = (raw.astype(np.float32) - camera_background).clip(0, 65535)

        gt_photon = F_cal_average_photon(raw, mean_axis=(0, 1, 2, 3, 4), conversion_factor=conversion_factor)

        gt = camera_background + raw

        gt_raw = WriteMRC(os.path.join(outpath, "Cell_{:0>2}".format(idx), "GT_Raw.mrc"), header=rm.header)
        gt_raw.write_data_append(gt.clip(0, 65535).astype(np.uint16))

        noisy_raw = WriteMRC(os.path.join(outpath, 'Cell_{:0>2}'.format(idx), 'Noisy_Raw.mrc'), header=rm.header)

        for photon_count in photon_list:

            for _ in range(2):
                # gt -> gt at low intensity
                gt_intensity_norm = raw / gt_photon * photon_count

                # [camera count -> photon], [float32->uint16], [poisson process], [uint16->float32], [photon count -> camera]
                poisson = np.random.poisson((gt_intensity_norm * conversion_factor).clip(0, 65535).round().astype(np.uint16)).astype(np.float32) / conversion_factor

                noisy = camera_background + poisson + gaussian_noise_std * np.random.randn(*raw.shape)

                noisy_raw.write_data_append(noisy.clip(0, 65535).astype(np.uint16))


def sirecon_noisy_raw_data_for_training(sirecon_path: str, sample: str):
    if sample == 'Microtubules':
        json_file = 'biosr_lownagi_488.json'
    elif sample in ['F-actin', 'CCPs']:
        json_file = 'biosr_tirf_488.json'
    else:
        raise NotImplementedError

    gt_files, noisy_files = [], []
    for idx in range(50):
        gt_files.append(os.path.join(sirecon_path, r'Cell_{:0>2}'.format(idx + 1), 'GT_Raw.mrc'))
        noisy_files.append(os.path.join(sirecon_path, r'Cell_{:0>2}'.format(idx + 1), 'Noisy_Raw.mrc'))

    for idx in range(len(gt_files)):

        mrc_file = gt_files[idx]
        opt_json = sir_parse2dict(os.path.join(os.path.split(os.getcwd())[0], 'SIR_options', json_file))  # return a class
        raw = ReadMRC(mrc_file, opt=opt_json)  # <opt append>
        opt = raw.opt
        save_json(opt, mrc_file[:-8] + '.json')
        opt['outinfo'] = WriteOutputTxt(mrc_file[:-8] + '.txt')
        opt['outinfo'].info('\n-------- all parameters --------\n')
        opt['outinfo'].info(dict2str(opt) + '\n')
        k0_guess = guess_k0(opt, device=device)
        otf = make_otf_2d(opt, k0=k0_guess, device=device)

        opt['outinfo'].info('\n-------- do estimate pattern --------\n')
        opt['outinfo'].info('load one tps data\n')
        data = raw.get_timepoint_data_as_mat(begin_timepoint=0, read_timepoint=1).mean(axis=0, keepdims=True)
        sem = SIMEstimate2D(k0_guess=k0_guess, opt=opt, otf=otf, device=device)
        with torch.no_grad():
            wave_vector = sem.esti(data)
        if opt['if_force_modamp']: wave_vector = force_modamp(wave_vector, opt)
        save_wave_vector(wave_vector=wave_vector, path=mrc_file[:-8])
        srm = SIMReconstr2D(opt=opt, wave_vector=wave_vector, otf=otf, device=device)

        opt['outinfo'].info('\n-------- do reconstruction --------\n')
        CW = WriteMRC(mrc_file.replace('Raw', 'SIM'), make_sr_header(raw.header, opt))
        batch_idx = 0
        while True:
            data = raw.get_next_timepoint_batch()
            if data is None: break
            opt['outinfo'].info('reconstruction of GT | batch_idx {}\n'.format(batch_idx))
            with torch.no_grad():
                result = srm.reconstr(data)  # [TOPCDHW] -> [TDHW]
            CW.write_data_append(result.unsqueeze(1).unsqueeze(1).unsqueeze(1))  # [TDHW] -> [TOPCDHW]
            batch_idx += 1

        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
        mrc_file = noisy_files[idx]
        save_wave_vector(wave_vector=wave_vector, path=mrc_file[:-8])
        opt_json = sir_parse2dict(os.path.join(os.path.split(os.getcwd())[0], 'SIR_options', json_file))  # return a class
        raw = ReadMRC(mrc_file, opt=opt_json)  # <opt append>
        opt = raw.opt
        save_json(opt, mrc_file[:-8] + '.json')
        opt['outinfo'] = WriteOutputTxt(mrc_file[:-8] + '.txt')
        opt['outinfo'].info('\n-------- all parameters --------\n')
        opt['outinfo'].info(dict2str(opt) + '\n')
        opt['outinfo'].info('\n-------- do estimate pattern --------\n')
        opt['outinfo'].info('pattern parameters are given by ones estimated with high-snr data \n')
        opt['outinfo'].info('\n-------- do reconstruction --------\n')
        CW = WriteMRC(mrc_file.replace('Raw', 'SIM'), make_sr_header(raw.header, opt))

        batch_idx = 0
        while True:
            data = raw.get_next_timepoint_batch()
            if data is None: break
            opt['outinfo'].info('reconstruction of noisy | batch_idx {}\n'.format(batch_idx))
            with torch.no_grad():
                result = srm.reconstr(data)  # [TOPCDHW] -> [TDHW]
            CW.write_data_append(result.unsqueeze(1).unsqueeze(1).unsqueeze(1))  # [TDHW] -> [TOPCDHW]
            batch_idx += 1


def convert_to_training_dataset(dataset_folder: str):
    min_allowed_photon = 5
    training_num_of_cell = 40
    validate_num_of_cell = 10  #

    # ----------------------------------------
    #            <check the number of cells available>
    # ----------------------------------------
    cell_gt_raw, cell_gt_sim, cell_noisy_raw, cell_noisy_sim = [], [], [], []
    for idx in range(50):
        cell_gt_raw.append(os.path.join(dataset_folder, r'Cell_{:0>2}'.format(idx + 1), 'GT_Raw.mrc'))
        cell_gt_sim.append(os.path.join(dataset_folder, r'Cell_{:0>2}'.format(idx + 1), 'GT_SIM.mrc'))
        cell_noisy_raw.append(os.path.join(dataset_folder, r'Cell_{:0>2}'.format(idx + 1), 'Noisy_Raw.mrc'))
        cell_noisy_sim.append(os.path.join(dataset_folder, r'Cell_{:0>2}'.format(idx + 1), 'Noisy_SIM.mrc'))

    # ----------------------------------------
    #            <mkdir and config-txt>
    # ----------------------------------------
    training_data_folder = os.path.join(os.path.split(dataset_folder)[0], os.path.split(dataset_folder)[1] + '_DST')
    if not os.path.exists(training_data_folder):
        mkdir(training_data_folder)
    else:
        training_data_folder = mkdir_with_time(os.path.join(os.path.split(dataset_folder)[0], os.path.split(dataset_folder)[1] + '_DST'))
    for s_1 in ['train']:
        for s_2 in ['Raw_LSNR_1', 'SIM_LSNR_1', 'Raw_LSNR_2', 'SIM_LSNR_2', 'Raw_HSNR', 'SIM_HSNR', 'Para', 'Json']:
            mkdir(os.path.join(training_data_folder, s_1 + '_' + s_2))
    for s_1 in ['val']:
        for s_2 in ['Raw_LSNR_1', 'SIM_LSNR_1', 'Raw_HSNR', 'SIM_HSNR', 'Para', 'Json']:
            mkdir(os.path.join(training_data_folder, s_1 + '_' + s_2))
    outinfo = WriteOutputTxt(os.path.join(training_data_folder, 'config.txt'))
    outinfo.info('\n-------- all parameters --------\n')
    outinfo.info('training_num_of_cell: {:d}\n'.format(training_num_of_cell))
    outinfo.info('validate_num_of_cell: {:d}\n'.format(validate_num_of_cell))
    outinfo.info('min_allowed_photon: {: .1f}\n'.format(min_allowed_photon))

    # ----------------------------------------
    #            <Process and Save>
    # ----------------------------------------
    outinfo.info('\n-------- Processing --------')
    training_patch_num = 0
    val_patch_num = 0

    val_avg_photon_list = []
    for idx in range(0, validate_num_of_cell + training_num_of_cell):
        outinfo.info('\non processing cell No.{:d} as {} data. '.format(idx, ['val', 'train'][idx >= validate_num_of_cell]))

        rm = ReadMRC(cell_gt_raw[idx])
        json_dict = (sir_parse2dict(cell_gt_raw[idx][:-8] + '.json'))  # return a class

        choice_list = None
        outinfo.info('use all the slices')

        gt_raw = ReadMrcImage(cell_gt_raw[idx]).astype(np.float32) - json_dict['camera_background']
        noisy_raw_1, noisy_raw_2 = ReadMrcImage_SSR(cell_noisy_raw[idx])
        noisy_raw_1 = noisy_raw_1.astype(np.float32) - json_dict['camera_background']
        noisy_raw_2 = noisy_raw_2.astype(np.float32) - json_dict['camera_background']
        gt_raw, noisy_raw_1, noisy_raw_2 = raw_weighted_np(gt_raw), raw_weighted_np(noisy_raw_1), raw_weighted_np(noisy_raw_2)
        gt_sim = ReadMrcImage(cell_gt_sim[idx]).astype(np.float32)
        noisy_sim_1, noisy_sim_2 = ReadMrcImage_SSR(cell_noisy_sim[idx])
        noisy_sim_1 = noisy_sim_1.astype(np.float32)
        noisy_sim_2 = noisy_sim_2.astype(np.float32)

        para = ParaProcess(np.load(cell_gt_raw[idx][:-8] + '_k0.npy'), np.load(cell_gt_raw[idx][:-8] + '_phase.npy'), rm.opt['height_space_sampling'])
        outinfo.info('pattern para [{}]\n'.format(", ".join(map(lambda x: str(round(x, 4)), para.tolist()))))

        for slice_num in range(noisy_raw_1.shape[0]):

            average_photon_1 = F_cal_average_photon(noisy_raw_1[slice_num:slice_num + 1, ...], mean_axis=(0, 1, 2, 3, 4))
            average_photon_2 = F_cal_average_photon(noisy_raw_2[slice_num:slice_num + 1, ...], mean_axis=(0, 1, 2, 3, 4))
            average_photon = min(average_photon_1, average_photon_2)
            if idx >= validate_num_of_cell:
                if average_photon < min_allowed_photon:
                    continue
                if noisy_sim_1.mean() < 1e-20 or noisy_sim_2.mean() < 1e-20:
                    continue

            noisy_raw_slice_1 = noisy_raw_1[slice_num:slice_num + 1, ...].copy()
            noisy_raw_slice_2 = noisy_raw_2[slice_num:slice_num + 1, ...].copy()
            gt_raw_slice = gt_raw[0:0 + 1, ...].copy()
            noisy_sim_slice_1 = noisy_sim_1[slice_num:slice_num + 1, ...].copy()
            noisy_sim_slice_2 = noisy_sim_2[slice_num:slice_num + 1, ...].copy()
            gt_sim_slice = gt_sim[0:0 + 1, ...].copy()

            intensity_div_1 = np.mean(gt_raw_slice) / np.mean(noisy_raw_slice_1)
            gt_raw_slice /= intensity_div_1
            gt_sim_slice /= intensity_div_1
            intensity_div_2 = np.mean(noisy_raw_slice_2) / np.mean(noisy_raw_slice_1)
            noisy_raw_slice_2 /= intensity_div_2
            noisy_sim_slice_2 /= intensity_div_2

            this_max = max(np.max(gt_raw_slice), np.max(noisy_raw_slice_1), np.max(noisy_raw_slice_2))

            noisy_raw_slice_1 /= this_max
            noisy_raw_slice_2 /= this_max
            gt_raw_slice /= this_max
            noisy_sim_slice_1 /= this_max
            noisy_sim_slice_2 /= this_max
            gt_sim_slice /= this_max
            outinfo.info('cell No.{:d}, slice No.{:d} normalized. raw H/L1 ratio {: .1f}. raw L2/L1 ratio {: .1f}. max raw and sim intensity {: .1f}\n'
                         .format(idx, slice_num, intensity_div_1, intensity_div_2, this_max))

            if idx >= validate_num_of_cell:

                np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), noisy_raw_slice_1)
                np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), noisy_sim_slice_1)
                np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), noisy_raw_slice_2)
                np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), noisy_sim_slice_2)
                np.save(os.path.join(training_data_folder, 'train_Raw_HSNR', '{:0>5}.npy'.format(training_patch_num)), gt_raw_slice)
                np.save(os.path.join(training_data_folder, 'train_SIM_HSNR', '{:0>5}.npy'.format(training_patch_num)), gt_sim_slice)
                np.save(os.path.join(training_data_folder, 'train_Para', '{:0>5}.npy'.format(training_patch_num)), para)
                save_json(json_dict, os.path.join(training_data_folder, 'train_Json', '{:0>5}.json'.format(training_patch_num)))
                training_patch_num += 1

                np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), noisy_raw_slice_2)
                np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_1', '{:0>5}.npy'.format(training_patch_num)), noisy_sim_slice_2)
                np.save(os.path.join(training_data_folder, 'train_Raw_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), noisy_raw_slice_1)
                np.save(os.path.join(training_data_folder, 'train_SIM_LSNR_2', '{:0>5}.npy'.format(training_patch_num)), noisy_sim_slice_1)
                np.save(os.path.join(training_data_folder, 'train_Raw_HSNR', '{:0>5}.npy'.format(training_patch_num)), gt_raw_slice)
                np.save(os.path.join(training_data_folder, 'train_SIM_HSNR', '{:0>5}.npy'.format(training_patch_num)), gt_sim_slice)
                np.save(os.path.join(training_data_folder, 'train_Para', '{:0>5}.npy'.format(training_patch_num)), para)
                save_json(json_dict, os.path.join(training_data_folder, 'train_Json', '{:0>5}.json'.format(training_patch_num)))
                training_patch_num += 1

                print(training_patch_num)

            else:

                val_avg_photon_list.append(average_photon)
                np.save(os.path.join(training_data_folder, 'val_Raw_LSNR_1', '{:0>5}.npy'.format(val_patch_num)), noisy_raw_slice_1)
                np.save(os.path.join(training_data_folder, 'val_SIM_LSNR_1', '{:0>5}.npy'.format(val_patch_num)), noisy_sim_slice_1)
                np.save(os.path.join(training_data_folder, 'val_Raw_HSNR', '{:0>5}.npy'.format(val_patch_num)), gt_raw_slice)
                np.save(os.path.join(training_data_folder, 'val_SIM_HSNR', '{:0>5}.npy'.format(val_patch_num)), gt_sim_slice)
                np.save(os.path.join(training_data_folder, 'val_Para', '{:0>5}.npy'.format(val_patch_num)), para)
                save_json(json_dict, os.path.join(training_data_folder, 'val_Json', '{:0>5}.json'.format(val_patch_num)))
                val_patch_num += 1

    outinfo.info('\n -------- average photon count -------- \n')
    for val_idx, photon_num in enumerate(val_avg_photon_list):
        outinfo.info('val data No.{:0>5}, average photon count {:0>5} \n'.format(val_idx, round(photon_num)))


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

    opt['supervise'] = "self-supervised-val"

    opt['read_whole_dataset_toGPU'] = True
    opt['read_whole_dataset_toCPU'] = True

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

    opt['supervise'] = "self-supervised-val"
    opt['net_tail'] = '_pl'

    opt['dataset_root'] = dataset_root
    opt['pretrained_netG'] = pretrained_netG

    assert os.path.exists(opt['pretrained_netG'])

    # transfer learning
    opt['dataloader_batch_size'] = 1
    opt['in_patch_size'] = 502
    opt['G_optimizer_lr'] = 3e-5
    opt['G_scheduler_IterNum'] = 20000

    opt['read_whole_dataset_toGPU'] = True
    opt['read_whole_dataset_toCPU'] = True

    opt = AI_parse_process(opt)
    main(json_path=None, opt=opt)


#  Generating noisy raw data at varying SNRs

def generate_noisy_raw_data_for_test(in_folder, out_folder, sample):
    conversion_factor = 0.6026
    camera_background = 100
    gaussian_noise_std = 10

    if sample == 'Microtubules':
        inpath = os.path.join(in_folder, 'Microtubules')
        outpath = os.path.join(out_folder, 'Microtubules')
    elif sample == 'F-actin':
        inpath = os.path.join(in_folder, 'F-actin')
        outpath = os.path.join(out_folder, 'F-actin')
    elif sample == 'CCPs':
        inpath = os.path.join(in_folder, 'CCPs')
        outpath = os.path.join(out_folder, 'CCPs')
    else:
        raise RuntimeError

    mkdir(outpath)

    photon_list = [10, 15, 20, 25, 35, 45, 55, 70, 85, 100, 120, 180, 240, 300, 360, 420, 480, 540, 600]

    for idx in range(1, 10 + 1):

        print(idx)

        rm = ReadMRC(os.path.join(inpath, 'Cell_{:0>3}'.format(idx), 'RawSIMData_gt.mrc'))

        raw = rm.get_total_data_as_mat(convert_to_tensor=False)

        raw = raw[..., (502 - 480) // 2: -(502 - 480) // 2, (502 - 480) // 2: -(502 - 480) // 2]
        header = rm.header.copy()
        header = list(header)
        header[0], header[1] = 480, 480
        header = tuple(header)

        mkdir(os.path.join(outpath, r'Cell_{:0>2}'.format(idx)))

        raw = (raw.astype(np.float32) - camera_background).clip(0, 65535)

        gt_photon = F_cal_average_photon(raw, mean_axis=(0, 1, 2, 3, 4), conversion_factor=conversion_factor)

        gt = camera_background + raw

        wm = WriteMRC(os.path.join(outpath, r'Cell_{:0>2}'.format(idx), 'GT_Noisy_Raw.mrc'), header=header)
        wm.write_data_append(gt.clip(0, 65535).astype(np.uint16))

        for photon_count in photon_list:
            # gt -> gt at low intensity
            gt_intensity_norm = raw / gt_photon * photon_count

            # [camera count -> photon], [float32->uint16], [poisson process], [uint16->float32], [photon count -> camera]
            poisson = np.random.poisson((gt_intensity_norm * conversion_factor).clip(0, 65535).round().astype(np.uint16)).astype(np.float32) / conversion_factor

            noisy = camera_background + poisson + gaussian_noise_std * np.random.randn(*raw.shape)

            wm.write_data_append(noisy.clip(0, 65535).astype(np.uint16))


def performing_SIEsti(sirecon_path: str, sample: str, need_SIRecon=False):
    if sample == 'Microtubules':
        json_file = 'biosr_lownagi_488.json'
    elif sample in ['F-actin', 'CCPs']:
        json_file = 'biosr_tirf_488.json'
    else:
        raise NotImplementedError

    files = []
    for idx in range(10):
        files.append(os.path.join(sirecon_path, r'Cell_{:0>2}'.format(idx + 1), 'GT_Noisy_Raw.mrc'))

    for mrc_file in files:

        opt_json = sir_parse2dict(os.path.join(os.path.split(os.getcwd())[0], 'SIR_options', json_file))  # return a class
        raw = ReadMRC(mrc_file, opt=opt_json)  # <opt append>
        opt = raw.opt
        save_json(opt, mrc_file[:-8] + '.json')
        opt['outinfo'] = WriteOutputTxt(mrc_file[:-8] + '.txt')
        opt['outinfo'].info('\n-------- all parameters --------\n')
        opt['outinfo'].info(dict2str(opt) + '\n')
        k0_guess = guess_k0(opt, device=device)
        otf = make_otf_2d(opt, k0=k0_guess, device=device)

        opt['outinfo'].info('\n-------- do estimate pattern --------\n')
        opt['outinfo'].info('load one tps data\n')
        data = raw.get_timepoint_data_as_mat(begin_timepoint=0, read_timepoint=1).mean(axis=0, keepdims=True)
        sem = SIMEstimate2D(k0_guess=k0_guess, opt=opt, otf=otf, device=device)
        with torch.no_grad():
            wave_vector = sem.esti(data)
        if opt['if_force_modamp']: wave_vector = force_modamp(wave_vector, opt)
        save_wave_vector(wave_vector=wave_vector, path=mrc_file[:-8])

        if need_SIRecon:

            srm = SIMReconstr2D(opt=opt, wave_vector=wave_vector, otf=otf, device=device)

            opt['outinfo'].info('\n-------- do reconstruction --------\n')
            CW = WriteMRC(mrc_file.replace('Raw', 'SIM'), make_sr_header(raw.header, opt))
            batch_idx = 0
            while True:
                data = raw.get_next_timepoint_batch()
                if data is None: break
                if batch_idx == 0:
                    opt['outinfo'].info('reconstruction of GT\n')
                else:
                    opt['outinfo'].info('reconstruction of Noisy | snr level {} |\n'.format(batch_idx))
                with torch.no_grad():
                    result = srm.reconstr(data)  # [TOPCDHW] -> [TDHW]
                CW.write_data_append(result.unsqueeze(1).unsqueeze(1).unsqueeze(1))  # [TDHW] -> [TOPCDHW]
                batch_idx += 1


def ssrsirecon(modelPath, dataSetPath):
    model = PHCT(9, 1, scale=2, para1='RLFB+ET+Conv3', para2='RLFB+ET+Conv3', para3=True, para4=48)
    model.load_state_dict(torch.load(modelPath), strict=True)
    model = model.to(device)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False

    for cell_idx in range(10):

        dataPath = os.path.join(dataSetPath, r'Cell_{:0>2}'.format(cell_idx + 1), 'GT_Noisy_Raw.mrc')

        rm = ReadMRC(dataPath)
        header = make_sr_header_assign_scale(rm.header.copy(), rm.opt, [2, 2, 1])
        header = tuple(header)

        wm = WriteMRC(dataPath[:-8] + '_SSR-SIM.mrc', header, compress='float32')

        json_path = os.path.join(dataPath[:-8] + '.json')
        json = sir_parse2dict(json_path)  # return a class
        k0 = np.load(os.path.join(dataPath[:-8] + '_k0.npy'))
        pha = np.load(os.path.join(dataPath[:-8] + '_phase.npy'))

        para = ParaProcess(k0, pha, rm.opt['height_space_sampling'])

        para = torch.from_numpy(para).to(device).squeeze().unsqueeze(0)

        for tps_idx in range(0, rm.opt['num_timepoint']):
            rm_data = rm.get_timepoint_data_as_mat(tps_idx, 1).to(device) - json['camera_background']
            rm_data = raw_weighted_torch(rm_data)

            data_shape = rm_data.shape

            # max_g = first_p999(rm_data)
            max_g = rm_data.max()

            T, O, P, C, D, H, W = data_shape
            out_shape = [T, 1, 1, C, D, 2 * H, 2 * W]

            data = rm_data.reshape(T, O * P, H, W) / max_g

            psim = generate_pattern(data, para, json, pattern_size='sim').to(device)

            with torch.no_grad():
                result = model.forward_with_pattern(data, None, psim)

            result = (result * max_g).reshape(*out_shape)

            wm.write_data_append(result)


if __name__ == '__main__':
    # ----------------------------------------
    # [Train SSR-SIM Model > DataSet Generation]
    # ----------------------------------------
    # generate_noisy_raw_data_for_training(r'K:\BioSR', r'K:\BioSR_Training_Data', 'CCPs', 'hsnr')
    # generate_noisy_raw_data_for_training(r'K:\BioSR', r'K:\BioSR_Training_Data', 'CCPs', 'lsnr')
    # generate_noisy_raw_data_for_training(r'K:\BioSR', r'K:\BioSR_Training_Data', 'F-actin', 'hsnr')
    # generate_noisy_raw_data_for_training(r'K:\BioSR', r'K:\BioSR_Training_Data', 'F-actin', 'lsnr')
    # generate_noisy_raw_data_for_training(r'K:\BioSR', r'K:\BioSR_Training_Data', 'Microtubules', 'hsnr')
    generate_noisy_raw_data_for_training(r'K:\BioSR', r'K:\BioSR_Training_Data', 'Microtubules', 'lsnr')

    # # ----------------------------------------
    # # [Train SSR-SIM Model > DataSet PreProcessing]
    # # ----------------------------------------
    # # sirecon_noisy_raw_data_for_training(r'K:\BioSR_Training_Data\CCPs_hsnr', 'CCPs')
    # # sirecon_noisy_raw_data_for_training(r'K:\BioSR_Training_Data\CCPs_lsnr', 'CCPs')
    # # sirecon_noisy_raw_data_for_training(r'K:\BioSR_Training_Data\F-actin_hsnr', 'F-actin')
    # # sirecon_noisy_raw_data_for_training(r'K:\BioSR_Training_Data\F-actin_lsnr', 'F-actin')
    # # sirecon_noisy_raw_data_for_training(r'K:\BioSR_Training_Data\Microtubules_hsnr', 'Microtubules')
    # sirecon_noisy_raw_data_for_training(r'K:\BioSR_Training_Data\Microtubules_lsnr', 'Microtubules')

    # # ----------------------------------------
    # # [Train SSR-SIM Model > DataSet Organization]
    # # ----------------------------------------
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\CCPs_hsnr')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\CCPs_lsnr')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\F-actin_hsnr')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\F-actin_lsnr')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\Microtubules_hsnr')
    # convert_to_training_dataset(r'K:\BioSR_Training_Data\Microtubules_lsnr')

    # # ----------------------------------------
    # # [Train SSR-SIM Model]
    # # ----------------------------------------
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\CCPs_hsnr_DST')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\CCPs_lsnr_DST')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\F-actin_hsnr_DST')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\F-actin_lsnr_DST')
    # # convert_to_training_dataset(r'K:\BioSR_Training_Data\Microtubules_hsnr_DST')
    # convert_to_training_dataset(r'K:\BioSR_Training_Data\Microtubules_lsnr_DST')

    # # ----------------------------------------
    # # [Refine SSR-SIM Model]
    # # ----------------------------------------
    # # refine(r'K:\BioSR_Training_Data\CCPs_hsnr_DST', r'K:\BioSR_Training_Data\CCPs_hsnr_DST\self-supervised-val_reconstruction_phct\model\PSNR_BEST_G.pth')
    # # refine(r'K:\BioSR_Training_Data\CCPs_lsnr_DST', r'K:\BioSR_Training_Data\CCPs_lsnr_DST\self-supervised-val_reconstruction_phct\model\PSNR_BEST_G.pth')
    # # refine(r'K:\BioSR_Training_Data\F-actin_hsnr_DST', r'K:\BioSR_Training_Data\F-actin_hsnr_DST\self-supervised-val_reconstruction_phct\model\PSNR_BEST_G.pth')
    # # refine(r'K:\BioSR_Training_Data\F-actin_lsnr_DST', r'K:\BioSR_Training_Data\F-actin_lsnr_DST\self-supervised-val_reconstruction_phct\model\PSNR_BEST_G.pth')
    # # refine(r'K:\BioSR_Training_Data\Microtubules_hsnr_DST', r'K:\BioSR_Training_Data\Microtubules_hsnr_DST\self-supervised-val_reconstruction_phct\model\PSNR_BEST_G.pth')
    # refine(r'K:\BioSR_Training_Data\Microtubules_lsnr_DST', r'K:\BioSR_Training_Data\Microtubules_lsnr_DST\self-supervised-val_reconstruction_phct\model\PSNR_BEST_G.pth')

    # # ----------------------------------------
    # # [Test SSR-SIM Model > DataSet Preparation]
    # # ----------------------------------------
    # # generate_noisy_raw_data_for_test(r'K:\BioSR', r'K:\BioSR_Test_Data', 'CCPs')
    # # generate_noisy_raw_data_for_test(r'K:\BioSR', r'K:\BioSR_Test_Data', 'F-actin')
    # generate_noisy_raw_data_for_test(r'K:\BioSR', r'K:\BioSR_Test_Data', 'Microtubules')

    # # ----------------------------------------
    # # [Test SSR-SIM Model -> Estimate Parameter]
    # # ----------------------------------------
    # # performing_SIEsti(r'K:\BioSR_Test_Data\CCPs', 'CCPs')
    # # performing_SIEsti(r'K:\BioSR_Test_Data\F-actin', 'F-actin')
    # performing_SIEsti(r'K:\BioSR_Test_Data\Microtubules', 'Microtubules')

    # # ----------------------------------------
    # # [Test SSR-SIM Model -> Model Forward]
    # # ----------------------------------------
    # # ssrsirecon(r'K:\BioSR_Training_Data\CCPs_hsnr_DST\self-supervised-val_reconstruction_phct_pl\model\PSNR_BEST_G.pth', r'K:\BioSR_Test_Data\CCPs')
    # # ssrsirecon(r'K:\BioSR_Training_Data\CCPs_lsnr_DST\self-supervised-val_reconstruction_phct_pl\model\PSNR_BEST_G.pth', r'K:\BioSR_Test_Data\CCPs')
    # # ssrsirecon(r'K:\BioSR_Training_Data\F-actin_hsnr_DST\self-supervised-val_reconstruction_phct_pl\model\PSNR_BEST_G.pth', r'K:\BioSR_Test_Data\F-actin')
    # # ssrsirecon(r'K:\BioSR_Training_Data\F-actin_lsnr_DST\self-supervised-val_reconstruction_phct_pl\model\PSNR_BEST_G.pth', r'K:\BioSR_Test_Data\F-actin')
    # # ssrsirecon(r'K:\BioSR_Training_Data\Microtubules_hsnr_DST\self-supervised-val_reconstruction_phct_pl\model\PSNR_BEST_G.pth', r'K:\BioSR_Test_Data\Microtubules')
    # ssrsirecon(r'K:\BioSR_Training_Data\Microtubules_lsnr_DST\self-supervised-val_reconstruction_phct_pl\model\PSNR_BEST_G.pth', r'K:\BioSR_Test_Data\Microtubules')

