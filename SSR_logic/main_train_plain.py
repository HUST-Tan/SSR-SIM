import numpy as np
import platform
import argparse
import torch
import random
import math
import os
import time
import sys
import threading

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from torch.utils.data import DataLoader

sys.path.append(os.path.split(os.getcwd())[0])

from utils.image_process import calculate_index
from utils.option import AI_parse, dict2str, sir_parse2dict
from utils.tools import mkdir, WriteOutputTxt, set_seed, get_timestamp
from utils.mrc import write_mrc_image
from SSR_data.select_dataset import define_Dataset
from SSR_model.select_model import define_Model


def main(json_path, opt=None):
    # ----------------------------------------
    #            <process opt and seed>
    # ----------------------------------------
    assert (json_path or opt) and (not json_path or not opt)
    if opt is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
        opt = AI_parse(parser.parse_args().opt, is_train=True)
    outinfo = WriteOutputTxt(os.path.join(opt['save_path'], 'config.txt'))
    outinfo.info(dict2str(opt) + '\n')
    opt['outinfo'] = outinfo
    current_step = 0
    seed = random.randint(1, 10000)
    set_seed(seed)
    outinfo.info('Random seed: {}\n'.format(seed))
    if_val_need_gt = opt['val_need_gt']
    if opt['supervise'] in ["self-supervised"]:
        opt['save_val_images'] = True
        opt['save_all_val_images'] = True

    # ----------------------------------------
    #            <creat dataloader>
    # ----------------------------------------
    train_set = define_Dataset(opt, 'train')
    outinfo.info('Training Dataset [{:s}] is created according to model [{}] and dataloader [{}]\n'.format(train_set.__class__.__name__, opt['model'], opt['data']))
    train_size = int(math.ceil(len(train_set) / opt['dataloader_batch_size']))
    outinfo.info('Number of train images: {:,d}, iters: {:,d}\n'.format(len(train_set), train_size))
    train_loader = DataLoader(train_set, batch_size=opt['dataloader_batch_size'], num_workers=opt['dataloader_num_workers'],
                              shuffle=True, drop_last=True, pin_memory=False)
    val_loader = DataLoader(define_Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=False)
    outinfo.info('DataLoader completed: {}\n'.format(seed))

    # ----------------------------------------
    #            <initialize model>
    # ----------------------------------------
    model = define_Model(opt)
    outinfo.info('Training model [{:s}] is created according to model [{}] and supervise manner [{}]\n'.format(model.__class__.__name__, opt['model'], opt['supervise']))
    model.init_train()

    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    outinfo.info('\nmain training\n')
    if opt['G_scheduler_type'] == 'MultiStepLR':
        opt['epoch_num'] = math.ceil(1 + (opt['G_scheduler_milestones'][-1] - current_step) / (len(train_set) // opt['dataloader_batch_size']))
    elif opt['G_scheduler_type'] == 'CosineAnnealingLR':
        opt['epoch_num'] = math.ceil(1 + (opt['G_scheduler_IterNum'] - current_step) / (len(train_set) // opt['dataloader_batch_size']))
    else:
        raise RuntimeError

    G_loss_idx, G_loss_list, G_total_norm_idx, G_total_norm_list, G_lr_idx, G_lr_list = [], [], [], [], [], []  # 用于画图
    psnr_ssim_idx = []
    avg_sim_psnr_list, avg_sim_nrmse_list, avg_sim_ssim_list, avg_sim_msssim_list = [], [], [], []
    all_sim_psnr_list, all_sim_nrmse_list, all_sim_ssim_list, all_sim_msssim_list = [], [], [], []
    report_total_sim_psnr, report_total_sim_nrmse, report_total_sim_ssim, report_total_sim_msssim = 0.0, None, 0.0, None
    sim_psnr_each_best, sim_ssim_each_best = [], []
    for epoch in range(opt['epoch_num']):
        for i, train_data in enumerate(train_loader):

            if opt['iter_sleep'] is not None:
                time.sleep(opt['iter_sleep'])

            # -------------------------------
            # 1) update learning rate
            # 2) feed patch pairs
            # 3) optimize parameters
            # -------------------------------
            model.feed_data_train(train_data)
            model.optimize_parameters(current_step)
            current_step += 1

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['num_iter_interval_print'] == 0:
                logs = model.current_log()  # such as loss
                current_lr = model.current_learning_rate()
                message = '{}  <epoch:{:3d}, iter:{:8,d}, G_lr:{:.3e}> '.format(get_timestamp(), epoch, current_step, current_lr['G'])
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                outinfo.info(message + '\n')
                G_total_norm_idx.append(current_step)
                G_total_norm_list.append(logs['G_total_norm'])
                G_loss_list.append(logs['G_loss'])
                G_loss_idx.append(current_step)
                G_lr_list.append(current_lr['G'])
                G_lr_idx.append(current_step)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['num_iter_interval_save'] == 0:
                outinfo.info('Saving the model, at iter {}\n'.format(current_step))
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step == 7 or current_step % opt['num_iter_interval_val'] == 0:

                wf_psnr_list, wf_ssim_list, wf_msssim_list, wf_nrmse_list = [], [], [], []
                raw_psnr_list, raw_ssim_list, raw_msssim_list, raw_nrmse_list = [], [], [], []
                sim_psnr_list, sim_ssim_list, sim_msssim_list, sim_nrmse_list = [], [], [], []

                idx = 0

                for test_data in val_loader:

                    idx += 1
                    image_name_ext = os.path.basename(test_data['input_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(os.path.join(opt['save_path'], 'image'), img_name)
                    mkdir(img_dir)

                    model.feed_data_val(test_data)
                    model.val()

                    if current_step == 7: break

                    need_input = True if current_step == opt['num_iter_interval_val'] else False
                    need_target = True if if_val_need_gt else False
                    visuals = model.current_visuals(need_input=need_input, need_target=need_target)

                    def check_and_write(key, save_append=''):
                        save_label = key[:-5] if key.find('_data') >= 0 else key
                        if len(save_append) >= 1:
                            save_label += '_' + save_append
                        if visuals[key] is not None:
                            if key.find('wf') >= 0:
                                sampling_rate = visuals['wf_sampling_rate']
                            elif key.find('raw') >= 0:
                                sampling_rate = visuals['raw_sampling_rate']
                            elif key.find('sim') >= 0:
                                sampling_rate = visuals['sim_sampling_rate']
                            write_mrc_image(visuals[key], os.path.join(img_dir, '{:s}_{}.mrc'.format(img_name, save_label)), sampling_rate=sampling_rate)

                    if opt['save_val_images'] is None or opt['save_val_images'] is True:
                        if current_step == opt['num_iter_interval_val']:
                            check_and_write('sim_input_data')
                            check_and_write('raw_input_data')
                            check_and_write('wf_input_data')
                            check_and_write('sim_target_data')
                            check_and_write('raw_target_data')
                            check_and_write('wf_target_data')

                        if opt['save_all_val_images'] is True:
                            check_and_write('wf_infer_data', '{:0>6}'.format(current_step))
                            check_and_write('raw_infer_data', '{:0>6}'.format(current_step))
                            check_and_write('sim_infer_data', '{:0>6}'.format(current_step))
                        else:
                            check_and_write('wf_infer_data', 'latest')
                            check_and_write('raw_infer_data', 'latest')
                            check_and_write('sim_infer_data', 'latest')

                    # -----------------------
                    # calculate PSNR, SSIM, MSSSIM, NRMSE, DECORR
                    # -----------------------
                    if if_val_need_gt:

                        border = 16 if opt['val_border'] is None else opt['val_border']

                        current_sim_nrmse, current_sim_psnr, current_sim_ssim, current_sim_msssim = calculate_index(visuals['sim_target_data'], visuals['sim_infer_data'], border=border * 2)

                        outinfo.info('SIM -> {:>10s} | psnr:{:<5.2f}dB | nrmse:{:<5.4f} | ssim:{:<5.4f} | msssim:{:<5.4f}\n'.format(image_name_ext, current_sim_psnr, current_sim_nrmse, current_sim_ssim, current_sim_msssim))

                        sim_psnr_list.append(current_sim_psnr)
                        sim_ssim_list.append(current_sim_ssim)
                        sim_msssim_list.append(current_sim_msssim)
                        sim_nrmse_list.append(current_sim_nrmse)

                        if current_step == opt['num_iter_interval_val']:
                            sim_psnr_each_best.append(current_sim_psnr)
                            sim_ssim_each_best.append(current_sim_ssim)
                        else:
                            if 0.005 + sim_psnr_each_best[idx - 1] < current_sim_psnr:
                                sim_psnr_each_best[idx - 1] = current_sim_psnr
                                if opt['save_val_images'] is None or opt['save_val_images'] is True:
                                    check_and_write('raw_infer_data', 'bestpsnr')
                                    check_and_write('sim_infer_data', 'bestpsnr')
                            if 0.0005 + sim_ssim_each_best[idx - 1] < current_sim_ssim:
                                sim_ssim_each_best[idx - 1] = current_sim_ssim
                                if opt['save_val_images'] is None or opt['save_val_images'] is True:
                                    check_and_write('raw_infer_data', 'bestssim')
                                    check_and_write('sim_infer_data', 'bestssim')

                if current_step == 7: continue

                if if_val_need_gt:

                    all_sim_psnr_list.append(sim_psnr_list)
                    all_sim_ssim_list.append(sim_ssim_list)
                    all_sim_msssim_list.append(sim_msssim_list)
                    all_sim_nrmse_list.append(sim_nrmse_list)
                    avg_sim_psnr, avg_sim_ssim, avg_sim_msssim, avg_sim_nrmse = np.mean(sim_psnr_list), np.mean(sim_ssim_list), np.mean(sim_msssim_list), np.mean(sim_nrmse_list)
                    avg_sim_psnr_list.append(avg_sim_psnr)
                    avg_sim_ssim_list.append(avg_sim_ssim)
                    avg_sim_msssim_list.append(avg_sim_msssim)
                    avg_sim_nrmse_list.append(avg_sim_nrmse)
                    outinfo.info('SIM --> avg_psnr:{:<5.2f}dB, avg_nrmse:{:<5.4f}, avg_ssim:{:<5.4f}, avg_msssim:{:<5.4f}\n'.format(avg_sim_psnr, avg_sim_nrmse, avg_sim_ssim, avg_sim_msssim))

                    psnr_ssim_idx.append(current_step)

                    if report_total_sim_psnr + 0.005 < avg_sim_psnr:
                        report_total_sim_psnr = avg_sim_psnr
                        model.save('PSNR_BEST')
                        outinfo.info('save psnr-best model | avg_psnr:{:<5.2f}dB | {:0>6}\n'.format(avg_sim_psnr, current_step))
                    if report_total_sim_ssim + 0.0005 < avg_sim_ssim:
                        report_total_sim_ssim = avg_sim_ssim
                        model.save('SSIM_BEST')
                        outinfo.info('save ssim-best model | avg_ssim:{:<5.4f} | {:0>6}\n'.format(avg_sim_ssim, current_step))

                # plot
                fig = plt.figure()
                # G - lr
                if len(G_lr_list) > 0:
                    plt.subplot(221)
                    plt.plot(G_lr_idx, G_lr_list, color='black', marker='o')
                    plt.title('G_lr')
                # G - loss
                if len(G_loss_list) > 0:
                    plt.subplot(222)
                    plt.plot(G_loss_idx, G_loss_list, color='black', marker='o')
                    plt.title('G_loss')
                # SIM - PSNR
                if len(avg_sim_psnr_list) > 0:
                    plt.subplot(223)
                    plt.plot(psnr_ssim_idx, avg_sim_psnr_list, color='black', marker='o')
                    plt.title('val_PSNR')
                # SIM - SSIM
                if len(avg_sim_ssim_list) > 0:
                    plt.subplot(224)
                    plt.plot(psnr_ssim_idx, avg_sim_ssim_list, color='black', marker='o')
                    plt.title('val_SSIM')
                # savefig
                fig.set_size_inches(20, 7)
                try:
                    fig.savefig(os.path.join(opt['save_path'], 'plot.png'), format='png', transparent=True, dpi=300, pad_inches=0)
                except OSError:  # builtins error
                    pass
                plt.close()

    outinfo.info('Saving the final model\n')
    model.save('latest')
    if if_val_need_gt and len(avg_sim_psnr_list) >= 1:
        avg_sim_psnr_list = np.array(avg_sim_psnr_list)
        avg_sim_ssim_list = np.array(avg_sim_ssim_list)
        avg_sim_msssim_list = np.array(avg_sim_msssim_list)
        avg_sim_nrmse_list = np.array(avg_sim_nrmse_list)
        outinfo.info('BEST SIM  PSNR :{:<5.2f}dB at iter {:0>6}\n'.format(avg_sim_psnr_list.max(), opt['num_iter_interval_val'] * avg_sim_psnr_list.argmax()))
        outinfo.info('BEST SIM  SSIM :{:<5.4f} at iter {:0>6}\n'.format(avg_sim_ssim_list.max(), opt['num_iter_interval_val'] * avg_sim_ssim_list.argmax()))
        outinfo.info('BEST SIM  NRMSE:{:<5.4f} at iter {:0>6}\n'.format(avg_sim_nrmse_list.min(), opt['num_iter_interval_val'] * avg_sim_nrmse_list.argmin()))
        outinfo.info('BEST SIM MSSSIM:{:<5.4f} at iter {:0>6}\n'.format(avg_sim_msssim_list.max(), opt['num_iter_interval_val'] * avg_sim_msssim_list.argmax()))

    mkdir(os.path.join(opt['save_path'], 'npyfiles'))
    if len(G_loss_list) >= 1:
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'G_loss_idx.npy'), G_loss_idx)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'G_loss_list.npy'), G_loss_list)
    if len(psnr_ssim_idx) >= 1:
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'psnr_ssim_idx.npy'), psnr_ssim_idx)
    if len(avg_sim_psnr_list) >= 1:
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'avg_sim_psnr_list.npy'), avg_sim_psnr_list)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'avg_sim_nrmse_list.npy'), avg_sim_nrmse_list)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'avg_sim_ssim_list.npy'), avg_sim_ssim_list)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'avg_sim_msssim_list.npy'), avg_sim_msssim_list)
    if len(all_sim_psnr_list) >= 1:
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'all_sim_psnr_list.npy'), all_sim_psnr_list)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'all_sim_nrmse_list.npy'), all_sim_nrmse_list)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'all_sim_ssim_list.npy'), all_sim_ssim_list)
        np.save(os.path.join(opt['save_path'], 'npyfiles', 'all_sim_msssim_list.npy'), all_sim_msssim_list)

    outinfo.info('End of training\n')