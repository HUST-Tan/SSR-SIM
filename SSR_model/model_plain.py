import torch
from torch.optim import lr_scheduler, Adam, AdamW
from torch.nn.utils import clip_grad_norm_

from SSR_model.select_network import define_G
from SSR_model.model_base import ModelBase
from SSR_model.loss import GP_loss_fun
from utils.option import sir_parse2dict, dict2nonedict

device = torch.device('cuda')


class ModelPlain(ModelBase):
    # ------------------------------------
    # define network and optimizer
    # ------------------------------------
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        self.netG = define_G(opt).to(self.device)
        # self.netG = DataParallel(self.netG)
        self.G_lossfn = None
        self.G_optimizer = None

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = {}  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['pretrained_netG']
        if load_path_G is not None:
            self.opt['outinfo'].info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # define G_loss and D_loss
    # ----------------------------------------
    def define_loss(self):
        self.G_lossfn = GP_loss_fun(self.opt['G_lossfn_type'])

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                self.opt['outinfo'].info('Params [{:s}] will not optimize.'.format(k))
        weight_decay = self.opt['G_optimizer_weight_decay'] if self.opt['G_optimizer_weight_decay'] is not None else 0.
        if self.opt['G_optimizer_type'].lower() == 'adam':
            assert weight_decay == 0, "Incorrect implementation of Weight Decay in ADAM optimizer"
            self.G_optimizer = Adam(G_optim_params, lr=self.opt['G_optimizer_lr'], weight_decay=weight_decay)
        elif self.opt['G_optimizer_type'].lower() == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt['G_optimizer_lr'], weight_decay=weight_decay)
        else:
            raise RuntimeError


    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers['G'] = lr_scheduler.MultiStepLR(self.G_optimizer, self.opt['G_scheduler_milestones'], self.opt['G_scheduler_gamma'])
        elif self.opt['G_scheduler_type'] == 'CosineAnnealingLR':
            self.schedulers['G'] = lr_scheduler.CosineAnnealingLR(self.G_optimizer, T_max=self.opt['G_scheduler_IterNum'], eta_min=self.opt['G_scheduler_MinLR'])
        else:
            raise RuntimeError

    # ----------------------------------------
    # feed training in/out data
    # ----------------------------------------
    def feed_data_train(self, data):
        opt = self.opt
        if opt['need_para']: self.para_data = data['para'].to(self.device)
        self.json_path = data['json_path']

        self.raw_input_data = data['train_RAW_LSNR_1'].to(self.device)
        if opt['supervise'] in ["full-supervised"]:
            self.sim_target_data = data['train_SIM_HSNR'].to(self.device)
        elif opt['supervise'] in ["self-supervised-val", "self-supervised"]:
            self.sim_target_data = data['train_SIM_LSNR_2'].to(self.device)
        else:
            raise NotImplementedError

    def feed_data_val(self, data):
        opt = self.opt
        if opt['need_para']: self.para_data = data['para'].to(self.device)
        self.json_path = data['json_path']

        self.raw_input_data = data['val_RAW_LSNR_1'].to(self.device)
        if opt['supervise'] in ["full-supervised", "self-supervised-val"]:
            self.sim_target_data = data['val_SIM_HSNR'].to(self.device)
        elif opt['supervise'] in ["self-supervised"]:
            pass
        else:
            raise NotImplementedError


    def cal_netG_loss(self):
        opt = self.opt
        # ----------------------------------------
        # netG
        # ----------------------------------------
        result = self.netG(self.raw_input_data, self.para_data, self.json_path)

        # ----------------------------------------
        # calculate loss
        # ----------------------------------------
        G_loss, self.sim_infer_data = self.cal_loss(result, self.sim_target_data, self.G_lossfn)

        return G_loss

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        opt = self.opt

        # ----------------------------------------
        # clean grad
        # ----------------------------------------
        self.G_optimizer.zero_grad()
        # ----------------------------------------
        # forward
        # ----------------------------------------
        G_loss = self.cal_netG_loss()
        # ----------------------------------------
        # back propagation
        # ----------------------------------------
        G_loss.backward()
        if opt['G_loss_grad_max_norm'] is not None and opt['G_loss_grad_max_norm'] > 0.:
            total_norm = clip_grad_norm_(self.netG.parameters(), max_norm=opt['G_loss_grad_max_norm'], norm_type=2)
            self.log_dict['G_total_norm'] = total_norm
            if total_norm > opt['G_loss_grad_max_norm']:
                opt['outinfo'].info("total_norm {:e} more than max_norm {:e} with G_loss {:e}\n".format(total_norm.cpu().numpy(), opt['G_loss_grad_max_norm'], G_loss.item()))
        else:
            self.log_dict['G_total_norm'] = -1
        # ----------------------------------------
        # do optimizition
        # ----------------------------------------
        self.G_optimizer.step()
        self.log_dict['G_loss'] = G_loss.item()
        self.update_learning_rate('G')

    # ----------------------------------------
    # val / inference
    # ----------------------------------------
    @staticmethod
    def plain_inference(model, *data):
        with torch.no_grad():
            result = model(*data)
        return result

    def val(self):
        # opt = self.opt
        self.netG.eval()
        # for _, v in self.netG.named_parameters():
        #     v.requires_grad = False

        with torch.no_grad():
            result = self.plain_inference(self.netG, self.raw_input_data, self.para_data, self.json_path)

            self.sim_infer_data = result

        # for _, v in self.netG.named_parameters():
        #     v.requires_grad = True
        self.netG.train()

    # ----------------------------------------
    # get batch results (first slice in batch):
    # ----------------------------------------
    def current_visuals(self, need_input=True, need_target=True):
        out_dict = dict2nonedict({})
        opt = self.opt
        raw_shape = opt['raw_shape_OPC']
        sim_shape = opt['sim_shape_OPC']

        json_opt = sir_parse2dict(self.json_path[0])
        out_dict['wf_sampling_rate'] = [json_opt['width_space_sampling'], json_opt['height_space_sampling'], json_opt['depth_space_sampling']]
        out_dict['raw_sampling_rate'] = [json_opt['width_space_sampling'], json_opt['height_space_sampling'], json_opt['depth_space_sampling']]
        if opt['sim_scale'] is not None:
            out_dict['sim_sampling_rate'] = [json_opt['width_space_sampling'] / opt['sim_scale'], json_opt['height_space_sampling'] / opt['sim_scale'], json_opt['depth_space_sampling']]

        if need_input:
            out_dict['sim_input_data'] = self.tensor2format(self.conv_rec(self.raw_input_data.detach()[0], self.para_data.detach()[0], raw_shape, json_opt), sim_shape)
            out_dict['raw_input_data'] = self.tensor2format(self.raw_input_data.detach()[0], raw_shape)

        if need_target:
            # out_dict['sim_target_data'] = self.tensor2format(self.conv_rec(self.raw_target_data.detach()[0], self.para_data.detach()[0], raw_shape, json_class), sim_shape)
            out_dict['sim_target_data'] = self.tensor2format(self.sim_target_data.detach()[0], sim_shape)
        out_dict['sim_infer_data'] = self.tensor2format(self.sim_infer_data.detach()[0], sim_shape)

        return out_dict
