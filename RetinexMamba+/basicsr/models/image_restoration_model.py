import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import os
import random

import torch.nn.functional as F
from functools import partial

import glob

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch.nn.parallel import DataParallel, DistributedDataParallel
class PhysicalAugmentation(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # === 1. 真实相机元数据 (Real Camera Metadata) ===
        self.camera_ccms = torch.tensor([
            [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, -0.1223, 1.1926]],  # Canon 5D
            [[0.4913, 0.5064, -0.0023], [-0.0911, 0.8753, 0.2159], [-0.0063, 0.1309, 0.8754]],  # Nikon D700
            [[0.8380, -0.2630, -0.0639], [-0.2887, 1.0725, 0.3249], [-0.0627, 0.1427, 0.5438]],  # Canon EOS 40D
            [[0.730, -0.198, -0.021], [-0.208, 0.994, 0.138], [-0.038, 0.119, 0.697]],  # Sony Alpha 7
        ], device=self.device).float()


    def sample_params(self, B):
        # 1. 随机选择 CCM 索引
        ccm_indices = torch.randint(0, len(self.camera_ccms), (B,), device=self.device)
        batch_ccms = self.camera_ccms[ccm_indices]

        # 2. 随机生成白平衡增益
        red_gains = torch.empty(B, 1, 1, 1, device=self.device).uniform_(1.9, 2.4)
        blue_gains = torch.empty(B, 1, 1, 1, device=self.device).uniform_(1.5, 1.9)
        wb_gains = torch.cat([red_gains, torch.ones_like(red_gains), blue_gains], dim=1)

        # 3. 随机噪声参数
        shot_k = torch.exp(torch.empty(B, device=self.device).uniform_(np.log(0.001), np.log(0.01)))
        read_s = torch.exp(torch.empty(B, device=self.device).uniform_(np.log(0.0001), np.log(0.005)))

        # 【关键修改】调整维度为 (B, 1) 以便后续 cat/stack
        shot_k = shot_k.view(B, 1)
        read_s = read_s.view(B, 1)

        # 【关键修改】返回字典，匹配 optimize_parameters 中的调用
        return {
            'ccm': batch_ccms,
            'wb_gains': wb_gains,
            'shot_k': shot_k,
            'read_s': read_s
        }

    # =============================================================
    # === Part A: Inverse ISP (sRGB -> Linear RAW) ===
    # =============================================================
    def inverse_isp(self, img, batch_ccms, wb_gains):
        """
        Ref: Brooks et al. "Unprocessing Images for Learned Raw Denoising"
        """
        B, C, H, W = img.shape

        # 1. Inverse Tone Mapping (Gamma Expansion)
        linear_srgb = torch.pow(img.clamp(1e-8, 1.0), 2.2)

        # 2. Inverse CCM (sRGB -> Camera RGB)
        batch_inv_ccms = torch.inverse(batch_ccms)  # (B, 3, 3)

        # Reshape for matmul: (B, H*W, 3)
        img_flat = linear_srgb.permute(0, 2, 3, 1).reshape(B, -1, 3)
        cam_rgb_flat = torch.bmm(img_flat, batch_inv_ccms.transpose(1, 2))
        cam_rgb = cam_rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # 3. Inverse White Balance
        raw_rgb = cam_rgb / (wb_gains + 1e-8)

        return torch.clamp(raw_rgb, 0.0, 1.0)

    # =============================================================
    # === Part B: Mosaic & Noise (The Attack Surface) ===
    # =============================================================
    def mosaic(self, rgb):
        """Extracts RGGB Bayer pattern from RGB image."""
        B, C, H, W = rgb.shape
        r = rgb[:, 0, 0::2, 0::2]
        g1 = rgb[:, 1, 0::2, 1::2]
        g2 = rgb[:, 1, 1::2, 0::2]
        b = rgb[:, 2, 1::2, 1::2]
        # Stack into 4 channels to simulate packed RAW (H/2, W/2, 4) or keep single channel
        # Here we return a single channel mosaic roughly simulating a sensor
        bayer = torch.zeros(B, 1, H, W, device=rgb.device)
        bayer[:, 0, 0::2, 0::2] = r
        bayer[:, 0, 0::2, 1::2] = g1
        bayer[:, 0, 1::2, 0::2] = g2
        bayer[:, 0, 1::2, 1::2] = b
        return bayer

    def apply_noise(self, bayer, shot_k, read_s):
        """
        Ref: Foi et al. TIP 2008.
        Model: y = x + n, where Var(n) = a*x + b
        """
        B = bayer.shape[0]
        # reshape
        k = shot_k.view(B, 1, 1, 1)
        s = read_s.view(B, 1, 1, 1)

        # (Poisson) + (Gaussian Read Noise)
        variance = k * bayer.clamp(min=0) + s ** 2

        # Reparameterization Trick: 保持梯度流向 variance -> k, s
        noise = torch.randn_like(bayer) * torch.sqrt(variance + 1e-10)

        noisy_bayer = bayer + noise
        return noisy_bayer, variance

    # =============================================================
    # === Part C: Forward ISP (Reprocessing) ===
    # =============================================================
    def demosaic(self, bayer):
        """
        Bilinear Demosaicing.
        """
        # 将 Bayer 拆分为 R, G1, G2, B 四个通道的图 (H/2, W/2)
        r = bayer[:, :, 0::2, 0::2]
        g1 = bayer[:, :, 0::2, 1::2]
        g2 = bayer[:, :, 1::2, 0::2]
        b = bayer[:, :, 1::2, 1::2]

        # Average Green
        g_avg = (g1 + g2) / 2

        # Upsample back to H, W
        scale = 2
        r_up = F.interpolate(r, scale_factor=scale, mode='bilinear', align_corners=False)
        g_up = F.interpolate(g_avg, scale_factor=scale, mode='bilinear', align_corners=False)
        b_up = F.interpolate(b, scale_factor=scale, mode='bilinear', align_corners=False)

        return torch.cat([r_up, g_up, b_up], dim=1)

    def forward_isp(self, raw_rgb, batch_ccms, wb_gains):
        """
        Re-applies ISP steps to go from Noisy RAW -> Noisy sRGB.
        """
        B, C, H, W = raw_rgb.shape

        # 1. Forward White Balance
        wb_rgb = raw_rgb * wb_gains

        # 2. Forward CCM (Camera RGB -> sRGB)
        img_flat = wb_rgb.permute(0, 2, 3, 1).reshape(B, -1, 3)
        # Use transpose of CCM for multiplication
        linear_srgb_flat = torch.bmm(img_flat, batch_ccms.transpose(1, 2))
        linear_srgb = linear_srgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # 3. Tone Mapping (Gamma Compression)
        srgb = torch.pow(linear_srgb.clamp(1e-8, 1.0), 1.0 / 2.2)

        return torch.clamp(srgb, 0, 1)

    def forward(self, gt_img, attack_params=None):

        B = gt_img.shape[0]

        # 1. 获取物理参数
        sample_dict = self.sample_params(B)
        batch_ccms, wb_gains, rand_shot, rand_read = sample_dict['ccm'], sample_dict['wb_gains'], sample_dict['shot_k'], sample_dict['read_s']

        if attack_params is not None:
            # === ATTACK INJECTION ===
            shot_k = attack_params['shot_k']
            read_s = attack_params['read_s']
        else:
            shot_k = rand_shot
            read_s = rand_read

        # 2. Unprocessing (sRGB -> Linear RAW)
        clean_raw = self.inverse_isp(gt_img, batch_ccms, wb_gains)
        # 3. Mosaicing (Linear RAW -> Bayer)
        clean_bayer = self.mosaic(clean_raw)
        # 4. Add Physics-based Noise
        noisy_bayer, noise_map = self.apply_noise(clean_bayer, shot_k, read_s)
        # 5. Demosaicing (Noisy Bayer -> Noisy Linear RAW)
        noisy_raw = self.demosaic(noisy_bayer)
        # 6. Reprocessing (Noisy Linear RAW -> Noisy sRGB)
        degraded_img = self.forward_isp(noisy_raw, batch_ccms, wb_gains)
        # 7. 构建 GT 参数 (用于训练 DA-MoE 的预测器)
        params_gt = torch.stack([shot_k, read_s], dim=1)

        return degraded_img, params_gt

class VGG19_Extractor(nn.Module):
    def __init__(self, layer_indices=None):

        super(VGG19_Extractor, self).__init__()

        vgg = models.vgg19(pretrained=True)

        self.vgg_features = vgg.features

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        if layer_indices is None:
            self.layer_indices = [1, 6, 11, 20, 29]
        else:
            self.layer_indices = layer_indices

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):

        x = (x - self.mean) / self.std

        features = []

        for name, layer in self.vgg_features._modules.items():
            x = layer(x)
            if int(name) in self.layer_indices:
                features.append(x)

            if int(name) >= max(self.layer_indices):
                break

        return features

class ContrastiveLossMargin(nn.Module):
    def __init__(self, weight=0.1, margin_scale=0.1):
        super(ContrastiveLossMargin, self).__init__()
        self.vgg = VGG19_Extractor().cuda()
        self.l1 = nn.L1Loss()
        self.weight = weight
        self.margin_scale = margin_scale

    def forward(self, anchor, positive, negative, noise_params):
        """
        anchor:   增强网络的输出
        positive: GT
        negative: 攻击样本 (var_L_perturb)
        noise_params: [B, 2] (Shot, Read)
        """
        noise_mag = torch.norm(noise_params, p=2, dim=1, keepdim=True)

        dynamic_margin = noise_mag * self.margin_scale

        with torch.no_grad():
            feat_p = self.vgg(positive.detach())
            feat_n = self.vgg(negative.detach())
        feat_a = self.vgg(anchor)

        loss = 0

        for fa, fp, fn in zip(feat_a, feat_p, feat_n):
            d_ap = torch.mean(torch.abs(fa - fp), dim=[1, 2, 3])
            d_an = torch.mean(torch.abs(fa - fn), dim=[1, 2, 3])

            current_margin = dynamic_margin.view(-1)

            d_an_clipped = F.relu(d_an - current_margin)

            # CR Loss
            loss += d_ap / (d_an_clipped + 1e-7)

        return torch.mean(loss) * self.weight

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define mixed precision
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')
                  
        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get(
                'mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get(
                'use_identity', False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.pgds = PhysicalAugmentation(device=self.device)
        self.cri_param = nn.MSELoss().to(self.device)
        self.cri_noise_vgg_margin_new = ContrastiveLossMargin().to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)  #根据pop出来的loss_type找到对应的loss函数
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)      #如何写 weighted loss 呢？传参构造Loss函数
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        with torch.no_grad():
            attack_params = self.pgds.sample_params(self.gt.shape[0])
            # 生成负样本
            img_neg, _ = self.pgds(self.gt, attack_params)
            # 构造 GT 标签
            p_shot_gt = attack_params['shot_k']
            p_read_gt = attack_params['read_s']
            params_gt = torch.cat([p_shot_gt, p_read_gt], dim=1).to(self.device)
            margin = (params_gt.sum(dim=1, keepdim=True) * 0.1).view(-1, 1, 1, 1)

        # 预测器前向 (负样本)
        if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
            params_pred_neg = self.net_g.module.noise_predictor(img_neg)
        else:
            params_pred_neg = self.net_g.noise_predictor(img_neg)

        l_pred_sup = self.cri_param(params_pred_neg, params_gt) * 10.0

        with autocast(enabled=self.use_amp):
            preds, self.params_pred_real = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]

            #== == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
            # Phase 3: Consistency (基于 PGDS 的一致性扰动)
            # ===========================================================
            # 1. 构造微小的扰动参数
            bs = self.lq.shape[0]
            # 构造 params 字典，格式需与 pgds.sample_params 返回的一致
            perturb_params = {}
            # Shot noise 增量设为 0 (或者极小值)
            perturb_params['shot_k'] = torch.zeros(bs, 1, device=self.device)
            # Read noise 增量: 随机小高斯 [0.01, 0.05]
            sigma_add = (torch.rand(bs, 1, device=self.device) * 0.04 + 0.01)
            perturb_params['read_s'] = sigma_add

            # 2. 调用 PGDS 进行同源加噪
            # 输入: 真实低光图 var_L (作为 Clean Base)
            # 操作: var_L -> Unprocess -> Add 'perturb_params' -> Reprocess -> var_L_perturb
            with torch.no_grad():
                var_L_perturb, _ = self.pgds(self.lq, perturb_params)

            # 3. 再次预测 
            if isinstance(self.net_g, (DataParallel, DistributedDataParallel)):
                params_pred_perturb = self.net_g.module.noise_predictor(var_L_perturb)
            else:
                params_pred_perturb = self.net_g.noise_predictor(var_L_perturb)

            # 4. 构建物理约束目标
            sigma_s_real = self.params_pred_real[:, 0:1]
            sigma_r_real = self.params_pred_real[:, 1:2]
            # 目标:
            # Shot noise: 理论上不应改变 (因为只加了 Read noise)
            target_s = sigma_s_real.detach()
            # Read noise: 遵循方差叠加
            target_r = torch.sqrt(sigma_r_real.pow(2) + sigma_add.pow(2)).detach()
            target_params_perturb = torch.cat([target_s, target_r], dim=1)
            # Loss: 一致性
            l_consist = self.cri_param(params_pred_perturb, target_params_perturb) * 5.0
            l_contrast = self.cri_noise_vgg_margin_new(preds[-1], self.gt, img_neg, params_gt)
            # if current_iter==0:

            self.output = preds[-1]

            loss_dict = OrderedDict()
            # pixel loss
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt) #此处统计batch的loss

            l_pix = l_pix +  l_pred_sup + l_consist + l_contrast

            loss_dict['l_pix'] = l_pix
            loss_dict['l_pred'] = l_pred_sup
            loss_dict['l_cons'] = l_consist  # Log consistency loss
            loss_dict['l_cont'] = l_contrast

        self.amp_scaler.scale(l_pix).backward()
        self.amp_scaler.unscale_(self.optimizer_g) 
        # l_pix.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        # self.optimizer_g.step()
        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred,_ = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred,_ = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:

                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, **kwargs):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter, **kwargs)

    def save_best(self, best_metric, param_key='params'):
        psnr = best_metric['psnr']
        cur_iter = best_metric['iter']
        save_filename = f'best_psnr_{psnr:.2f}_{cur_iter}.pth'
        exp_root = self.opt['path']['experiments_root']
        save_path = os.path.join(
            self.opt['path']['experiments_root'], save_filename)

        if not os.path.exists(save_path):
            for r_file in glob.glob(f'{exp_root}/best_*'):
                os.remove(r_file)
            net = self.net_g

            net = net if isinstance(net, list) else [net]
            param_key = param_key if isinstance(
                param_key, list) else [param_key]
            assert len(net) == len(
                param_key), 'The lengths of net and param_key should be the same.'

            save_dict = {}
            for net_, param_key_ in zip(net, param_key):
                net_ = self.get_bare_model(net_)
                state_dict = net_.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    state_dict[key] = param.cpu()
                save_dict[param_key_] = state_dict

            torch.save(save_dict, save_path)
