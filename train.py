"""
Adapted training script for SynDiff on BraTS2020.

Changes from original:
- Single GPU (no DDP) for simplified training
- MONAI-based data loading with nifti files
- TensorBoard logging for epoch loss and learning rate
- Model saving to results/task_{timestamp}/models/
- 4-modality support with random source-target pair sampling
- 200 epochs default training

Original SynDiff copyright: ICON Lab 2023
Adaptation for unified experiment framework.
"""

import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import shutil
import time
import datetime

from dataset import CreateDatasetSynthesis

# ============== Diffusion Coefficients (from original SynDiff) ==============


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """Diffuse the data (t == 0 means diffused for t step)"""
    if noise is None:
        noise = torch.randn_like(x_start)
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    return x_t


def q_sample_pairs(coeff, x_start, t):
    """Generate a pair of disturbed images for training"""
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    return x_t, x_t_plus_one


# ============== Posterior Sampling (from original SynDiff) ==============


class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:, [0], :]
    source = x_init[:, [1], :]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, [0], :], x, t)
            x = x_new.detach()
    return x


# ============== Main Training Function ==============


def train_syndiff(args):
    """Single-GPU training for SynDiff on BraTS2020."""

    from backbones.discriminator import Discriminator_small, Discriminator_large
    from backbones.ncsnpp_generator_adagn import NCSNpp
    import backbones.generator_resnet

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = args.batch_size
    nz = args.nz  # latent dimension

    # ===== Create dataset and data loader =====
    # NOTE: Only train + val datasets. Test dataset is loaded by eval.py separately.
    dataset = CreateDatasetSynthesis(
        phase="train",
        input_path=args.input_path,
        contrast1=args.contrast1,
        contrast2=args.contrast2,
        datalist_dir=args.datalist_dir,
    )
    dataset_val = CreateDatasetSynthesis(
        phase="val",
        input_path=args.input_path,
        contrast1=args.contrast1,
        contrast2=args.contrast2,
        datalist_dir=args.datalist_dir,
    )

    # Use standard DataLoader (no DDP)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print(f'Train data size: {len(data_loader)} batches')
    print(f'Val data size: {len(data_loader_val)} batches')

    to_range_0_1 = lambda x: (x + 1.) / 2.

    # ===== Initialize networks =====
    # Diffusive generators (reverse denoising)
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    # Non-diffusive translators (CycleGAN ResNet generators)
    args.num_channels = 1  # Temporarily set for ResNet generator
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[0])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[0])
    args.num_channels = 2  # Restore for diffusive generator

    # Discriminators
    disc_diffusive_1 = Discriminator_large(
        nc=2, ngf=args.ngf,
        t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_large(
        nc=2, ngf=args.ngf,
        t_emb_dim=args.t_emb_dim,
        act=nn.LeakyReLU(0.2)).to(device)

    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(gpu_ids=[0])
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(gpu_ids=[0])

    # ===== Setup optimizers =====
    optimizer_disc_diffusive_1 = optim.Adam(
        disc_diffusive_1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(
        disc_diffusive_2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizer_gen_diffusive_1 = optim.Adam(
        gen_diffusive_1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(
        gen_diffusive_2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    optimizer_gen_non_diffusive_1to2 = optim.Adam(
        gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(
        gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    optimizer_disc_non_diffusive_cycle1 = optim.Adam(
        disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(
        disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # ===== Setup schedulers =====
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen_non_diffusive_1to2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen_non_diffusive_2to1, args.num_epoch, eta_min=1e-5)

    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)

    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc_non_diffusive_cycle1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_disc_non_diffusive_cycle2, args.num_epoch, eta_min=1e-5)

    # ===== Setup experiment directory =====
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_dir = os.path.join(args.output_path, f'task_{current_time}')
    model_dir = os.path.join(task_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Copy source files for reproducibility
    shutil.copyfile(__file__, os.path.join(task_dir, os.path.basename(__file__)))

    print(f"Experiment directory: {task_dir}")
    print(f"Model directory: {model_dir}")

    # ===== Setup TensorBoard =====
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_path = os.path.join(task_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)
    print(f"TensorBoard logs: {tensorboard_path}")

    # ===== Setup diffusion coefficients =====
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)

    # ===== Training loop =====
    global_step = 0

    for epoch in range(0, args.num_epoch + 1):
        # Track epoch losses
        epoch_losses = {
            'G_cycle': 0.0, 'G_L1': 0.0, 'G_adv': 0.0,
            'G_cycle_adv': 0.0, 'G_total': 0.0,
            'D_total': 0.0, 'D_cycle': 0.0
        }
        epoch_batches = 0

        for iteration, (x1, x2) in enumerate(data_loader):
            # ---- Discriminator Training ----
            # Enable discriminator gradients
            for p in disc_diffusive_1.parameters():
                p.requires_grad = True
            for p in disc_diffusive_2.parameters():
                p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = True
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = True

            disc_diffusive_1.zero_grad()
            disc_diffusive_2.zero_grad()

            # Sample from p(x_0)
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            # Sample t
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)

            # Sample x_t and x_tp1
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True

            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x2_t.requires_grad = True

            # Train discriminator with real
            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)

            errD1_real = F.softplus(-D1_real).mean()
            errD2_real = F.softplus(-D2_real).mean()
            errD_real = errD1_real + errD2_real
            errD_real.backward(retain_graph=True)

            # R1 gradient penalty
            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(
                    outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                grad1_penalty = (
                    grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad2_real = torch.autograd.grad(
                    outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                grad2_penalty = (
                    grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(
                        outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                    grad1_penalty = (
                        grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad2_real = torch.autograd.grad(
                        outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                    grad2_penalty = (
                        grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                    grad_penalty.backward()

            # Train with fake
            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)

            # x_tp1 concatenated with source contrast; x_0_predict is predicted
            x1_0_predict_diff = gen_diffusive_1(
                torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(
                torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)

            # Sampling q(x_t | x_0_predict, x_t+1)
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)

            # D output for fake sample x_pos_sample
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

            errD1_fake = F.softplus(output1).mean()
            errD2_fake = F.softplus(output2).mean()
            errD_fake = errD1_fake + errD2_fake
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizer_disc_diffusive_1.step()
            optimizer_disc_diffusive_2.step()

            # ---- Cycle Discriminator Training ----
            disc_non_diffusive_cycle1.zero_grad()
            disc_non_diffusive_cycle2.zero_grad()

            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1)

            errD_cycle1_real = F.softplus(-D_cycle1_real).mean()
            errD_cycle2_real = F.softplus(-D_cycle2_real).mean()
            errD_cycle_real = errD_cycle1_real + errD_cycle2_real
            errD_cycle_real.backward(retain_graph=True)

            # Train with fake
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)

            errD_cycle1_fake = F.softplus(D_cycle1_fake).mean()
            errD_cycle2_fake = F.softplus(D_cycle2_fake).mean()
            errD_cycle_fake = errD_cycle1_fake + errD_cycle2_fake
            errD_cycle_fake.backward()

            errD_cycle = errD_cycle_real + errD_cycle_fake
            optimizer_disc_non_diffusive_cycle1.step()
            optimizer_disc_non_diffusive_cycle2.step()

            # ---- Generator Training ----
            for p in disc_diffusive_1.parameters():
                p.requires_grad = False
            for p in disc_diffusive_2.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters():
                p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters():
                p.requires_grad = False

            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad()
            gen_non_diffusive_2to1.zero_grad()

            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)

            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)

            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            # Translation networks
            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)

            # Diffusive prediction
            x1_0_predict_diff = gen_diffusive_1(
                torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(
                torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)

            # Posterior sampling
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:, [0], :], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:, [0], :], x2_tp1, t2)

            # Adversarial losses for generators
            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)

            errG1 = F.softplus(-output1).mean()
            errG2 = F.softplus(-output2).mean()
            errG_adv = errG1 + errG2

            # Cycle adversarial losses
            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1)

            errG_cycle_adv1 = F.softplus(-D_cycle1_fake).mean()
            errG_cycle_adv2 = F.softplus(-D_cycle2_fake).mean()
            errG_cycle_adv = errG_cycle_adv1 + errG_cycle_adv2

            # L1 reconstruction loss
            errG1_L1 = F.l1_loss(x1_0_predict_diff[:, [0], :], real_data1)
            errG2_L1 = F.l1_loss(x2_0_predict_diff[:, [0], :], real_data2)
            errG_L1 = errG1_L1 + errG2_L1

            # Cycle consistency loss
            errG1_cycle = F.l1_loss(x1_0_predict_cycle, real_data1)
            errG2_cycle = F.l1_loss(x2_0_predict_cycle, real_data2)
            errG_cycle = errG1_cycle + errG2_cycle

            # Total generator loss
            errG = (args.lambda_l1_loss * errG_cycle + errG_adv +
                    errG_cycle_adv + args.lambda_l1_loss * errG_L1)
            errG.backward()

            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step()
            optimizer_gen_non_diffusive_2to1.step()

            # Accumulate epoch losses
            epoch_losses['G_cycle'] += errG_cycle.item()
            epoch_losses['G_L1'] += errG_L1.item()
            epoch_losses['G_adv'] += errG_adv.item()
            epoch_losses['G_cycle_adv'] += errG_cycle_adv.item()
            epoch_losses['G_total'] += errG.item()
            epoch_losses['D_total'] += errD.item()
            epoch_losses['D_cycle'] += errD_cycle.item()
            epoch_batches += 1

            global_step += 1

            # Log iteration-level losses
            if iteration % 100 == 0:
                print(f'epoch {epoch} iter {iteration}, '
                      f'G-Cycle: {errG_cycle.item():.4f}, G-L1: {errG_L1.item():.4f}, '
                      f'G-Adv: {errG_adv.item():.4f}, G-cycle-Adv: {errG_cycle_adv.item():.4f}, '
                      f'G-Sum: {errG.item():.4f}, D: {errD.item():.4f}, D-cycle: {errD_cycle.item():.4f}')

                writer.add_scalar('train/G_cycle_iter', errG_cycle.item(), global_step)
                writer.add_scalar('train/G_L1_iter', errG_L1.item(), global_step)
                writer.add_scalar('train/G_adv_iter', errG_adv.item(), global_step)
                writer.add_scalar('train/G_total_iter', errG.item(), global_step)
                writer.add_scalar('train/D_total_iter', errD.item(), global_step)
                writer.add_scalar('train/D_cycle_iter', errD_cycle.item(), global_step)

        # ---- End of epoch ----

        # Learning rate scheduling
        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()
            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()

        # Average epoch losses and log to TensorBoard
        for key in epoch_losses:
            epoch_losses[key] /= max(epoch_batches, 1)

        writer.add_scalar('epoch/G_cycle', epoch_losses['G_cycle'], epoch)
        writer.add_scalar('epoch/G_L1', epoch_losses['G_L1'], epoch)
        writer.add_scalar('epoch/G_adv', epoch_losses['G_adv'], epoch)
        writer.add_scalar('epoch/G_cycle_adv', epoch_losses['G_cycle_adv'], epoch)
        writer.add_scalar('epoch/G_total', epoch_losses['G_total'], epoch)
        writer.add_scalar('epoch/D_total', epoch_losses['D_total'], epoch)
        writer.add_scalar('epoch/D_cycle', epoch_losses['D_cycle'], epoch)
        writer.add_scalar('epoch/lr_g', optimizer_gen_diffusive_1.param_groups[0]['lr'], epoch)
        writer.add_scalar('epoch/lr_d', optimizer_disc_diffusive_1.param_groups[0]['lr'], epoch)

        print(f'=== Epoch {epoch} ===')
        print(f'  G-Cycle: {epoch_losses["G_cycle"]:.4f}, G-L1: {epoch_losses["G_L1"]:.4f}, '
              f'G-Adv: {epoch_losses["G_adv"]:.4f}, G-cycle-Adv: {epoch_losses["G_cycle_adv"]:.4f}')
        print(f'  G-Total: {epoch_losses["G_total"]:.4f}, D: {epoch_losses["D_total"]:.4f}, '
              f'D-cycle: {epoch_losses["D_cycle"]:.4f}')
        print(f'  LR_G: {optimizer_gen_diffusive_1.param_groups[0]["lr"]:.2e}, '
              f'LR_D: {optimizer_disc_diffusive_1.param_groups[0]["lr"]:.2e}')

        # ---- Save sample images ----
        if epoch % 10 == 0:
            with torch.no_grad():
                # Save diffusive samples
                torchvision.utils.save_image(
                    x1_pos_sample, os.path.join(task_dir, f'xpos1_epoch_{epoch}.png'), normalize=True)
                torchvision.utils.save_image(
                    x2_pos_sample, os.path.join(task_dir, f'xpos2_epoch_{epoch}.png'), normalize=True)

                # Generate samples from noise
                x1_t_sample = torch.cat((torch.randn_like(real_data1), real_data2), axis=1)
                fake_sample1 = sample_from_model(
                    pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t_sample, T, args)
                fake_sample1 = torch.cat((real_data2, fake_sample1), axis=-1)
                torchvision.utils.save_image(
                    fake_sample1, os.path.join(task_dir, f'sample1_epoch_{epoch}.png'), normalize=True)

                x2_t_sample = torch.cat((torch.randn_like(real_data2), real_data1), axis=1)
                fake_sample2 = sample_from_model(
                    pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t_sample, T, args)
                fake_sample2 = torch.cat((real_data1, fake_sample2), axis=-1)
                torchvision.utils.save_image(
                    fake_sample2, os.path.join(task_dir, f'sample2_epoch_{epoch}.png'), normalize=True)

        # ---- Save checkpoint ----
        if epoch % args.save_ckpt_every == 0:
            print(f'Saving checkpoint at epoch {epoch}...')
            torch.save(gen_diffusive_1.state_dict(),
                       os.path.join(model_dir, f'gen_diffusive_1_{epoch}.pth'))
            torch.save(gen_diffusive_2.state_dict(),
                       os.path.join(model_dir, f'gen_diffusive_2_{epoch}.pth'))
            torch.save(gen_non_diffusive_1to2.state_dict(),
                       os.path.join(model_dir, f'gen_non_diffusive_1to2_{epoch}.pth'))
            torch.save(gen_non_diffusive_2to1.state_dict(),
                       os.path.join(model_dir, f'gen_non_diffusive_2to1_{epoch}.pth'))

        # Save latest model
        if epoch % args.save_content_every == 0:
            print(f'Saving full checkpoint at epoch {epoch}...')
            content = {
                'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                'gen_diffusive_1_dict': gen_diffusive_1.state_dict(),
                'gen_diffusive_2_dict': gen_diffusive_2.state_dict(),
                'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(),
                'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(),
                'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),
                'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict(),
                'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                'optimizer_gen_non_diffusive_1to2': optimizer_gen_non_diffusive_1to2.state_dict(),
                'optimizer_gen_non_diffusive_2to1': optimizer_gen_non_diffusive_2to1.state_dict(),
                'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(),
                'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(),
                'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(),
                'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(),
                'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(),
                'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(),
                'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(),
                'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
            }
            torch.save(content, os.path.join(model_dir, 'content.pth'))

        # ---- Validation ----
        if epoch % 10 == 0:
            val_l1_1, val_l1_2 = 0.0, 0.0
            val_count_1, val_count_2 = 0, 0

            with torch.no_grad():
                for iteration, (x_val, y_val) in enumerate(data_loader_val):
                    real_data = x_val.to(device, non_blocking=True)
                    source_data = y_val.to(device, non_blocking=True)

                    x_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
                    fake_sample = sample_from_model(
                        pos_coeff, gen_diffusive_1, args.num_timesteps, x_t, T, args)

                    fake_sample = to_range_0_1(fake_sample)
                    fake_sample = fake_sample / fake_sample.mean()
                    real = to_range_0_1(real_data)
                    real = real / real.mean()

                    val_l1_1 += abs(fake_sample - real).mean().item()
                    val_count_1 += 1

                for iteration, (y_val, x_val) in enumerate(data_loader_val):
                    real_data = x_val.to(device, non_blocking=True)
                    source_data = y_val.to(device, non_blocking=True)

                    x_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
                    fake_sample = sample_from_model(
                        pos_coeff, gen_diffusive_1, args.num_timesteps, x_t, T, args)

                    fake_sample = to_range_0_1(fake_sample)
                    fake_sample = fake_sample / fake_sample.mean()
                    real = to_range_0_1(real_data)
                    real = real / real.mean()

                    val_l1_2 += abs(fake_sample - real).mean().item()
                    val_count_2 += 1

            if val_count_1 > 0:
                writer.add_scalar('val/L1_direction1', val_l1_1 / val_count_1, epoch)
            if val_count_2 > 0:
                writer.add_scalar('val/L1_direction2', val_l1_2 / val_count_2, epoch)

            print(f'  Val L1: dir1={val_l1_1/max(val_count_1,1):.4f}, '
                  f'dir2={val_l1_2/max(val_count_2,1):.4f}')

    writer.close()
    print(f"Training completed. Models saved to: {model_dir}")
    return task_dir


# ============== Main Entry Point ==============

if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=256,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='channel of image (2 = source + noise)')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=64,
                        help='number of initial channels in denoising model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        default=[1, 1, 2, 2, 4, 4],
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='type of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none',
                        choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual',
                        choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum',
                        choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional',
                        choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # Generator and training
    parser.add_argument('--exp', default='BraTS20_syndiff', help='name of experiment')
    parser.add_argument('--input_path', help='path to BraTS2020 patient directories')
    parser.add_argument('--datalist_dir', default=None,
                        help='path to datalist directory (train.list/val.list). '
                             'Defaults to $COMPARE_ROOT/datalist/BraTS2020')
    parser.add_argument('--output_path', default='./results', help='path to output saves')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=200,
                        help='number of epochs (default 200 for unified experiment)')
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.6e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=True,
                        help='save full checkpoint for resuming')
    parser.add_argument('--save_content_every', type=int, default=50,
                        help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=50,
                        help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.5,
                        help='weightening of l1 loss part of diffusion and cycle models')

    parser.add_argument('--contrast1', type=str, default='T1',
                        help='contrast selection for model (kept for API compatibility)')
    parser.add_argument('--contrast2', type=str, default='T2',
                        help='contrast selection for model (kept for API compatibility)')

    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process (kept for compatibility)')

    args = parser.parse_args()

    # Run training
    train_syndiff(args)
