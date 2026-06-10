"""
Inference (prediction) script for SynDiff on BraTS2020.

For each test patient, generates all 14 missing modality combinations:
- mask 0111/1011/1101/1110:  1 modality missing  (4 patterns)
- mask 0011/0101/0110/1001/1010/1100: 2 modalities missing (6 patterns)
- mask 0001/0010/0100/1000:  3 modalities missing (4 patterns)

Saves:
- predictions: results/task_{timestamp}/prediction/{mask_str}/

This script ONLY does inference. Metrics are computed separately by syn_metric.py.

Mask format: 'flair_t1_t1ce_t2' where 1=available, 0=missing.
"""

import argparse
import torch
import numpy as np
import os
import sys
import nibabel as nib
import cv2
import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backbones.ncsnpp_generator_adagn import NCSNpp
import backbones.generator_resnet


# ============== Diffusion Utilities ==============

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


class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device),
             self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) /
            (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
            (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(
            coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(
            0.5 * log_var) * noise

    return p_sample(x_0, x_t, t)


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    """Generate a sample using the diffusive generator (4-step reverse diffusion)."""
    x = x_init[:, [0], :]
    source = x_init[:, [1], :]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x, source), axis=1), t, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, [0], :], x, t)
            x = x_new.detach()
    return x


# ============== Missing Modality Patterns ==============

MODALITY_ORDER = ['flair', 't1', 't1ce', 't2']

# 14 patterns (mask_str: 'flair_t1_t1ce_t2', 1=available, 0=missing)
MASK_PATTERNS = [
    '0111',   # 1:  flair missing
    '1011',   # 2:  t1 missing
    '1101',   # 3:  t1ce missing
    '1110',   # 4:  t2 missing
    '0011',   # 5:  flair+t1 missing
    '0101',   # 6:  flair+t1ce missing
    '0110',   # 7:  flair+t2 missing
    '1001',   # 8:  t1+t1ce missing
    '1010',   # 9:  t1+t2 missing
    '1100',   # 10: t1ce+t2 missing
    '0001',   # 11: flair+t1+t1ce missing
    '0010',   # 12: flair+t1+t2 missing
    '0100',   # 13: flair+t1ce+t2 missing
    '1000',   # 14: t1+t1ce+t2 missing
]


# ============== Data Loading Helpers ==============

def load_patient_ids(phase, datalist_dir):
    """Load patient IDs from datalist file."""
    list_file = os.path.join(datalist_dir, f'{phase}.list')
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"Datalist not found: {list_file}")
    with open(list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_nifti_volume(file_path):
    """Load a nifti volume."""
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.float32), nii.affine


def normalize_volume(data, lower=0, upper=99.5, b_min=0.0, b_max=1.0):
    """Percentile-based normalization (MONAI equivalent)."""
    v_min = np.percentile(data, lower)
    v_max = np.percentile(data, upper)
    data = np.clip(data, v_min, v_max)
    if v_max > v_min:
        data = (data - v_min) / (v_max - v_min) * (b_max - b_min) + b_min
    else:
        data = np.zeros_like(data)
    return data


def load_checkpoint(checkpoint_file, netG, device='cuda:0'):
    """Load model checkpoint, handling DDP 'module.' prefix."""
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    new_state = {}
    for key, val in checkpoint.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state[new_key] = val
    netG.load_state_dict(new_state)
    netG.eval()
    return netG


# ============== Main Evaluation ==============

def run_evaluation(args):
    """Main evaluation: load model → generate predictions for all 14 masks."""
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Setup paths ----
    task_dir = os.path.join(args.output_path, f'task_{args.task_ts}')
    model_dir = os.path.join(task_dir, 'models')
    pred_base = os.path.join(task_dir, 'prediction')

    # ---- Load models ----
    print("Loading models...")
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    args_save = args.num_channels
    args.num_channels = 1
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[args.gpu])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[args.gpu])
    args.num_channels = args_save

    # Load checkpoint weights
    ckpt_epoch = args.ckpt_epoch
    checkpoint_file = os.path.join(model_dir, '{}_{}.pth')

    gen_diffusive_1 = load_checkpoint(
        checkpoint_file.format('gen_diffusive_1', ckpt_epoch),
        gen_diffusive_1, device)
    gen_diffusive_2 = load_checkpoint(
        checkpoint_file.format('gen_diffusive_2', ckpt_epoch),
        gen_diffusive_2, device)
    gen_non_diffusive_1to2 = load_checkpoint(
        checkpoint_file.format('gen_non_diffusive_1to2', ckpt_epoch),
        gen_non_diffusive_1to2, device)
    gen_non_diffusive_2to1 = load_checkpoint(
        checkpoint_file.format('gen_non_diffusive_2to1', ckpt_epoch),
        gen_non_diffusive_2to1, device)

    print(f"Models loaded from {model_dir}, epoch={ckpt_epoch}")

    # ---- Setup diffusion ----
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # ---- Create output directories ----
    for mask_str in MASK_PATTERNS:
        os.makedirs(os.path.join(pred_base, mask_str), exist_ok=True)

    # ---- Load test patients ----
    patient_ids = load_patient_ids('test', args.datalist_dir)
    print(f"Test patients: {len(patient_ids)}")

    # ---- Process each patient ----
    for pidx, patient_id in enumerate(patient_ids):
        print(f"\n[{pidx+1}/{len(patient_ids)}] Processing: {patient_id}")

        patient_dir = os.path.join(args.input_path, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"  [WARNING] Patient dir not found: {patient_dir}")
            continue

        # Load all 4 modality volumes
        volumes = {}
        affine = None
        valid = True
        for mod in MODALITY_ORDER:
            # Try .nii first, then .nii.gz
            nii_path = os.path.join(patient_dir, f'{patient_id}_{mod}.nii')
            if not os.path.exists(nii_path):
                nii_path = os.path.join(patient_dir, f'{patient_id}_{mod}.nii.gz')
            if not os.path.exists(nii_path):
                print(f"  [WARNING] Missing file: {nii_path}, skipping patient")
                valid = False
                break
            data, aff = load_nifti_volume(nii_path)
            data = normalize_volume(data)
            volumes[mod] = data  # (H, W, D)
            if affine is None:
                affine = aff

        if not valid:
            continue

        num_slices = volumes[MODALITY_ORDER[0]].shape[2]
        orig_h, orig_w = volumes[MODALITY_ORDER[0]].shape[:2]

        # Pre-process all slices to [-1,1] range
        slices_11 = {}  # [-1, 1] normalized
        for mod in MODALITY_ORDER:
            vol = volumes[mod]
            mod_slices = np.zeros(
                (num_slices, args.image_size, args.image_size), dtype=np.float32)
            for s in range(num_slices):
                slc = vol[:, :, s].copy()
                slc = np.flipud(slc)
                if slc.shape[0] != args.image_size or slc.shape[1] != args.image_size:
                    slc = cv2.resize(slc, (args.image_size, args.image_size),
                                     interpolation=cv2.INTER_LINEAR)
                mod_slices[s] = slc * 2.0 - 1.0  # [0,1] → [-1,1]
            slices_11[mod] = mod_slices

        # For each mask pattern: synthesize missing modalities
        for mask_str in MASK_PATTERNS:
            mask = [int(c) for c in mask_str]
            available_indices = [i for i, v in enumerate(mask) if v == 1]
            missing_indices = [i for i, v in enumerate(mask) if v == 0]

            # Synthesize each missing modality
            predictions = {}
            for mi in missing_indices:
                pred_vol = np.zeros(
                    (num_slices, args.image_size, args.image_size), dtype=np.float32)

                # Pick the first available modality as source
                si = available_indices[0]

                for s in range(num_slices):
                    source_slice = torch.from_numpy(
                        slices_11[MODALITY_ORDER[si]][s]
                    ).unsqueeze(0).to(device)  # (1, H, W)

                    # Use gen_diffusive_1 + gen_non_diffusive_1to2 for
                    # source→target translation. Since training randomly
                    # samples pairs from all 4 modalities, this generalizes.
                    with torch.no_grad():
                        translated = gen_non_diffusive_1to2(
                            source_slice.unsqueeze(0))  # (1, 1, H, W)
                        x_init = torch.cat(
                            [torch.randn_like(translated), translated], dim=1)
                        pred = sample_from_model(
                            pos_coeff, gen_diffusive_1,
                            args.num_timesteps, x_init, T, args)

                    pred_01 = to_range_0_1(pred).squeeze().cpu().numpy()
                    pred_vol[s] = pred_01

                predictions[mi] = pred_vol

            # ---- Save predictions as NIfTI ----
            for mi in missing_indices:
                # Resize back to original dimensions
                vol_pred = np.zeros((orig_h, orig_w, num_slices), dtype=np.float32)
                for s in range(num_slices):
                    if orig_h != args.image_size or orig_w != args.image_size:
                        vol_pred[:, :, s] = cv2.resize(
                            predictions[mi][s], (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR)
                    else:
                        vol_pred[:, :, s] = predictions[mi][s]
                    vol_pred[:, :, s] = np.flipud(vol_pred[:, :, s])

                mod_name = MODALITY_ORDER[mi]
                save_path = os.path.join(
                    pred_base, mask_str, f'{patient_id}_{mod_name}_syn.nii.gz')
                nii = nib.Nifti1Image(vol_pred, affine=affine)
                nib.save(nii, save_path)

            # ---- Save input (available) and ground truth (missing) ----
            for si in available_indices:
                mod_name = MODALITY_ORDER[si]
                save_path = os.path.join(
                    pred_base, mask_str, f'{patient_id}_{mod_name}_input.nii.gz')
                nii = nib.Nifti1Image(
                    volumes[mod_name].astype(np.float32), affine=affine)
                nib.save(nii, save_path)

            for mi in missing_indices:
                mod_name = MODALITY_ORDER[mi]
                save_path = os.path.join(
                    pred_base, mask_str, f'{patient_id}_{mod_name}_gt.nii.gz')
                nii = nib.Nifti1Image(
                    volumes[mod_name].astype(np.float32), affine=affine)
                nib.save(nii, save_path)

    print(f"\n=== Prediction complete. Results in: {pred_base}/ ===")


def main():
    parser = argparse.ArgumentParser('syndiff evaluation (prediction only)')

    # Path arguments
    parser.add_argument('--input_path', required=True,
                        help='Path to BraTS2020 patient directories ($DATA_ROOT)')
    parser.add_argument('--datalist_dir', required=True,
                        help='Path to datalist directory ($DATALIST_DIR)')
    parser.add_argument('--output_path', required=True,
                        help='Path to results directory ($COMPARE_ROOT/results)')
    parser.add_argument('--task_ts', required=True,
                        help='Task timestamp (e.g., 20250101_120000)')
    parser.add_argument('--ckpt_epoch', default='200',
                        help='Checkpoint epoch to use (default: 200)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    # Model architecture (must match training)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=2)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none')
    parser.add_argument('--progressive_input', type=str, default='residual')
    parser.add_argument('--progressive_combine', type=str, default='sum')
    parser.add_argument('--embedding_type', type=str, default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1024)

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
