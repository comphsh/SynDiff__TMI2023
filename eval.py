"""
Inference (prediction) script for SynDiff on BraTS2020.

For each test patient, generates all 14 missing modality combinations
and saves ONLY the synthesized images (no GT, no input — global DATA_ROOT
serves as the single source of truth).

Mask order follows D2Diff convention:
- mask 0001/0010/0100/1000: 3 missing modalities (4 patterns)
- mask 0011/0101/0110/1001/1010/1100: 2 missing (6 patterns)
- mask 0111/1011/1101/1110: 1 missing (4 patterns)

Output structure (Rule 7b):
  prediction/{mask_str}/{patient_id}/{patient_id}_{mod}.nii.gz

This script ONLY does inference. Metrics are computed separately by the
user's global evaluation script.

Mask format: 'flair_t1_t1ce_t2', 1=available, 0=missing
"""

import argparse
import torch
import numpy as np
import os
import sys
import time
import nibabel as nib
import cv2
from tqdm import tqdm

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
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t)
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
    """4-step reverse diffusion."""
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


# ============== Mask Patterns (D2Diff order) ==============

MODALITY_ORDER = ['flair', 't1', 't1ce', 't2']

# 14 patterns (1=available, 0=missing), D2Diff convention
MASK_PATTERNS = [
    '0001',   # Only t2 available (3-miss)
    '0010',   # Only t1ce available (3-miss)
    '0011',   # t1ce + t2 (2-miss)
    '0100',   # Only t1 available (3-miss)
    '0101',   # t1 + t2 (2-miss)
    '0110',   # t1 + t1ce (2-miss)
    '0111',   # t1 + t1ce + t2 (1-miss: flair)
    '1000',   # Only flair available (3-miss)
    '1001',   # flair + t2 (2-miss)
    '1010',   # flair + t1ce (2-miss)
    '1011',   # flair + t1ce + t2 (1-miss: t1)
    '1100',   # flair + t1 (2-miss)
    '1101',   # flair + t1 + t2 (1-miss: t1ce)
    '1110',   # flair + t1 + t1ce (1-miss: t2)
]


# ============== Helpers ==============

def load_patient_ids(phase, datalist_dir):
    list_file = os.path.join(datalist_dir, f'{phase}.list')
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"Datalist not found: {list_file}")
    with open(list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_nifti_volume(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.float32), nii.affine


def normalize_volume(data, lower=0, upper=99.5, b_min=0.0, b_max=1.0):
    v_min = np.percentile(data, lower)
    v_max = np.percentile(data, upper)
    data = np.clip(data, v_min, v_max)
    if v_max > v_min:
        data = (data - v_min) / (v_max - v_min) * (b_max - b_min) + b_min
    else:
        data = np.zeros_like(data)
    return data


# ============== Main ==============

def run_evaluation(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Paths ----
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

    # Load unified checkpoint (Rule 9 format)
    ckpt_path = os.path.join(model_dir, args.ckpt_name)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract generator state dicts from combined checkpoint
    def extract_state(combined, key_prefix):
        """Load state dict from combined checkpoint, handling DDP prefix."""
        state_dict = combined.get(key_prefix)
        if state_dict is None:
            raise KeyError(f"Key '{key_prefix}' not found in checkpoint")
        new_state = {}
        for k, v in state_dict.items():
            new_state[k[7:] if k.startswith('module.') else k] = v
        return new_state

    gen_diffusive_1.load_state_dict(extract_state(ckpt_data, 'gen_diffusive_1_dict'))
    gen_diffusive_2.load_state_dict(extract_state(ckpt_data, 'gen_diffusive_2_dict'))
    gen_non_diffusive_1to2.load_state_dict(extract_state(ckpt_data, 'gen_non_diffusive_1to2_dict'))
    gen_non_diffusive_2to1.load_state_dict(extract_state(ckpt_data, 'gen_non_diffusive_2to1_dict'))

    gen_diffusive_1.eval()
    gen_diffusive_2.eval()
    gen_non_diffusive_1to2.eval()
    gen_non_diffusive_2to1.eval()

    epoch_info = ckpt_data.get('epoch', '?')
    print(f"Models loaded, epoch={epoch_info}")

    # ---- Setup diffusion ----
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # ---- Create output dirs ----
    for mask_str in MASK_PATTERNS:
        os.makedirs(os.path.join(pred_base, mask_str), exist_ok=True)

    # ---- Load test patients ----
    patient_ids = load_patient_ids('test', args.datalist_dir)
    print(f"Test patients: {len(patient_ids)}")

    # ---- Estimated time (Rule 7b) ----
    n_masks = len(MASK_PATTERNS)
    n_patients = len(patient_ids)
    est_sec = n_masks * n_patients * 155 * 0.01  # ~0.01s per slice per mask
    est_h = est_sec / 3600
    print(f"[INFO] Estimated total eval time: ~{est_h:.1f} h "
          f"({n_masks} masks × {n_patients} patients)")
    eval_start = time.time()

    # ---- Three-level tqdm (Rule 7b) ----
    for mask_str in tqdm(MASK_PATTERNS, desc='Masks', unit='mask'):
        mask = [int(c) for c in mask_str]
        available = [i for i, v in enumerate(mask) if v == 1]
        missing = [i for i, v in enumerate(mask) if v == 0]
        if not missing:
            continue

        for patient_id in tqdm(patient_ids, desc=f'  Patients ({mask_str})',
                               leave=False):
            patient_dir = os.path.join(args.input_path, patient_id)
            if not os.path.isdir(patient_dir):
                continue

            # Load 4 modalities
            volumes = {}
            affine = None
            valid = True
            for mod in MODALITY_ORDER:
                p = os.path.join(patient_dir, f'{patient_id}_{mod}.nii')
                if not os.path.exists(p):
                    p = os.path.join(patient_dir, f'{patient_id}_{mod}.nii.gz')
                if not os.path.exists(p):
                    valid = False
                    break
                data, aff = load_nifti_volume(p)
                volumes[mod] = normalize_volume(data)
                if affine is None:
                    affine = aff
            if not valid:
                continue

            num_slices = volumes[MODALITY_ORDER[0]].shape[2]
            orig_h, orig_w = volumes[MODALITY_ORDER[0]].shape[:2]

            # Pre-process slices to [-1, 1]
            slices_11 = {}
            for mod in MODALITY_ORDER:
                vol = volumes[mod]
                arr = np.zeros((num_slices, args.image_size, args.image_size),
                               dtype=np.float32)
                for s in range(num_slices):
                    slc = np.flipud(vol[:, :, s].copy())
                    if slc.shape != (args.image_size, args.image_size):
                        slc = cv2.resize(slc, (args.image_size, args.image_size),
                                         interpolation=cv2.INTER_LINEAR)
                    arr[s] = slc * 2.0 - 1.0
                slices_11[mod] = arr

            # Synthesize each missing modality
            for mi in missing:
                si = available[0]
                mod_name = MODALITY_ORDER[mi]

                pred_vol = np.zeros((num_slices, args.image_size, args.image_size),
                                    dtype=np.float32)

                for s in tqdm(range(num_slices), desc=f'    gen {mod_name}',
                             leave=False):
                    src = torch.from_numpy(
                        slices_11[MODALITY_ORDER[si]][s]
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        translated = gen_non_diffusive_1to2(src.unsqueeze(0))
                        x_init = torch.cat(
                            [torch.randn_like(translated), translated], dim=1)
                        pred = sample_from_model(
                            pos_coeff, gen_diffusive_1,
                            args.num_timesteps, x_init, T, args)

                    pred_vol[s] = to_range_0_1(pred).squeeze().cpu().numpy()

                # Resize back, flip back, save (Rule 7b: {patient}_{mod}.nii.gz)
                vol_out = np.zeros((orig_h, orig_w, num_slices), dtype=np.float32)
                for s in range(num_slices):
                    if (orig_h, orig_w) != (args.image_size, args.image_size):
                        vol_out[:, :, s] = cv2.resize(
                            pred_vol[s], (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR)
                    else:
                        vol_out[:, :, s] = pred_vol[s]
                    vol_out[:, :, s] = np.flipud(vol_out[:, :, s])

                out_dir = os.path.join(pred_base, mask_str, patient_id)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir,
                                        f'{patient_id}_{mod_name}.nii.gz')
                nib.save(nib.Nifti1Image(vol_out, affine=affine), out_path)

    eval_elapsed = time.time() - eval_start
    print(f"[INFO] Total time: {eval_elapsed/3600:.2f} h")
    print(f"\nDone. Predictions saved to: {pred_base}/")


def main():
    parser = argparse.ArgumentParser('SynDiff eval (prediction-only)')

    # Path args (Rule 0: env var defaults)
    parser.add_argument('--input_path',
                        default=os.environ.get('DATA_ROOT'),
                        help='Path to BraTS2020 data (env: $DATA_ROOT)')
    parser.add_argument('--datalist_dir',
                        default=os.environ.get('DATALIST_DIR'),
                        help='Path to datalist (env: $DATALIST_DIR)')
    parser.add_argument('--output_path',
                        default=os.environ.get('COMPARE_ROOT',
                        os.path.dirname(os.path.abspath(__file__))).rstrip('/') + '/results',
                        help='Results root (env: $COMPARE_ROOT)')
    parser.add_argument('--task_ts', required=True,
                        help='Task timestamp (e.g. 20260605_143022)')
    parser.add_argument('--ckpt_name', default='latest.pt',
                        help='Checkpoint filename in models/ (default: latest.pt)')
    parser.add_argument('--gpu', type=int, default=0)

    # Model arch (must match training)
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
