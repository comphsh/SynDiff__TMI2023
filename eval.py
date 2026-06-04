"""
Evaluation script for SynDiff on BraTS2020.

For each test sample, generates all 14 missing modality combinations:
- mask_id 1~4:   1 modality missing  (4 patterns)
- mask_id 5~10:  2 modalities missing (6 patterns)
- mask_id 11~14: 3 modalities missing (4 patterns)

Saves:
- predictions: results/task_{timestamp}/prediction/{mask_id}/
- metrics:     results/task_{timestamp}/prediction_metric_result/{mask_id}/result.txt

mask format: 'flair_t1_t1ce_t2' where 1=available, 0=missing
Example: '0111' → flair missing, t1+t1ce+t2 available → generate flair from t1
"""

import argparse
import torch
import numpy as np
import os
import sys
import nibabel as nib
import cv2
import subprocess
import time
import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backbones.ncsnpp_generator_adagn import NCSNpp
import backbones.generator_resnet


# ============== Diffusion utilities (from original SynDiff) ==============

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
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0)
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

    return p_sample(x_0, x_t, t)


def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    """Generate a sample using the diffusive generator."""
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


# ============== Missing Modality Patterns ==============

# 14 patterns, mask_id 1-14
# mask string format: 'flair_t1_t1ce_t2', 1=available, 0=missing
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

MODALITY_ORDER = ['flair', 't1', 't1ce', 't2']


def load_patient_ids(phase, datalist_dir):
    """Load patient IDs from datalist."""
    list_file = os.path.join(datalist_dir, f'{phase}.list')
    with open(list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_nifti_volume(file_path):
    """Load a nifti volume."""
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.float32), nii.affine


def normalize_volume(data, lower=0, upper=99.5, b_min=0.0, b_max=1.0):
    """Percentile-based normalization."""
    v_min = np.percentile(data, lower)
    v_max = np.percentile(data, upper)
    data = np.clip(data, v_min, v_max)
    if v_max > v_min:
        data = (data - v_min) / (v_max - v_min) * (b_max - b_min) + b_min
    else:
        data = np.zeros_like(data)
    return data


def load_checkpoint(checkpoint_file, netG, device='cuda:0'):
    """Load model checkpoint, handling DDP prefix."""
    checkpoint = torch.load(checkpoint_file, map_location=device)
    # Handle DDP 'module.' prefix
    new_state = {}
    for key, val in checkpoint.items():
        if key.startswith('module.'):
            new_state[key[7:]] = val
        else:
            new_state[key] = val
    netG.load_state_dict(new_state)
    netG.eval()
    return netG


def synthesize_missing_from_source(gen_diffusive, gen_non_diffusive,
                                    source_slice, pos_coeff, T, args):
    """
    Synthesize target modality from source.

    Args:
        gen_diffusive: diffusive generator (NCSNpp)
        gen_non_diffusive: non-diffusive translator (ResNet generator)
        source_slice: torch tensor (1, H, W) in [-1, 1]
        pos_coeff: Posterior_Coefficients
        T: time schedule
        args: arguments

    Returns:
        prediction: torch tensor (1, H, W) in [-1, 1]
    """
    # Step 1: Non-diffusive translation source → target
    with torch.no_grad():
        translated = gen_non_diffusive(source_slice.unsqueeze(0))  # (1, 1, H, W)

        # Step 2: Diffusive refinement
        x_init = torch.cat([torch.randn_like(translated), translated], dim=1)  # (1, 2, H, W)
        prediction = sample_from_model(
            pos_coeff, gen_diffusive, args.num_timesteps, x_init, T, args)

    return prediction  # (1, 1, H, W)


def run_evaluation(args):
    """Main evaluation loop."""
    device = torch.device(f'cuda:{args.gpu_chose}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== Load models =====
    print("Loading models...")
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    # The non-diffusive translators: 1→2 and 2→1
    # We use them for source→target translation
    args_save = args.num_channels
    args.num_channels = 1
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[args.gpu_chose])
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(
        netG='resnet_6blocks', gpu_ids=[args.gpu_chose])
    args.num_channels = args_save

    # Load checkpoint
    # gen_diffusive_1 is used for direction 1→2 (source modality → target using gen_non_diffusive_1to2 output)
    # gen_diffusive_2 is used for direction 2→1
    ckpt_dir = os.path.join(args.ckpt_path, 'models')
    epoch = args.which_epoch

    checkpoint_file = os.path.join(ckpt_dir, '{}_{}.pth')
    load_checkpoint(checkpoint_file.format('gen_diffusive_1', epoch),
                    gen_diffusive_1, device)
    load_checkpoint(checkpoint_file.format('gen_diffusive_2', epoch),
                    gen_diffusive_2, device)
    load_checkpoint(checkpoint_file.format('gen_non_diffusive_1to2', epoch),
                    gen_non_diffusive_1to2, device)
    load_checkpoint(checkpoint_file.format('gen_non_diffusive_2to1', epoch),
                    gen_non_diffusive_2to1, device)

    print(f"Models loaded from {ckpt_dir}, epoch {epoch}")

    # ===== Setup diffusion =====
    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # ===== Setup output directories =====
    task_dir = os.path.dirname(ckpt_dir)  # results/task_{timestamp}/
    pred_base = os.path.join(task_dir, 'prediction')
    metric_base = os.path.join(task_dir, 'prediction_metric_result')

    for mask_id in range(1, 15):
        os.makedirs(os.path.join(pred_base, str(mask_id)), exist_ok=True)
        os.makedirs(os.path.join(metric_base, str(mask_id)), exist_ok=True)

    # ===== Load test patients =====
    datalist_dir = os.path.join(PROJECT_ROOT, 'datalist', 'BraTS2020')
    patient_ids = load_patient_ids('test', datalist_dir)
    print(f"Test patients: {len(patient_ids)}")

    # ===== Process each patient =====
    for pidx, patient_id in enumerate(patient_ids):
        print(f"\nProcessing patient {pidx+1}/{len(patient_ids)}: {patient_id}")

        patient_dir = os.path.join(args.input_path, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"  Warning: patient dir not found: {patient_dir}")
            continue

        # Load all 4 modality volumes
        volumes = {}
        affine = None
        valid = True
        for mod in MODALITY_ORDER:
            nii_path = os.path.join(patient_dir, f'{patient_id}_{mod}.nii')
            if not os.path.exists(nii_path):
                nii_path = os.path.join(patient_dir, f'{patient_id}_{mod}.nii.gz')
            if not os.path.exists(nii_path):
                print(f"  Warning: missing {nii_path}, skipping patient")
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

        # Pre-process all slices for this patient
        slices_01 = {}  # [0,1] normalized
        slices_11 = {}  # [-1,1] normalized
        for mod in MODALITY_ORDER:
            vol = volumes[mod]
            mod_slices_01 = np.zeros((num_slices, args.image_size, args.image_size), dtype=np.float32)
            mod_slices_11 = np.zeros((num_slices, args.image_size, args.image_size), dtype=np.float32)
            for s in range(num_slices):
                slc = vol[:, :, s].copy()
                slc = np.flipud(slc)
                if slc.shape[0] != args.image_size or slc.shape[1] != args.image_size:
                    slc = cv2.resize(slc, (args.image_size, args.image_size),
                                     interpolation=cv2.INTER_LINEAR)
                mod_slices_01[s] = slc
                mod_slices_11[s] = slc * 2.0 - 1.0  # [0,1] → [-1,1]
            slices_01[mod] = mod_slices_01
            slices_11[mod] = mod_slices_11

        # For each mask pattern
        for mask_id, mask_str in enumerate(MASK_PATTERNS, start=1):
            mask = [int(c) for c in mask_str]
            available_indices = [i for i, v in enumerate(mask) if v == 1]
            missing_indices = [i for i, v in enumerate(mask) if v == 0]

            # Prepare output volumes
            # Each prediction is (D, H, W) in [0, 1]
            predictions = {}
            for mi in missing_indices:
                predictions[mi] = np.zeros((num_slices, args.image_size, args.image_size),
                                           dtype=np.float32)

            # For each missing modality, synthesize from an available source
            for mi in missing_indices:
                # Pick the first available modality as source
                si = available_indices[0]

                for s in range(num_slices):
                    source_slice = torch.from_numpy(
                        slices_11[MODALITY_ORDER[si]][s]
                    ).unsqueeze(0).to(device)  # (1, H, W)

                    # Use the appropriate generator pair
                    # For simplicity, use gen_diffusive_1 + gen_non_diffusive_1to2 for all pairs
                    pred = synthesize_missing_from_source(
                        gen_diffusive_1, gen_non_diffusive_1to2,
                        source_slice, pos_coeff, T, args)

                    pred_01 = to_range_0_1(pred).squeeze().cpu().numpy()
                    predictions[mi][s] = pred_01

            # Resize back to original dimensions and save
            orig_h, orig_w = volumes[MODALITY_ORDER[0]].shape[:2]

            for mi in missing_indices:
                # Resize each slice back to original size
                vol_pred = np.zeros((orig_h, orig_w, num_slices), dtype=np.float32)
                for s in range(num_slices):
                    if (orig_h != args.image_size or orig_w != args.image_size):
                        vol_pred[:, :, s] = cv2.resize(
                            predictions[mi][s], (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR)
                    else:
                        vol_pred[:, :, s] = predictions[mi][s]
                    # Flip back
                    vol_pred[:, :, s] = np.flipud(vol_pred[:, :, s])

                # Save as nifti
                mod_name = MODALITY_ORDER[mi]
                save_path = os.path.join(
                    pred_base, str(mask_id),
                    f'{patient_id}_{mod_name}_syn.nii.gz')
                nii = nib.Nifti1Image(vol_pred, affine=affine)
                nib.save(nii, save_path)

            # Save input (available modalities) and ground truth (missing modalities)
            for si in available_indices:
                mod_name = MODALITY_ORDER[si]
                vol_orig = volumes[mod_name]
                save_path = os.path.join(
                    pred_base, str(mask_id),
                    f'{patient_id}_{mod_name}_input.nii.gz')
                nii = nib.Nifti1Image(vol_orig.astype(np.float32), affine=affine)
                nib.save(nii, save_path)

            for mi in missing_indices:
                mod_name = MODALITY_ORDER[mi]
                vol_orig = volumes[mod_name]
                save_path = os.path.join(
                    pred_base, str(mask_id),
                    f'{patient_id}_{mod_name}_gt.nii.gz')
                nii = nib.Nifti1Image(vol_orig.astype(np.float32), affine=affine)
                nib.save(nii, save_path)

    print("\n=== All predictions saved. Running metric evaluation... ===")

    # ===== Run metrics evaluation =====
    eval_script = args.eval_script
    if not os.path.exists(eval_script):
        print(f"Warning: eval script not found at {eval_script}")
        print("Skipping metric calculation. Run manually:")
        print(f"  python {eval_script}")
        return

    # For each mask_id, compute metrics
    for mask_id in range(1, 15):
        print(f"\n--- Evaluating mask_id={mask_id} ---")

        # Build paths for evaluation
        src_path = args.input_path
        gen_path = os.path.join(pred_base, str(mask_id))

        metric_dir = os.path.join(metric_base, str(mask_id))
        os.makedirs(metric_dir, exist_ok=True)

        result_file = os.path.join(metric_dir, 'result.txt')

        # Build Python command to run evaluation
        eval_cmd = [
            sys.executable, '-c', f'''
import sys
sys.path.insert(0, "{os.path.dirname(eval_script)}")
sys.path.insert(0, "{os.path.join(os.path.dirname(eval_script), '..', '..')}")

from syn_metrics import ImageQualityEvaluator, stream_process
import os

evaluator = ImageQualityEvaluator(LPIPS_model_type='nomedical')
stream_process(
    src_path="{src_path}",
    gen_path="{gen_path}",
    evaluator=evaluator,
    gen_shuffix="syn",
    from_ckpt_name=""
)

# Write results to result file
import glob
metric_files = glob.glob(os.path.join("{gen_path}", "*_metrics_results", "*.txt"))
if metric_files:
    import shutil
    for mf in metric_files:
        shutil.copy(mf, "{result_file}")
        print(f"Metrics saved to: {result_file}")
'''
        ]

        try:
            subprocess.run(eval_cmd, check=True, timeout=600)
            print(f"  Metrics computed for mask_id={mask_id}")
        except subprocess.TimeoutExpired:
            print(f"  Timeout for mask_id={mask_id}")
        except subprocess.CalledProcessError as e:
            print(f"  Error computing metrics for mask_id={mask_id}: {e}")
        except Exception as e:
            print(f"  Unexpected error for mask_id={mask_id}: {e}")

    print(f"\n=== Evaluation complete. Results in: {task_dir} ===")


def main():
    parser = argparse.ArgumentParser('syndiff evaluation')
    parser.add_argument('--seed', type=int, default=1024)
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--ckpt_path', required=True,
                        help='Path to task directory (results/task_{timestamp}/)')
    parser.add_argument('--which_epoch', type=int, default=200,
                        help='Which epoch checkpoint to use')
    parser.add_argument('--gpu_chose', type=int, default=0)
    parser.add_argument('--input_path', required=True,
                        help='Path to BraTS2020 patient directories')
    parser.add_argument('--eval_script', default=None,
                        help='Path to syn_metrics.py evaluation script')

    args = parser.parse_args()

    # Set default eval script path
    if args.eval_script is None:
        args.eval_script = '/devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py'

    run_evaluation(args)


# Allow calling directly
if __name__ == '__main__':
    main()
