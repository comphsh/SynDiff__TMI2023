"""
Adapted dataset for BraTS2020 using MONAI-based loading.
Replaces the original .mat-based 2D dataset with nifti-based 3D→2D slice loading.

Modality order: [flair, t1, t1ce, t2]
Normalization: ScaleIntensityRangePercentilesd(lower=0, upper=99.5, b_min=0, b_max=1)
Input range: [0, 1], then mapped to [-1, 1] for SynDiff (centered=True by default).

Supports:
- Loading from nifti files with patient IDs from datalist
- Random missing modality masking during training (excludes all-0 and all-1 patterns)
- 2D slice extraction for SynDiff's 2D architecture
"""

import torch
import numpy as np
import nibabel as nib
import os
import random


# Modality order consistent with the unified experiment framework
MODALITY_ORDER = ['flair', 't1', 't1ce', 't2']

# 14 missing modality patterns (1=available, 0=missing)
# Pattern format: 'flair_t1_t1ce_t2'
# Only patterns with at least 1 available and at least 1 missing are valid
MASK_PATTERNS = [
    '0111',  # mask_id=1:  flair missing, t1+t1ce+t2 available (1 missing)
    '1011',  # mask_id=2:  t1 missing, flair+t1ce+t2 available
    '1101',  # mask_id=3:  t1ce missing, flair+t1+t2 available
    '1110',  # mask_id=4:  t2 missing, flair+t1+t1ce available
    '0011',  # mask_id=5:  flair+t1 missing, t1ce+t2 available (2 missing)
    '0101',  # mask_id=6:  flair+t1ce missing, t1+t2 available
    '0110',  # mask_id=7:  flair+t2 missing, t1+t1ce available
    '1001',  # mask_id=8:  t1+t1ce missing, flair+t2 available
    '1010',  # mask_id=9:  t1+t2 missing, flair+t1ce available
    '1100',  # mask_id=10: t1ce+t2 missing, flair+t1 available
    '0001',  # mask_id=11: flair+t1+t1ce missing, t2 available (3 missing)
    '0010',  # mask_id=12: flair+t1+t2 missing, t1ce available
    '0100',  # mask_id=13: flair+t1ce+t2 missing, t1 available
    '1000',  # mask_id=14: t1+t1ce+t2 missing, flair available
]


def load_patient_ids(phase, datalist_dir):
    """Load patient IDs from datalist files (train/val/test)."""
    list_file = os.path.join(datalist_dir, f'{phase}.list')
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"Datalist file not found: {list_file}")
    with open(list_file, 'r') as f:
        patient_ids = [line.strip() for line in f if line.strip()]
    return patient_ids


def load_nifti_volume(file_path):
    """Load a nifti volume and return as float32 numpy array."""
    nii = nib.load(file_path)
    data = nii.get_fdata().astype(np.float32)
    return data, nii.affine


def normalize_volume(data, lower=0, upper=99.5, b_min=0.0, b_max=1.0):
    """
    Normalize a 3D volume using percentile-based scaling.
    Equivalent to MONAI's ScaleIntensityRangePercentilesd.
    """
    # Compute percentiles
    v_min = np.percentile(data, lower)
    v_max = np.percentile(data, upper)

    # Clip to percentile range
    data = np.clip(data, v_min, v_max)

    # Scale to [b_min, b_max]
    if v_max > v_min:
        data = (data - v_min) / (v_max - v_min) * (b_max - b_min) + b_min
    else:
        data = np.zeros_like(data)

    return data


class BraTSDataset2D(torch.utils.data.Dataset):
    """
    2D slice dataset for SynDiff training on BraTS2020.

    Loads 3D nifti volumes and extracts 2D slices.
    Each item is a pair of (source_modality_slice, target_modality_slice)
    randomly selected from available modalities.

    For training: returns (source, target) 2D tensors
    For eval: returns all 4 modalities with a mask pattern
    """

    def __init__(self, phase, data_root, datalist_dir,
                 modality_order=None, image_size=256,
                 random_mask=False):
        """
        Args:
            phase: 'train', 'val', or 'test'
            data_root: root directory containing patient folders
            datalist_dir: directory containing {phase}.list files
            modality_order: list of modality names, default ['flair','t1','t1ce','t2']
            image_size: target image size (square)
            random_mask: if True, randomly mask modalities during training
        """
        self.phase = phase
        self.data_root = data_root
        self.modality_order = modality_order or MODALITY_ORDER
        self.image_size = image_size
        self.random_mask = random_mask

        # Load patient IDs
        self.patient_ids = load_patient_ids(phase, datalist_dir)
        print(f"[{phase}] Loaded {len(self.patient_ids)} patients")

        # Pre-load all volumes and slices for efficiency
        self._preload_data()

    def _preload_data(self):
        """Pre-load all volumes, normalize, and create slice index."""
        self.slices = []  # List of (patient_idx, slice_idx)
        self.volumes = {}  # patient_idx -> {modality_name: 3D numpy array}

        valid_patients = []
        for pidx, patient_id in enumerate(self.patient_ids):
            patient_dir = os.path.join(self.data_root, patient_id)
            if not os.path.isdir(patient_dir):
                print(f"  Warning: patient dir not found: {patient_dir}")
                continue

            # Load all 4 modalities
            modality_volumes = {}
            valid = True
            for mod in self.modality_order:
                nii_path = os.path.join(patient_dir, f'{patient_id}_{mod}.nii')
                if not os.path.exists(nii_path):
                    # Try .nii.gz extension
                    nii_path = os.path.join(patient_dir, f'{patient_id}_{mod}.nii.gz')
                if not os.path.exists(nii_path):
                    print(f"  Warning: missing file {nii_path}")
                    valid = False
                    break

                vol, _ = load_nifti_volume(nii_path)
                # Normalize to [0, 1]
                vol = normalize_volume(vol)
                modality_volumes[mod] = vol

            if not valid:
                continue

            self.volumes[pidx] = modality_volumes
            valid_patients.append(pidx)

            # Use the first modality to get slice count
            num_slices = modality_volumes[self.modality_order[0]].shape[2]

            # Add all slices
            for sidx in range(num_slices):
                self.slices.append((pidx, sidx))

        # Update patient_ids to only include valid ones
        self.valid_patient_indices = valid_patients
        print(f"[{phase}] Valid patients: {len(valid_patients)}, Total 2D slices: {len(self.slices)}")

    def _get_slice(self, pidx, sidx, modality):
        """Get a 2D slice for a specific modality."""
        vol = self.volumes[pidx][modality]
        # vol shape: (H, W, D) → extract slice → (H, W)
        slc = vol[:, :, sidx]
        slc = np.flipud(slc)  # Flip to match radiological orientation convention

        # Resize to target image_size if needed
        # SynDiff expects square images; BraTS is 240x240 → pad or resize
        h, w = slc.shape
        if h != self.image_size or w != self.image_size:
            # Center crop or pad to target size
            slc = self._resize_to_target(slc)

        return slc.copy()

    def _resize_to_target(self, slc):
        """Resize 2D slice to target image_size."""
        import cv2
        return cv2.resize(slc, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

    def __len__(self):
        if self.random_mask:
            return len(self.slices)
        else:
            # For eval, return number of unique (patient, mask_pattern) combinations
            return len(self.slices)

    def __getitem__(self, idx):
        pidx, sidx = self.slices[idx]

        # Load all 4 modality slices
        mod_slices = []
        for mod in self.modality_order:
            slc = self._get_slice(pidx, sidx, mod)
            mod_slices.append(slc)

        # Stack to (4, H, W)
        all_mods = np.stack(mod_slices, axis=0).astype(np.float32)  # (4, H, W)

        if self.random_mask and self.phase == 'train':
            # Randomly select source and target modalities
            # Pick one as target, one as source (must be different)
            src_idx, tgt_idx = random.sample(range(4), 2)

            # Map [0,1] to [-1,1] for SynDiff
            source = torch.from_numpy(all_mods[src_idx]).unsqueeze(0) * 2.0 - 1.0  # (1, H, W)
            target = torch.from_numpy(all_mods[tgt_idx]).unsqueeze(0) * 2.0 - 1.0  # (1, H, W)

            return source, target

        else:
            # For eval/test: return all 4 modalities
            all_mods_tensor = torch.from_numpy(all_mods)  # (4, H, W)
            return all_mods_tensor


def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    """
    Compatibility wrapper matching the original SynDiff API.
    Now uses MONAI-based BraTS2020 loading.

    Args:
        phase: 'train', 'val', or 'test'
        input_path: path to BraTS2020 data (patient directories with nifti files)
        contrast1: ignored (kept for API compatibility)
        contrast2: ignored (kept for API compatibility)
    """
    # Determine datalist directory
    datalist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datalist', 'BraTS2020')

    is_train = (phase == 'train')
    dataset = BraTSDataset2D(
        phase=phase,
        data_root=input_path,
        datalist_dir=datalist_dir,
        modality_order=MODALITY_ORDER,
        image_size=256,  # SynDiff default
        random_mask=is_train,
    )
    return dataset
