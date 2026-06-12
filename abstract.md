# SynDiff — Pipeline Rule Compliance Log

Reference: `FgC2F-UDiff__TCI2024/train_eval_pipe.md` Section 4.

| Rule | Description | File(s) Changed | What Changed |
|------|-------------|-----------------|--------------|
| **Rule 0** | Global path variables (env var defaults) | `train.py` L805-812, `eval.py` L367-374 | `--input_path` defaults to `$DATA_ROOT`, `--datalist_dir` to `$DATALIST_DIR`, `--output_path` to `$COMPARE_ROOT/results`. Users only need `export` 3 lines when migrating machines. |
| **Rule 1** | Train never loads test data | `train.py` L197, L204 | Only `phase="train"` and `phase="val"` datasets are created. `test.list` never appears in training DataLoader. |
| **Rule 2** | Eval uses test.list, batch_size=1 | `eval.py` L256 | `load_patient_ids('test', …)` loads only test patients. Single-patient iteration (no DataLoader, no batching). |
| **Rule 3** | Eval only generates images | `eval.py` L340-355 | Only saves synthesized NIfTI. No metric computation, no subprocess calls, no PSNR/SSIM/LPIPS code. |
| **Rule 4** | Metric is standalone | — | Already compliant. Metric script reads NIfTI from `prediction/`, GT from `DATA_ROOT`, writes to `prediction_metric_result/`. No model/diffusion/GPU dependency. |
| **Rule 5** | Mask encoding `[flair,t1,t1ce,t2]`, 1=available | `eval.py` L150-166, `dataset.py` L28-43 | MASK_PATTERNS uses D2Diff order, 14 masks (2^4 - 2). |
| **Rule 6** | Modality order `flair,t1,t1ce,t2` | `eval.py` L146, `dataset.py` L23 | Fixed `MODALITY_ORDER` list used consistently. |
| **Rule 7a** | Train output: `models/` + `tensorboard/` + `logs/train.log` | `train.py` L307-309, L316-323 | Creates `models/`, `tensorboard/`, `logs/` dirs under `task_{timestamp}/`. `train.log` via Python `logging` module with FileHandler + StreamHandler. |
| **Rule 7b** | Eval output: `prediction/{mask}/{patient}/{patient}_{mod}.nii.gz`, three-level tqdm, estimated+actual time | `eval.py` L263-270, L268-357 | Output naming changed from `_syn.nii.gz` to `.nii.gz`. Three-level tqdm: Mask (14) → Patient (73) → Slice (~155). Prints `[INFO] Estimated total eval time:~X.X h` at start, `[INFO] Total time: X.XX h` at end. |
| **Rule 7c** | Metric output: `prediction_metric_result/{mask}/result.txt` | — | Already compliant via user's global `syn_metric.py`. |
| **Rule 8** | Training log format | `train.py` L316-323, L597-603, L641-647 | Uses Python `logging` module. Per-step (every `--log_interval`): `Epoch {n}/{N} \| Step {k}/{K} [global {g}/{G}] \| Loss: {v} \| LR: {v}`. Epoch-end summary: `Epoch [{n}/{N}] \| Loss: {avg} \| LR: {v} \| Time: {s}s`. Dual output to console + `logs/train.log`. |
| **Rule 9** | Checkpoint save rules | `train.py` L342-362, L649-685, L731-734 | `0.pt` — before training (init weights). `latest.pt` — every epoch (overwrites). `checkpoint_epoch_{N}.pt` — every 20 epochs (milestone). `final_model.pt` — end of training. All use combined dict: `{epoch, global_step, epoch_loss, lr, args, gen_diffusive_1_dict, …}` per spec format. |
| — | Checkpoint loading (eval side) | `eval.py` L222-243, L230-235 | Unified checkpoint loading from single `.pt` file. Extracts `gen_diffusive_{1,2}_dict` and `gen_non_diffusive_{1to2,2to1}_dict` keys from combined dict. Supports old checkpoint names via `--ckpt_name` arg. |

## Files Modified

| File | Changes |
|------|---------|
| `train.py` | logging module, env var defaults, Rule 8 log format, Rule 9 checkpoint saves, 0.pt/latest.pt/checkpoint_epoch_N.pt/final_model.pt |
| `eval.py` | env var defaults, unified checkpoint loading, `_syn` suffix removed, three-level tqdm, estimated/actual time |
| `run_train.sh` | added `--log_interval 10`, removed old save args |
| `run_eval.sh` | `--ckpt_epoch` → `--ckpt_name latest.pt` |
| `abstract.md` | this file |
