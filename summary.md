# SynDiff 代码分析与适配报告

## 1. 方法基本信息

| 项目 | 内容 |
|------|------|
| **方法名称** | SynDiff: Unsupervised Medical Image Translation With Adversarial Diffusion Models |
| **发表期刊** | IEEE Transactions on Medical Imaging (TMI), Vol. 42, No. 12, Dec. 2023 |
| **论文链接** | https://arxiv.org/abs/2207.08208 |
| **官方仓库** | https://github.com/icon-lab/SynDiff (upstream) |
| **本复现仓库** | https://github.com/comphsh/SynDiff__TMI2023 |
| **基础框架** | PyTorch 2.7, torchvision |
| **硬件需求** | GPU ≥ 48GB (Quadro RTX 8000) |
| **许可协议** | NVIDIA Source Code License |

## 2. 核心创新点

1. **对抗扩散模型 (Adversarial Diffusion)**：将 GAN 对抗训练引入扩散模型的去噪过程，判别器在扩散时间步上判断去噪图像的真假

2. **双生成器架构**：
   - **扩散生成器 (NCSNpp)**：基于 Score SDE 的条件生成器，4 步快速采样
   - **非扩散翻译器 (ResNet)**：CycleGAN 风格的快速翻译网络，提供初始翻译

3. **后验采样**：在 DDPM 后验分布 q(xt | x0, xt+1) 中采样，结合条件生成

4. **双向循环一致性**：同时训练正向和反向翻译，使用循环一致性损失

## 3. 代码仓库结构

```
SynDiff__TMI2023/
├── backbones/                       # 模型骨干网络
│   ├── ncsnpp_generator_adagn.py    # NCSN++ 扩散生成器 (AdaGN)
│   ├── generator_resnet.py          # ResNet 翻译器 + PatchGAN 判别器
│   ├── discriminator.py             # 时间条件判别器
│   ├── layers.py / layerspp.py      # 基础网络层
│   ├── dense_layer.py               # 全连接层
│   └── up_or_down_sampling.py       # 抗混叠上下采样
├── utils/
│   ├── EMA.py                       # 指数移动平均
│   └── op/                          # CUDA 自定义算子 (已降级为 PyTorch)
├── datalist/BraTS2020/              # 数据集划分 (70/10/20)
│   ├── train.list (260), val.list (36), test.list (73)
├── dataset.py                        # 数据加载 (NIfTI 在线加载)
├── train.py                          # 训练脚本 (单卡)
├── eval.py                           # 推理脚本 (仅预测，不含指标)
├── run_train.sh                      # 训练启动脚本
├── run_eval.sh                       # 评估启动脚本
└── summary.md                        # 本报告
```

## 4. 数据加载方式

- **格式**：直接从 NIfTI (.nii / .nii.gz) 加载
- **维度**：3D → 逐 2D 切片提取 (256×256)
- **归一化**：百分位归一化 (0-99.5%) → [0,1] → 映射到 [-1,1]
- **训练**：随机采样源-目标模态对 (4 选 2)
- **测试**：按 14 种 mask 模式加载

## 5. 原模型架构

```
训练: x_t+1 + source_modality → Gen_diffusive → x_0_predict
      source → Gen_non_diffusive → translated
      ↓
      Disc_diffusive(x_pos_sample, t, x_t+1) → real/fake
      Disc_cycle → cycle consistency
      ↓
      Loss = L_adv + λ*L_cycle + L_cycle_adv + λ*L_l1
```

**超参数**：
- num_timesteps=4, num_channels_dae=64, ch_mult=[1,1,2,2,4,4]
- nz=100, z_emb_dim=256, t_emb_dim=256
- lr_g=1.6e-4, lr_d=1e-4, beta1=0.5, beta2=0.9
- batch_size=1, num_epoch=200
- lambda_l1_loss=0.5, r1_gamma=0.05

## 6. 统一实验配置

### 数据集
- **BraTS 2020**：369 例
- **划分**：train 260, val 36, test 73 (70/10/20)
- **数据路径**：`$DATA_ROOT/{patient_name}/{patient_name}_{modal}.nii`
- **模态顺序**：flair, t1, t1ce, t2

### 缺失模式 (14 种)
| mask | 模式 | 可用模态 | 缺失数 | 描述 |
|------|------|---------|--------|------|
| 0111 | 1-miss | t1, t1ce, t2 | 1 | Flair 缺失 |
| 1011 | 1-miss | flair, t1ce, t2 | 1 | T1 缺失 |
| 1101 | 1-miss | flair, t1, t2 | 1 | T1CE 缺失 |
| 1110 | 1-miss | flair, t1, t1ce | 1 | T2 缺失 |
| 0011 | 2-miss | t1ce, t2 | 2 | Flair+T1 缺失 |
| 0101 | 2-miss | t1, t2 | 2 | Flair+T1CE 缺失 |
| 0110 | 2-miss | t1, t1ce | 2 | Flair+T2 缺失 |
| 1001 | 2-miss | flair, t2 | 2 | T1+T1CE 缺失 |
| 1010 | 2-miss | flair, t1ce | 2 | T1+T2 缺失 |
| 1100 | 2-miss | flair, t1 | 2 | T1CE+T2 缺失 |
| 0001 | 3-miss | t2 | 3 | Flair+T1+T1CE 缺失 |
| 0010 | 3-miss | t1ce | 3 | Flair+T1+T2 缺失 |
| 0100 | 3-miss | t1 | 3 | Flair+T1CE+T2 缺失 |
| 1000 | 3-miss | flair | 3 | T1+T1CE+T2 缺失 |

### 评价指标
- SSIM, PSNR, MSE, MAE, LPIPS, FID
- 由外部 `syn_metric.py` 统一计算

---

## 7. 完整执行命令指南

### 7.0 环境准备

```bash
# ============================================================
# 全局路径变量（按实际修改）
# ============================================================
export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/SynDiff__TMI2023
export DATA_ROOT=/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
export DATALIST_DIR=$COMPARE_ROOT/datalist/BraTS2020

# ============================================================
# 1. 切换到实验分支
# ============================================================
cd $COMPARE_ROOT
git checkout experiment/SynDiff_adapt

# ============================================================
# 2. 验证环境依赖
# ============================================================
# 需要安装的 Python 包:
#   torch, torchvision, tensorboard, nibabel, numpy, opencv-python
#
# 检查 CUDA 可用性:
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# ============================================================
# 3. 验证数据目录结构（无需预处理！）
# ============================================================
# 期望格式: $DATA_ROOT/{patient_name}/{patient_name}_{modal}.nii
# 示例:
#   $DATA_ROOT/BraTS20_Training_001/BraTS20_Training_001_flair.nii
#   $DATA_ROOT/BraTS20_Training_001/BraTS20_Training_001_t1.nii
#   $DATA_ROOT/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii
#   $DATA_ROOT/BraTS20_Training_001/BraTS20_Training_001_t2.nii
#
ls $DATA_ROOT/BraTS20_Training_001/

# ============================================================
# 4. 验证数据集划分
# ============================================================
echo "Train: $(wc -l < $DATALIST_DIR/train.list) patients"
echo "Val:   $(wc -l < $DATALIST_DIR/val.list) patients"
echo "Test:  $(wc -l < $DATALIST_DIR/test.list) patients"

# ============================================================
# 5. 快速冒烟测试（验证数据加载 + 模型前向，约2分钟）
# ============================================================
python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from dataset import BraTSDataset2D, MODALITY_ORDER
from backbones.ncsnpp_generator_adagn import NCSNpp
from backbones.discriminator import Discriminator_large
import backbones.generator_resnet
import torch.nn as nn

DATA_ROOT = '$DATA_ROOT'
DATALIST_DIR = '$DATALIST_DIR'

# 测试数据加载
print('=== 测试数据加载 ===')
ds = BraTSDataset2D('train', data_root=DATA_ROOT, datalist_dir=DATALIST_DIR, random_mask=True)
src, tgt = ds[0]
print(f'Train sample: source={src.shape}, range=[{src.min():.2f},{src.max():.2f}]')

ds_val = BraTSDataset2D('val', data_root=DATA_ROOT, datalist_dir=DATALIST_DIR, random_mask=True)
src, tgt = ds_val[0]
print(f'Val sample: source={src.shape}, target={tgt.shape}')

# 测试模型
print('=== 测试模型创建与前向传播 ===')
class A: pass
args = A()
for k,v in {'image_size':256,'num_channels':2,'num_channels_dae':64,
            'ch_mult':[1,1,2,2,4,4],'num_res_blocks':2,'attn_resolutions':(16,),
            'dropout':0.,'resamp_with_conv':True,'conditional':True,'fir':True,
            'fir_kernel':[1,3,3,1],'skip_rescale':True,'resblock_type':'biggan',
            'progressive':'none','progressive_input':'residual','progressive_combine':'sum',
            'embedding_type':'positional','fourier_scale':16.,'not_use_tanh':False,
            'centered':True,'nz':100,'n_mlp':3,'z_emb_dim':256,'t_emb_dim':256,'ngf':64}.items():
    setattr(args, k, v)

device = 'cuda'
gen = NCSNpp(args).to(device)
x = torch.cat([torch.randn(1,1,256,256), torch.randn(1,1,256,256)], dim=1).to(device)
out = gen(x, torch.randint(0,4,(1,)).to(device), torch.randn(1,100).to(device))
print(f'NCSNpp forward: {x.shape} -> {out.shape}')

disc = Discriminator_large(nc=2, ngf=64, t_emb_dim=256, act=nn.LeakyReLU(0.2)).to(device)
d_out = disc(x, torch.randint(0,4,(1,)).to(device), x)
print(f'Discriminator forward: -> {d_out.shape}')

args.num_channels = 1
trans = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[0])
t_out = trans(torch.randn(1,1,256,256).to(device))
print(f'Translator forward: -> {t_out.shape}')

print('🎉 冒烟测试全部通过！')
"
```

### 7.1 训练命令

#### 方式一：使用 shell 脚本（推荐）

```bash
# 直接执行训练脚本
cd $COMPARE_ROOT
bash run_train.sh
```

#### 方式二：直接调用 Python（可自定义参数）

```bash
# ============================================================
# 完整训练命令（等效于 run_train.sh）
# ============================================================
cd $COMPARE_ROOT

python train.py \
    --image_size 256 \
    --num_channels 2 \
    --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --batch_size 1 \
    --num_epoch 200 \
    --ngf 64 \
    --embedding_type positional \
    --z_emb_dim 256 \
    --t_emb_dim 256 \
    --lr_g 1.6e-4 \
    --lr_d 1e-4 \
    --beta1 0.5 \
    --beta2 0.9 \
    --r1_gamma 0.05 \
    --lazy_reg 10 \
    --lambda_l1_loss 0.5 \
    --nz 100 \
    --save_content \
    --save_ckpt_every 50 \
    --save_content_every 50 \
    --exp BraTS20_syndiff \
    --input_path "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
    --output_path "$COMPARE_ROOT/results"
```

#### 训练输出结构

```
results/
└── task_20260605_143022/          # task_{timestamp}
    ├── models/                     # 模型保存目录
    │   ├── gen_diffusive_1_0.pth   # Epoch 0
    │   ├── gen_diffusive_2_200.pth # 最终模型
    │   ├── gen_non_diffusive_1to2_200.pth
    │   ├── gen_non_diffusive_2to1_200.pth
    │   └── content.pth             # 完整 checkpoint（可 resume）
    ├── tensorboard/                # TensorBoard 日志
    │   └── events.out.tfevents.*
    ├── train.py                    # 训练脚本备份
    ├── xpos1_epoch_10.png          # 训练样本可视化
    └── sample1_epoch_10.png
```

#### 训练过程监控

```bash
# ============================================================
# 启动 TensorBoard 查看训练曲线
# ============================================================
tensorboard --logdir $COMPARE_ROOT/results/task_*/tensorboard --port 6006
# 浏览器打开: http://localhost:6006

# 监控的指标:
#   epoch/G_total      - 生成器总损失
#   epoch/G_L1         - L1 重建损失
#   epoch/G_cycle      - 循环一致性损失
#   epoch/G_adv        - 对抗损失
#   epoch/D_total      - 判别器损失
#   epoch/lr_g         - 生成器学习率
#   epoch/lr_d         - 判别器学习率

# ============================================================
# 用 nvidia-smi 监控 GPU 使用
# ============================================================
watch -n 1 nvidia-smi

# ============================================================
# 后台运行（推荐）
# ============================================================
nohup bash run_train.sh > train.log 2>&1 &
tail -f train.log
```

#### 训练时间预估

- 每 epoch 约 8-15 分钟（取决于 slice 数量和 GPU）
- 200 epochs 总计约 **27-50 小时**

---

### 7.2 推理命令 (eval.py — 仅预测合成)

> **eval.py 只做推理预测，不计算指标。** 指标计算由 syn_metric.py 单独完成。

#### 方式一：使用 shell 脚本（推荐）

```bash
# ============================================================
# 完整 pipeline：推理 → 指标（两步自动执行）
# ============================================================
cd $COMPARE_ROOT

# <task_ts> 替换为训练时生成的时间戳，例如 20260605_143022
bash run_eval.sh 20260605_143022

# 使用特定 epoch 的模型
bash run_eval.sh 20260605_143022 200

# 仅重新计算指标（跳过推理，需要已有预测结果）
bash run_eval.sh 20260605_143022 200 --metrics_only
```

#### 方式二：单独调用 eval.py（仅推理）

```bash
# ============================================================
# Step 1: eval.py — 加载模型 → 14种mask推理 → 保存 NIfTI
# ============================================================
TASK_TS="20260605_143022"   # 替换为实际时间戳

cd $COMPARE_ROOT

python eval.py \
    --input_path "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
    --output_path "$COMPARE_ROOT/results" \
    --task_ts "$TASK_TS" \
    --ckpt_epoch 200 \
    --gpu 0 \
    --image_size 256 \
    --num_channels 2 \
    --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --embedding_type positional \
    --z_emb_dim 256 \
    --t_emb_dim 256
```

#### 推理输出结构 (mask_str = 4位二进制字符串)

```
results/task_20260605_143022/
├── models/                                   # 训练输出（权重在此）
│   ├── gen_diffusive_1_200.pth
│   └── gen_diffusive_2_200.pth
│
├── prediction/                               # 推理输出（eval.py 生成）
│   ├── 0111/                                 # mask=0111: flair缺失 (1-miss)
│   │   ├── BraTS20_Training_137_flair_syn.nii.gz   (合成)
│   │   ├── BraTS20_Training_137_t1_input.nii.gz     (输入)
│   │   ├── BraTS20_Training_137_t1ce_input.nii.gz   (输入)
│   │   ├── BraTS20_Training_137_t2_input.nii.gz     (输入)
│   │   ├── BraTS20_Training_137_t1_gt.nii.gz        (真值)
│   │   └── ... (73 patients × 4 modalities)
│   ├── 1011/                                 # mask=1011: t1缺失 (1-miss)
│   ├── 1101/                                 # mask=1101: t1ce缺失 (1-miss)
│   ├── 1110/                                 # mask=1110: t2缺失 (1-miss)
│   ├── 0011/                                 # mask=0011: flair+t1缺失 (2-miss)
│   ├── 0101/                                 # mask=0101: flair+t1ce缺失 (2-miss)
│   ├── 0110/                                 # mask=0110: flair+t2缺失 (2-miss)
│   ├── 1001/                                 # mask=1001: t1+t1ce缺失 (2-miss)
│   ├── 1010/                                 # mask=1010: t1+t2缺失 (2-miss)
│   ├── 1100/                                 # mask=1100: t1ce+t2缺失 (2-miss)
│   ├── 0001/                                 # mask=0001: flair+t1+t1ce缺失 (3-miss)
│   ├── 0010/                                 # mask=0010: flair+t1+t2缺失 (3-miss)
│   ├── 0100/                                 # mask=0100: flair+t1ce+t2缺失 (3-miss)
│   └── 1000/                                 # mask=1000: t1+t1ce+t2缺失 (3-miss)
│
└── prediction_metric_result/                 # 评估输出（syn_metric.py 生成）
    ├── 0111/result.txt
    ├── 1011/result.txt
    ├── ...
    └── 1110/result.txt
```

**Mask 编码规则**：
- 4位二进制字符串，顺序 **`[flair, t1, t1ce, t2]`**
- `1` = 可用（作为条件输入），`0` = 缺失（需要合成）
- 例如 `1110` = flair/t1/t1ce 可用，t2 缺失（1-missing 模式）
- 共 14 种（排除 `0000` 全缺失和 `1111` 全可用）

#### result.txt 格式

```
ssim     psnr     mse      mae      fid     lpips
0.891234   28.456789  0.001234  0.012345  -1.000000   0.056789
```

> **注意**: `fid` 为 -1 表示未计算（FID 需要全数据集特征统计）。LPIPS 需要 `monai` 包启用。

---

### 7.3 查看与汇总所有指标

```bash
TASK_TS="20260605_143022"
METRIC_DIR="$COMPARE_ROOT/results/task_$TASK_TS/prediction_metric_result"

# ============================================================
# 逐 mask 查看
# ============================================================
for mask_str in 0111 1011 1101 1110 0011 0101 0110 1001 1010 1100 0001 0010 0100 1000; do
    f="$METRIC_DIR/$mask_str/result.txt"
    if [ -f "$f" ]; then
        printf "mask_%-4s: %s\n" "$mask_str" "$(tail -1 "$f")"
    fi
done

# ============================================================
# 按类型分组的汇总脚本
# ============================================================
echo ""
echo "=============================================="
echo " SynDiff BraTS2020 评估结果汇总"
echo "=============================================="
printf "%-8s %-8s %-10s %-10s %-12s %-12s %-10s\n" \
    "mask" "type" "ssim" "psnr" "mse" "mae" "lpips"
echo "--------------------------------------------------------------------"

ONE_MISS=("0111" "1011" "1101" "1110")
TWO_MISS=("0011" "0101" "0110" "1001" "1010" "1100")
THREE_MISS=("0001" "0010" "0100" "1000")

for mask_str in "${ONE_MISS[@]}"; do
    f="$METRIC_DIR/$mask_str/result.txt"
    vals=$(tail -1 "$f" 2>/dev/null || echo "N/A")
    printf "%-8s %-8s %s\n" "$mask_str" "1-miss" "$vals"
done
for mask_str in "${TWO_MISS[@]}"; do
    f="$METRIC_DIR/$mask_str/result.txt"
    vals=$(tail -1 "$f" 2>/dev/null || echo "N/A")
    printf "%-8s %-8s %s\n" "$mask_str" "2-miss" "$vals"
done
for mask_str in "${THREE_MISS[@]}"; do
    f="$METRIC_DIR/$mask_str/result.txt"
    vals=$(tail -1 "$f" 2>/dev/null || echo "N/A")
    printf "%-8s %-8s %s\n" "$mask_str" "3-miss" "$vals"
done
echo "--------------------------------------------------------------------"
```

---

### 7.4 完整端到端流程

```bash
# ============================================================
# 从训练到评估的完整流程（一键执行）
# ============================================================
set -e

export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/SynDiff__TMI2023
cd $COMPARE_ROOT

# Step 1: 训练
echo "================================"
echo " Step 1/3: 训练 SynDiff"
echo "================================"
bash run_train.sh

# Step 2: 获取 task 时间戳（最新创建的）
TASK_TS=$(ls -t results/task_* 2>/dev/null | head -1 | grep -oP 'task_\K.*')
echo "Task timestamp: $TASK_TS"

# Step 3: 推理 + 评估（eval.py → syn_metric.py）
echo ""
echo "================================"
echo " Step 2/3: 推理（14 种缺失模式）"
echo "================================"
python eval.py \
    --input_path "$DATA_ROOT" \
    --datalist_dir "$DATALIST_DIR" \
    --output_path results \
    --task_ts "$TASK_TS" \
    --ckpt_epoch 200 \
    --gpu 0 \
    --image_size 256 --num_channels 2 --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 \
    --embedding_type positional --z_emb_dim 256 --t_emb_dim 256

echo ""
echo "================================"
echo " Step 3/3: 评估指标"
echo "================================"
bash run_eval.sh "$TASK_TS" --metrics_only

echo ""
echo "================================"
echo " ✅ 全部完成！"
echo " 模型: results/task_$TASK_TS/models/"
echo " 预测: results/task_$TASK_TS/prediction/"
echo "       子目录: 0111 1011 1101 1110 0011 0101 0110 1001 1010 1100 0001 0010 0100 1000"
echo " 指标: results/task_$TASK_TS/prediction_metric_result/"
echo "================================"
```

---

### 7.5 常见问题排查

#### 问题1：CUDA Out of Memory

```bash
# 解决: batch_size 已经是 1 (最小值)
# 如果仍然 OOM，减小 num_channels_dae
python train.py ... --num_channels_dae 32
# 或减少 ch_mult
python train.py ... --ch_mult 1 1 2 2 4
```

#### 问题2：找不到模态文件

```bash
# 检查数据目录结构是否匹配
ls $DATA_ROOT/BraTS20_Training_001/
# 应该包含: *_flair.nii(.gz), *_t1.nii(.gz), *_t1ce.nii(.gz), *_t2.nii(.gz)

# 代码会自动尝试 .nii 和 .nii.gz 两种后缀
```

#### 问题3：TensorBoard 无法启动

```bash
# 安装 tensorboard
pip install tensorboard

# 如果端口被占用:
tensorboard --logdir results/task_*/tensorboard --port 6007
```

#### 问题4：eval.py 找不到 checkpoint

```bash
# 确认 models 目录中有对应 epoch 的权重文件
ls results/task_$TASK_TS/models/

# 文件命名格式: {network_name}_{epoch}.pth
# 例如: gen_diffusive_1_200.pth, gen_non_diffusive_1to2_200.pth

# 如果使用 "final" 作为 epoch，请确保存在对应的 pth 文件
# 默认使用 epoch=200
```

#### 问题5：训练速度过慢

```bash
# num_workers 调整（根据 CPU 核心数，train.py DataLoader 中默认4）
# 减少验证频率（train.py 中默认 epoch % 10）
# 减少保存频率
python train.py ... --save_ckpt_every 100 --save_content_every 100
```

---

## 8. 代码适配清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `dataset.py` | ✅ 已修改 | MONAI 风格 NIfTI 加载，支持 datalist_dir 参数 |
| `train.py` | ✅ 已修改 | 单GPU + TensorBoard + 全局路径变量 |
| `eval.py` | ✅ 已重写 | 仅预测合成，14 mask 模式，NIfTI 保存，无指标耦合 |
| `utils/op/fused_act.py` | ✅ 已修改 | CUDA kernel → 纯 PyTorch fallback |
| `utils/op/upfirdn2d.py` | ✅ 已修改 | CUDA kernel → 纯 PyTorch fallback |
| `run_train.sh` | ✅ 已更新 | 全局路径变量 + 完整参数 |
| `run_eval.sh` | ✅ 已更新 | 两步 pipeline (eval → metrics) |
| `backbones/` | 未修改 | 模型架构保持原样 |

## 9. 已知限制与注意事项

1. **2D vs 3D**: SynDiff 原生为 2D 架构，3D 体积需逐切片处理再拼接。可能导致切片间不一致。
2. **成对翻译**: SynDiff 设计为两两模态翻译 (1→1)，处理多模态缺失需级联推理（选可用模态→生成缺失模态）。
3. **训练随机性**: 训练时随机采样源-目标对，模型需要学习所有 12 种有向模态对。
4. **CUDA 自定义算子**: 已通过纯 PyTorch 替代方案解决。编译失败时会自动降级。
5. **GPU 显存**: batch_size=1 适配 48G 显存，2D 切片处理相对轻量。
6. **移植性**: 所有路径通过全局变量设置，本地电脑和服务器之间只需修改 `$DATA_ROOT`。

---

*报告生成时间: 2026-06-11*
*分支: experiment/SynDiff_adapt*
