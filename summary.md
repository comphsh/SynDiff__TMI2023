# SynDiff 代码分析与适配报告

## 1. 方法基本信息

| 属性 | 值 |
|------|-----|
| 方法名称 | SynDiff (Unsupervised Medical Image Translation With Adversarial Diffusion Models) |
| 发表期刊 | IEEE Transactions on Medical Imaging (TMI), Vol. 42, No. 12, Dec. 2023 |
| 作者 | Muzaffer Özbey*, Onat Dalmaz*, Salman UH Dar, Hasan A Bedel, Şaban Özturk, Alper Güngör, Tolga Çukur |
| 机构 | ICON Lab, Bilkent University / Stanford University |
| 开源地址 | https://github.com/icon-lab/SynDiff |
| 许可证 | NVIDIA Source Code License |
| 框架 | PyTorch (>=1.7.1) |

## 2. 核心创新点

SynDiff 提出了一种**无监督对抗扩散模型**用于医学图像跨模态翻译：

1. **对抗扩散模型 (Adversarial Diffusion)**：将 GAN 的对抗训练引入扩散模型的去噪过程。判别器在扩散时间步上判断去噪图像的真假，迫使生成器产生更真实的去噪结果。

2. **双生成器架构**：
   - **扩散生成器 (Diffusive Generator)**：基于 NCSN++ (Score-based) 架构，接受噪声图像 $x_{t+1}$ 和源模态图像作为条件，预测干净图像 $x_0$。
   - **非扩散翻译器 (Non-diffusive Translator)**：基于 ResNet (CycleGAN 风格) 的快速翻译网络，提供初始翻译结果。

3. **快速采样**：使用 4 步扩散过程（而非传统扩散模型的 1000+ 步），大幅加速推理。

4. **后验采样**：在 DDPM 后验分布 $q(x_t | x_0, x_{t+1})$ 中采样，结合扩散模型的条件生成能力。

5. **双向循环一致性**：类似 CycleGAN，同时训练正向和反向翻译，使用循环一致性损失。

## 3. 原代码结构

```
SynDiff__TMI2023/
├── train.py                          # 主训练脚本 (DDP分布式)
├── test.py                           # 推理测试脚本
├── dataset.py                        # 数据集加载 (.mat格式, 2D)
├── backbones/
│   ├── ncsnpp_generator_adagn.py     # NCSN++ 扩散生成器 (AdaGN条件)
│   ├── generator_resnet.py           # ResNet 翻译器 + PatchGAN 判别器
│   ├── discriminator.py              # 时间条件判别器 (Large/Small)
│   ├── layers.py, layerspp.py        # 网络层 (卷积、注意力、上下采样)
│   ├── dense_layer.py                # 全连接层
│   └── up_or_down_sampling.py        # 抗混叠上下采样
├── utils/
│   ├── EMA.py                        # 指数移动平均优化器
│   ├── utils.py                      # TensorFlow checkpoint 工具
│   └── op/                           # CUDA 自定义算子
├── datalist/BraTS2020/               # 数据集划分 (我们添加)
│   ├── train.list (260 subjects)
│   ├── val.list (36 subjects)
│   └── test.list (73 subjects)
├── run_BraTS20_train.sh              # 原训练脚本
└── run_Noise_train.sh                # 原去噪训练脚本
```

## 4. 原数据加载方式

- **格式**: `.mat` 文件 (MATLAB/HDF5)
- **维度**: 2D 切片，形状 `(N, H, W)` 其中 N=切片数
- **预处理**: 每个模态独立存储为 `data_{phase}_{contrast}.mat`
- **归一化**: `(data - 0.5) / 0.5` → 映射到 [-1, 1]
- **模态对**: 一次只加载两个模态 (contrast1, contrast2)

## 5. 原模型架构

### 扩散生成器 (NCSNpp)
- **输入**: 噪声图像 $x_{t+1}$ + 源模态图像 (2通道)
- **输出**: 预测的去噪图像 $x_0$ (1通道)
- **架构**: U-Net 风格编码器-解码器，含 BigGAN 残差块、注意力层
- **条件注入**: AdaGN (Adaptive Group Normalization) + 时间嵌入 + 潜在编码z

### 非扩散翻译器 (ResNet Generator)
- **输入**: 源模态图像 (1通道)
- **输出**: 翻译后的目标模态图像 (1通道)
- **架构**: 6个 ResNet 块的编码器-解码器

### 判别器
- **扩散判别器**: 时间条件判别器，输入 $[x_t, x_{t+1}]$ (2通道)
- **循环判别器**: PatchGAN 判别器 (70×70)

### 损失函数
| 损失 | 权重 | 说明 |
|------|------|------|
| G_adv | 1.0 | 扩散生成器对抗损失 (softplus) |
| G_cycle_adv | 1.0 | 翻译器对抗损失 |
| G_L1 | 0.5 | 扩散预测的 L1 重构损失 |
| G_cycle | 0.5 | 循环一致性 L1 损失 |
| D_real | 1.0 | 判别器真实样本损失 |
| D_fake | 1.0 | 判别器生成样本损失 |
| R1 | 0.05 | R1 梯度惩罚 |

## 6. 原评估方式

- 仅计算 PSNR (skimage) 和 L1 Loss
- 可视化保存为 PNG 图像
- 不涉及 FID/LPIPS/SSIM 等深度特征指标

## 7. 与 MySparseDiffusion 的关键差异

| 维度 | SynDiff | MySparseDiffusion |
|------|---------|-------------------|
| **范式** | 对抗扩散模型 (Adversarial Diffusion) | 稀疏扩散模型 (Sparse Diffusion + MoE) |
| **生成方式** | 条件生成 (源模态→目标模态) | 无条件生成 + 条件引导 |
| **架构** | NCSN++ + ResNet + PatchGAN | VAE + Diffusion + MoE |
| **扩散步数** | 4步 (极快) | 标准 DDPM (1000步) |
| **维度** | 2D (逐切片处理) | 3D (体积处理) |
| **模态数** | 2模态 (双向) | 4模态 (全对全) |
| **训练策略** | GAN对抗训练 + 循环一致性 | VAE重建 + 扩散去噪 |
| **推理速度** | 极快 (4步) | 较慢 (1000步) |
| **图像质量** | 依赖对抗训练，细节较好 | 依赖扩散过程，多样性好 |
| **训练稳定性** | GAN训练不稳定 | 扩散训练较稳定 |
| **GPU显存** | 适中 (2D处理) | 大 (3D处理) |

## 8. 统一实验配置

### 数据集
- **数据源**: BraTS 2020 训练集
- **样本数**: 369例 (训练260 / 验证36 / 测试73)
- **模态**: flair, t1, t1ce, t2 (顺序固定)
- **数据划分**: `datalist/BraTS2020/{train,val,test}.list`

### 缺失模式 (14种)
| mask_id | 模式 | 缺失模态 | 可用模态 |
|---------|------|----------|----------|
| 1  | 0111 | flair | t1+t1ce+t2 |
| 2  | 1011 | t1 | flair+t1ce+t2 |
| 3  | 1101 | t1ce | flair+t1+t2 |
| 4  | 1110 | t2 | flair+t1+t1ce |
| 5  | 0011 | flair+t1 | t1ce+t2 |
| 6  | 0101 | flair+t1ce | t1+t2 |
| 7  | 0110 | flair+t2 | t1+t1ce |
| 8  | 1001 | t1+t1ce | flair+t2 |
| 9  | 1010 | t1+t2 | flair+t1ce |
| 10 | 1100 | t1ce+t2 | flair+t1 |
| 11 | 0001 | flair+t1+t1ce | t2 |
| 12 | 0010 | flair+t1+t2 | t1ce |
| 13 | 0100 | flair+t1ce+t2 | t1 |
| 14 | 1000 | t1+t1ce+t2 | flair |

### 评价指标
- SSIM (结构相似性)
- PSNR (峰值信噪比)
- MSE (均方误差)
- MAE (平均绝对误差)
- LPIPS (学习感知相似度)
- FID (Fréchet Inception Distance, 2.5D)

所有指标通过 `$EVAL_SCRIPT` (syn_metrics.py) 统一计算。

### 训练超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| optimizer | Adam | 沿用官方设置 |
| lr_g | 1.6e-4 | 生成器学习率 |
| lr_d | 1e-4 | 判别器学习率 |
| beta1, beta2 | 0.5, 0.9 | Adam参数 |
| batch_size | 1 | 显存限制 (48G RTX 8000) |
| num_epoch | 200 | 统一训练轮数 |
| lr_scheduler | CosineAnnealingLR | 余弦退火到 1e-5 |
| num_timesteps | 4 | 扩散步数 |
| image_size | 256 | 输入图像尺寸 |

## 9. 代码适配清单

### 已完成适配

1. **数据加载 (dataset.py)** ✅
   - 替换 `.mat` → `.nii/.nii.gz` 加载
   - 按患者ID组织，从 nifti 提取 2D 切片
   - 百分位归一化 (0-99.5%)
   - 随机源-目标模态对采样
   - 保持 `CreateDatasetSynthesis` API 兼容

2. **训练 (train.py)** ✅
   - 移除 DDP 分布式训练 → 单GPU训练
   - 添加 TensorBoard 日志 (epoch loss/lr)
   - 模型保存至 `results/task_{timestamp}/models/`
   - 默认 200 epochs
   - 保持原始架构和损失函数

3. **推理 (eval.py)** ✅ (新建)
   - 对测试集逐样本生成14种缺失组合
   - 切片级推理 → 3D体积重建
   - 保存输入、真值、预测图像
   - 自动调用 syn_metrics.py 计算指标

4. **启动脚本** ✅
   - `run_train.sh`: 一键训练
   - `run_eval.sh`: 一键评估

5. **CUDA 兼容性修复** ✅
   - `utils/op/fused_act.py`: 自定义 CUDA kernel 改为纯 PyTorch `F.leaky_relu` 实现
   - `utils/op/upfirdn2d.py`: 自定义 CUDA kernel 改为纯 PyTorch `upfirdn2d_native` 实现
   - 两处均包裹 try-except，编译失败时自动降级

### 未修改文件
- `backbones/`: 所有模型架构文件保持不变
- `utils/EMA.py`: 保持不变
- `test.py`: 原推理脚本保留（不再使用）

---

## 10. 完整操作指南

### 10.1 环境检查

在开始之前，确认以下条件满足：

```bash
# ===== 1. 检查 GPU 和 CUDA =====
nvidia-smi
# 预期: Quadro RTX 8000 48G 或 RTX 5090

# ===== 2. 检查 Python 环境 =====
python3 --version
# 预期: Python 3.10+

python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# 预期: PyTorch 2.7.0+cu128 或类似版本

# ===== 3. 检查依赖库 =====
python3 -c "import numpy; import nibabel; import cv2; print('numpy, nibabel, cv2: OK')"
python3 -c "import torch.utils.tensorboard; print('tensorboard: OK')"

# ===== 4. 检查数据集路径 =====
DATA_ROOT="/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
ls "$DATA_ROOT" | head -5
# 预期: BraTS20_Training_001  BraTS20_Training_002 ...

ls "$DATA_ROOT/BraTS20_Training_001/"
# 预期: BraTS20_Training_001_flair.nii  BraTS20_Training_001_t1.nii  ...

# ===== 5. 检查 datalist =====
wc -l datalist/BraTS2020/*.list
# 预期: train.list=260, val.list=36, test.list=73
```

### 10.2 快速语法验证（不加载数据，5秒完成）

```bash
# 验证所有模块能正常导入和模型前向传播
python3 -c "
import sys; sys.path.insert(0, '.')
import torch

# 1. 模块导入测试
from backbones.ncsnpp_generator_adagn import NCSNpp
import backbones.generator_resnet
from backbones.discriminator import Discriminator_large
from dataset import load_patient_ids, normalize_volume, MODALITY_ORDER
print('1. All imports: OK')

# 2. Datalist 加载测试
for phase in ['train', 'val', 'test']:
    ids = load_patient_ids(phase, './datalist/BraTS2020')
    print(f'2. {phase}: {len(ids)} patients')

# 3. 模型构建 + 前向传播测试
class Args:
    seed=1024; image_size=256; num_channels=2; num_channels_dae=64
    ch_mult=[1,1,2,2,4,4]; num_res_blocks=2; attn_resolutions=(16,)
    dropout=0.; resamp_with_conv=True; conditional=True; fir=True
    fir_kernel=[1,3,3,1]; skip_rescale=True; resblock_type='biggan'
    progressive='none'; progressive_input='residual'; progressive_combine='sum'
    embedding_type='positional'; fourier_scale=16.; not_use_tanh=False
    centered=True; nz=100; n_mlp=3; z_emb_dim=256; t_emb_dim=256
    ngf=64; use_geometric=False

args = Args()
device = torch.device('cuda:0')

gen = NCSNpp(args).to(device)
x = torch.cat([torch.randn(1,1,256,256), torch.randn(1,1,256,256)], dim=1).to(device)
out = gen(x, torch.randint(0,4,(1,)).to(device), torch.randn(1,100).to(device))
print(f'3. NCSNpp forward: {x.shape} -> {out.shape}')

disc = Discriminator_large(nc=2, ngf=64, t_emb_dim=256, act=torch.nn.LeakyReLU(0.2)).to(device)
d_out = disc(x, torch.randint(0,4,(1,)).to(device), x)
print(f'4. Discriminator forward: -> {d_out.shape}')

args.num_channels = 1
trans = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[0])
t_out = trans(torch.randn(1,1,256,256).to(device))
args.num_channels = 2
print(f'5. Translator forward: -> {t_out.shape}')

print('ALL CHECKS PASSED - Ready to train!')
"
```

如果看到 `ALL CHECKS PASSED`，说明代码和环境就绪。

---

### 10.3 训练

#### 一键启动（推荐）

```bash
# 进入项目目录
cd /devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/SynDiff__TMI2023

# 启动训练
bash run_train.sh
```

#### 手动命令（完整参数）

```bash
python3 train.py \
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
    --r1_gamma 0.05 \
    --z_emb_dim 256 \
    --lr_d 1e-4 \
    --lr_g 1.6e-4 \
    --lazy_reg 10 \
    --save_content \
    --save_content_every 50 \
    --save_ckpt_every 50 \
    --lambda_l1_loss 0.5 \
    --contrast1 T1 \
    --contrast2 T2 \
    --exp BraTS20_syndiff \
    --input_path /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --output_path ./results
```

#### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epoch` | 200 | 训练轮数，统一实验框架要求 |
| `--batch_size` | 1 | 若48G显存不足可保持1 |
| `--lr_g` | 1.6e-4 | 生成器学习率（官方推荐） |
| `--lr_d` | 1e-4 | 判别器学习率（官方推荐） |
| `--num_timesteps` | 4 | 扩散步数，越大质量越好但越慢 |
| `--num_channels_dae` | 64 | 扩散生成器通道数 |
| `--save_ckpt_every` | 50 | 每隔N个epoch保存模型权重 |
| `--save_content_every` | 50 | 每隔N个epoch保存完整checkpoint |
| `--lambda_l1_loss` | 0.5 | L1损失权重 |
| `--r1_gamma` | 0.05 | R1梯度惩罚系数 |
| `--lazy_reg` | 10 | 每N步计算一次R1惩罚（节省显存） |

#### 训练监控

```bash
# 终端1：启动 TensorBoard
tensorboard --logdir results/task_*/tensorboard --port 6006

# 终端2：浏览器打开 http://localhost:6006
# 查看曲线: epoch/G_total, epoch/D_total, epoch/lr_g, epoch/lr_d 等

# 终端3：查看实时 GPU 使用情况
watch -n 1 nvidia-smi

# 查看最新日志输出
tail -f results/task_*/outputs/log.txt 2>/dev/null || ls results/task_*/
```

#### 训练输出目录结构

```
results/task_20260605_143022/
├── models/                          # 模型保存目录
│   ├── gen_diffusive_1_50.pth       # 扩散生成器1 (epoch 50)
│   ├── gen_diffusive_1_100.pth      # 扩散生成器1 (epoch 100)
│   ├── gen_diffusive_1_150.pth
│   ├── gen_diffusive_1_200.pth      # 最终模型
│   ├── gen_diffusive_2_50.pth       # 扩散生成器2
│   ├── gen_diffusive_2_200.pth
│   ├── gen_non_diffusive_1to2_50.pth  # 翻译器 1→2
│   ├── gen_non_diffusive_1to2_200.pth
│   ├── gen_non_diffusive_2to1_50.pth  # 翻译器 2→1
│   ├── gen_non_diffusive_2to1_200.pth
│   └── content.pth                  # 完整checkpoint（可resume）
├── tensorboard/                     # TensorBoard 日志
│   └── events.out.tfevents.xxx
├── train.py                         # 训练脚本备份
├── xpos1_epoch_10.png               # 训练样本可视化
├── sample1_epoch_10.png
└── ...
```

#### 训练时间预估

- 每 epoch 约 8-15 分钟（取决于 slice 数量和 GPU）
- 200 epochs 总计约 **27-50 小时**
- 建议使用 `nohup` 或 `screen`/`tmux` 在后台运行：

```bash
# 后台运行（推荐）
nohup bash run_train.sh > train.log 2>&1 &

# 查看进度
tail -f train.log

# 或使用 screen
screen -S syndiff_train
bash run_train.sh
# Ctrl+A, D 分离
# screen -r syndiff_train 重新连接
```

#### 断点续训

```bash
# 如果训练中断，添加 --resume 继续训练
# 注意：需要修改 train.py 中的 checkpoint 加载路径
# 或手动指定 content.pth 路径
```

---

### 10.4 推理与评估

#### 一键评估（推荐）

```bash
# 替换为实际的 task 时间戳
TASK_TS="20260605_143022"
bash run_eval.sh $TASK_TS

# 也可以指定 checkpoint epoch（默认200）
bash run_eval.sh $TASK_TS 200
```

#### 手动命令（完整参数）

```bash
TASK_TS="20260605_143022"
WHICH_EPOCH=200

python3 eval.py \
    --image_size 256 \
    --num_channels 2 \
    --num_channels_dae 64 \
    --ch_mult 1 1 2 2 4 4 \
    --num_timesteps 4 \
    --num_res_blocks 2 \
    --batch_size 1 \
    --embedding_type positional \
    --z_emb_dim 256 \
    --ngf 64 \
    --ckpt_path "./results/task_${TASK_TS}" \
    --which_epoch ${WHICH_EPOCH} \
    --gpu_chose 0 \
    --input_path /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
    --eval_script /devdata2/hsh/program/python/methods/MySparseDiffusion/my_sparse_diff-moe-006/scripts/_01_vae/metrics/syn_metrics.py
```

#### 评估输出目录结构

```
results/task_20260605_143022/
├── prediction/                       # 预测图像目录
│   ├── 1/                            # mask_id=1 (0111: flair缺失)
│   │   ├── BraTS20_Training_137_flair_syn.nii.gz    # flair 合成图
│   │   ├── BraTS20_Training_137_t1_input.nii.gz      # t1 输入（可用模态）
│   │   ├── BraTS20_Training_137_t1ce_input.nii.gz    # t1ce 输入
│   │   ├── BraTS20_Training_137_t2_input.nii.gz      # t2 输入
│   │   ├── BraTS20_Training_137_flair_gt.nii.gz      # flair 真值
│   │   └── ...
│   ├── 2/                            # mask_id=2 (1011: t1缺失)
│   ├── ...
│   └── 14/                           # mask_id=14 (1000: t1+t1ce+t2缺失)
├── prediction_metric_result/         # 指标结果目录
│   ├── 1/
│   │   └── result.txt                # mask_id=1 的 SSIM/PSNR/MSE/MAE/LPIPS/FID
│   ├── 2/
│   │   └── result.txt
│   └── ...
│       └── result.txt
└── (models/, tensorboard/ 等同上)
```

#### 结果文件格式 (result.txt)

```
ssim     psnr     mse      mae      fid     lpips
0.876543   28.123456  0.001234  0.023456  45.678901   0.123456
```

#### 推理时间预估

- 测试集: 73 个患者 × 14 种 mask × ~155 slices = ~158K 次推理
- 每次推理（4步扩散）约 0.05 秒
- 总计约 **2-3 小时**

---

### 10.5 查看和汇总所有 mask 的评估结果

```bash
# 汇总所有 14 种 mask 的指标
TASK_TS="20260605_143022"
echo "mask_id | SSIM   | PSNR    | MSE      | MAE      | LPIPS   | FID"
echo "--------|--------|---------|----------|----------|---------|------"
for i in $(seq 1 14); do
    result_file="results/task_${TASK_TS}/prediction_metric_result/${i}/result.txt"
    if [ -f "$result_file" ]; then
        # 读取第二行（数值行）
        values=$(sed -n '2p' "$result_file")
        printf "  %2d    | %s\n" "$i" "$values"
    else
        echo "  $i    | (pending)"
    fi
done
```

---

### 10.6 超参数调优指南

| 场景 | 修改参数 | 建议值 |
|------|----------|--------|
| 显存不足（OOM） | `--batch_size` | 保持 1（已是最小值） |
| | `--num_channels_dae` | 降为 32 |
| | `--ch_mult` | 改为 `1 1 2 2 4` |
| 训练不稳定（Loss震荡） | `--r1_gamma` | 增加至 1.0 |
| | `--lazy_reg` | 取消（设为None）每步都计算R1 |
| 生成图像模糊 | `--num_timesteps` | 增至 8 |
| | `--lambda_l1_loss` | 降为 0.1 |
| 生成图像噪声多 | `--lambda_l1_loss` | 增至 1.0 |
| 训练太慢 | `--num_epoch` | 减为 100 |
| | `--lazy_reg` | 增至 20 |

---

## 11. 常见问题排查

### 11.1 CUDA kernel 编译警告

训练启动时会看到以下消息，这是**正常的**：

```
[SynDiff] Custom CUDA kernel 'fused' failed to compile: ...
[SynDiff] Using pure PyTorch fallback (F.leaky_relu) instead.
```

原因：PyTorch 2.7 + CUDA 12.8 的 C++ API 与原 StyleGAN2 kernel 不兼容。
影响：无。已用纯 PyTorch 实现替代，功能等价、GPU 加速。

### 11.2 训练中 Loss 为 NaN

```bash
# 可能原因及解决方案：
# 1. 学习率过高 → 降低 lr_g 到 1e-4
# 2. 尝试添加梯度裁剪（train.py 中搜索 loss.backward 位置添加）
# 3. 检查数据是否有异常值
```

### 11.3 推理时 OOM

```bash
# eval.py 逐切片处理，理论上不应 OOM
# 如遇到，检查是否有其他进程占用 GPU：
nvidia-smi
# 清理 GPU 缓存：
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 11.4 数据加载报 FileNotFoundError

```bash
# 确认数据路径正确
ls /devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/

# 确认 datalist 中的 patient ID 能在数据目录中找到
while read pid; do
  [ -d "/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/$pid" ] || echo "MISSING: $pid"
done < datalist/BraTS2020/train.list
```

---

## 12. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `dataset.py` | 已修改 | MONAI 风格数据加载 |
| `train.py` | 已修改 | 单GPU + TensorBoard |
| `eval.py` | 新建 | 14模式推理评估 |
| `utils/op/fused_act.py` | 已修改 | CUDA kernel → 纯 PyTorch fallback |
| `utils/op/upfirdn2d.py` | 已修改 | CUDA kernel → 纯 PyTorch fallback |
| `run_train.sh` | 新建 | 训练启动脚本 |
| `run_eval.sh` | 新建 | 评估启动脚本 |
| `summary.md` | 新建 | 本报告 |
| `git_learning.md` | 新建 | Git 命令记录 |
| `.gitignore` | 已修改 | 排除 results/ 等 |

## 13. 已知限制与注意事项

1. **2D vs 3D**: SynDiff 原生为2D架构，3D体积需逐切片处理再拼接。这可能导致切片间不一致性。
2. **成对翻译**: SynDiff 设计为两两模态翻译 (1→1)，处理多模态缺失需级联推理（选可用模态→生成缺失模态）。
3. **训练随机性**: 训练时随机采样源-目标对，模型需要学习所有12种有向模态对。200 epochs 可能不足以充分训练所有对。
4. **CUDA 自定义算子**: 已通过纯 PyTorch 替代方案解决。编译失败时会自动降级，不影响训练和推理。
5. **GPU 显存**: batch_size=1 适配 48G 显存，2D 切片处理相对轻量。
6. **依赖**: 需要 `nibabel`, `opencv-python`, `tensorboard`, `monai` (仅评估)，原 `tensorflow` 依赖已移除。

---
*报告生成时间: 2026-06-05*
*分支: experiment/SynDiff_adapt*
