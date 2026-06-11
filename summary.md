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

1. **对抗扩散模型**：GAN 对抗训练引入扩散模型去噪过程，4 步快速采样
2. **双生成器架构**：NCSN++ 扩散生成器 + ResNet 非扩散翻译器
3. **后验采样**：DDPM 后验分布 q(xt|x0, xt+1) 采样
4. **双向循环一致性**：CycleGAN 风格双向翻译

## 3. 代码结构

```
SynDiff__TMI2023/
├── backbones/                       # 模型骨干
│   ├── ncsnpp_generator_adagn.py    # NCSN++ 扩散生成器
│   ├── generator_resnet.py          # ResNet 翻译器 + PatchGAN 判别器
│   ├── discriminator.py             # 时间条件判别器
│   └── ...
├── datalist/BraTS2020/              # 数据划分 (70/10/20)
│   ├── train.list (260), val.list (36), test.list (73)
├── dataset.py                        # NIfTI 数据加载
├── train.py                          # 训练 (单GPU)
├── eval.py                           # 推理预测 (仅合成图像, 不计算指标)
├── run_train.sh                      # 训练启动
├── run_eval.sh                       # 推理启动
└── summary.md                        # 本报告
```

## 4. 统一实验配置

### 数据集
- BraTS 2020: 369 例 (train 260 / val 36 / test 73)
- 数据路径: `$DATA_ROOT/{patient_name}/{patient_name}_{modal}.nii(.gz)`
- 模态顺序: `[flair, t1, t1ce, t2]`

### 缺失模式 (14 种, D2Diff 顺序)
| mask | 类型 | 可用模态 | 缺失数 |
|------|------|---------|--------|
| 0001 | 3-miss | t2 | 3 |
| 0010 | 3-miss | t1ce | 3 |
| 0011 | 2-miss | t1ce, t2 | 2 |
| 0100 | 3-miss | t1 | 3 |
| 0101 | 2-miss | t1, t2 | 2 |
| 0110 | 2-miss | t1, t1ce | 2 |
| 0111 | 1-miss | t1, t1ce, t2 | 1 (缺 flair) |
| 1000 | 3-miss | flair | 3 |
| 1001 | 2-miss | flair, t2 | 2 |
| 1010 | 2-miss | flair, t1ce | 2 |
| 1011 | 1-miss | flair, t1ce, t2 | 1 (缺 t1) |
| 1100 | 2-miss | flair, t1 | 2 |
| 1101 | 1-miss | flair, t1, t2 | 1 (缺 t1ce) |
| 1110 | 1-miss | flair, t1, t1ce | 1 (缺 t2) |

> mask 编码: 4位二进制 `[flair, t1, t1ce, t2]`，1=可用，0=缺失

### 评价指标
- SSIM, PSNR, MSE, MAE, LPIPS, FID
- 由用户全局统一评估脚本计算（不内嵌在 eval.py 中）

### 训练超参数
- optimizer: Adam (lr_g=1.6e-4, lr_d=1e-4, beta1=0.5, beta2=0.9)
- batch_size=1, num_epoch=200
- num_timesteps=4, num_channels_dae=64, ch_mult=[1,1,2,2,4,4]
- lr_scheduler: CosineAnnealingLR (eta_min=1e-5)
- loss: lambda_l1_loss=0.5, r1_gamma=0.05

---

## 5. 完整执行命令指南

### 5.0 环境准备

```bash
# ============================================================
# 全局路径变量（本地电脑/服务器按实际修改）
# ============================================================
export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/SynDiff__TMI2023
export DATA_ROOT=/devdata/hsh/datasets/seg_dataset/BraTS2020/brats20-dataset-training-validation/versions/1/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
export DATALIST_DIR=$COMPARE_ROOT/datalist/BraTS2020

# ============================================================
# 切换到实验分支
# ============================================================
cd $COMPARE_ROOT
git checkout experiment/SynDiff_adapt

# ============================================================
# 检查环境
# ============================================================
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
print(f'GPU:     {torch.cuda.get_device_name(0)}')
"

# ============================================================
# 验证数据
# ============================================================
ls $DATA_ROOT/BraTS20_Training_001/
echo "Train: $(wc -l < $DATALIST_DIR/train.list)"
echo "Val:   $(wc -l < $DATALIST_DIR/val.list)"
echo "Test:  $(wc -l < $DATALIST_DIR/test.list)"

# ============================================================
# 快速冒烟测试 (约2分钟)
# ============================================================
python3 -c "
import sys; sys.path.insert(0, '.')
import torch, torch.nn as nn
from dataset import BraTSDataset2D
from backbones.ncsnpp_generator_adagn import NCSNpp
from backbones.discriminator import Discriminator_large
import backbones.generator_resnet

# 数据加载
ds = BraTSDataset2D('train', data_root='$DATA_ROOT', datalist_dir='$DATALIST_DIR', random_mask=True)
src, tgt = ds[0]
print(f'Data OK: source={src.shape}')

# 模型前向
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
print(f'Gen OK: {x.shape} -> {out.shape}')

disc = Discriminator_large(nc=2, ngf=64, t_emb_dim=256, act=nn.LeakyReLU(0.2)).to(device)
d_out = disc(x, torch.randint(0,4,(1,)).to(device), x)
print(f'Disc OK: -> {d_out.shape}')

args.num_channels = 1
trans = backbones.generator_resnet.define_G(netG='resnet_6blocks', gpu_ids=[0])
t_out = trans(torch.randn(1,1,256,256).to(device))
print(f'Trans OK: -> {t_out.shape}')
print('🎉 ALL CHECKS PASSED')
"
```

### 5.1 训练

#### 方式一：一键启动

```bash
cd $COMPARE_ROOT
bash run_train.sh
```

#### 方式二：手动命令

```bash
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

#### 训练日志格式

```
Epoch [0/200], Step [100/12345], Global Step: 100
  G-Cycle: 0.1234, G-L1: 0.0567, G-Adv: 1.2345, G-cycle-Adv: 0.2345, G-Sum: 1.6480, D: 1.0000, D-cycle: 0.5678
...
=== End of Epoch [0/200], Total Steps: 12345 ===
  G-Cycle: 0.1200, G-L1: 0.0550, G-Adv: 1.2000, G-cycle-Adv: 0.2300
  G-Total: 1.6050, D: 0.9800, D-cycle: 0.5500
  LR_G: 1.60e-04, LR_D: 1.00e-04
```

#### 训练输出

```
results/task_20260605_143022/
├── models/
│   ├── gen_diffusive_1_200.pth
│   ├── gen_diffusive_2_200.pth
│   ├── gen_non_diffusive_1to2_200.pth
│   ├── gen_non_diffusive_2to1_200.pth
│   └── content.pth
├── tensorboard/
│   └── events.out.tfevents.*
├── train.py                    # 脚本备份
└── xpos1_epoch_10.png          # 样本可视化
```

#### 监控

```bash
tensorboard --logdir $COMPARE_ROOT/results/task_*/tensorboard --port 6006
watch -n 1 nvidia-smi

# 后台运行
nohup bash run_train.sh > train.log 2>&1 &
tail -f train.log
```

---

### 5.2 推理 (eval.py — 仅合成图像)

> eval.py 只生成合成图像，不保存 GT / 输入，不计算指标。
> 全局唯一的 `$DATA_ROOT` 作为 GT 来源。

#### 方式一：一键启动

```bash
cd $COMPARE_ROOT
bash run_eval.sh 20260605_143022        # 默认 epoch 200
bash run_eval.sh 20260605_143022 200    # 指定 epoch
```

#### 方式二：手动命令

```bash
TASK_TS="20260605_143022"

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

#### 推理输出结构

```
results/task_20260605_143022/
├── prediction/
│   ├── 0001/                              # 仅 t2 可用 (3-miss)
│   │   ├── BraTS20_Training_137/
│   │   │   ├── BraTS20_Training_137_flair_syn.nii.gz
│   │   │   ├── BraTS20_Training_137_t1_syn.nii.gz
│   │   │   └── BraTS20_Training_137_t1ce_syn.nii.gz
│   │   └── BraTS20_Training_138/ ...
│   ├── 0010/                              # 仅 t1ce 可用
│   ├── 0011/                              # t1ce + t2 (2-miss)
│   ├── 0100/                              # 仅 t1 可用
│   ├── 0101/                              # t1 + t2 (2-miss)
│   ├── 0110/                              # t1 + t1ce (2-miss)
│   ├── 0111/                              # flair 缺失 (1-miss)
│   ├── 1000/                              # 仅 flair 可用
│   ├── 1001/                              # flair + t2 (2-miss)
│   ├── 1010/                              # flair + t1ce (2-miss)
│   ├── 1011/                              # t1 缺失 (1-miss)
│   ├── 1100/                              # flair + t1 (2-miss)
│   ├── 1101/                              # t1ce 缺失 (1-miss)
│   └── 1110/                              # t2 缺失 (1-miss)
```

> 每个 mask 子目录下按 patient_id 分组，只保存 `*_syn.nii.gz`（合成图像）。
> GT 和输入图像均来自 `$DATA_ROOT`，不再重复保存。

---

### 5.3 完整端到端流程

```bash
set -e

export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/SynDiff__TMI2023
export DATA_ROOT=/devdata/hsh/datasets/seg_dataset/BraTS2020/.../MICCAI_BraTS2020_TrainingData
cd $COMPARE_ROOT

# Step 1: 训练
echo "========== Step 1/2: Training =========="
bash run_train.sh

# Step 2: 推理
TASK_TS=$(ls -t results/task_* | head -1 | grep -oP 'task_\K.*')
echo "========== Step 2/2: Inference (task=$TASK_TS) =========="
bash run_eval.sh "$TASK_TS"

echo "========== Done! =========="
echo "Predictions: results/task_$TASK_TS/prediction/"
```

---

## 6. 代码适配清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `dataset.py` | ✅ | NIfTI 加载, D2Diff mask 顺序, datalist_dir 参数 |
| `train.py` | ✅ | 单GPU, TensorBoard, Epoch/Step/Global Step 日志, 全局路径 |
| `eval.py` | ✅ | 仅预测合成, 只存 syn, 不存 GT/input, 不计算指标 |
| `utils/op/fused_act.py` | ✅ | CUDA kernel → PyTorch fallback |
| `utils/op/upfirdn2d.py` | ✅ | CUDA kernel → PyTorch fallback |
| `run_train.sh` | ✅ | 全局路径变量 |
| `run_eval.sh` | ✅ | 全局路径变量, 仅推理无指标 |

## 7. 移植指南 (本地电脑 → 服务器)

只需修改 3 个环境变量:

```bash
# 本地电脑:
export COMPARE_ROOT=/home/user/projects/SynDiff__TMI2023
export DATA_ROOT=/home/user/data/BraTS2020

# 服务器:
export COMPARE_ROOT=/devdata2/hsh/program/python/methods/contrast_method_selected_of_diff_moe_synthesis/SynDiff__TMI2023
export DATA_ROOT=/devdata/hsh/datasets/seg_dataset/BraTS2020/.../MICCAI_BraTS2020_TrainingData
```

`run_train.sh` 和 `run_eval.sh` 中的 `COMPARE_ROOT` 自动从脚本位置推导，只需修改 `DATA_ROOT`。

## 8. 常见问题

### OOM
```bash
# 减小 num_channels_dae
python train.py ... --num_channels_dae 32
```

### 找不到模态文件
```bash
# 代码自动尝试 .nii 和 .nii.gz
ls $DATA_ROOT/BraTS20_Training_001/
```

### 训练慢
```bash
# 减少保存频率
python train.py ... --save_ckpt_every 100 --save_content_every 100
```

## 9. 已知限制

1. **2D 架构**: 3D 体积逐切片处理，切片间可能有不一致
2. **成对翻译**: 设计为 1→1 模态翻译，多模态缺失需级联推理
3. **训练随机性**: 随机采样源-目标对，200 epochs 可能不足以充分训练所有对
4. **CUDA 算子**: 已降级为纯 PyTorch，功能等价

---

*报告时间: 2026-06-11*
*分支: experiment/SynDiff_adapt*
