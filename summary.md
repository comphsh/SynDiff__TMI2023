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

## 9. 代码适配指南

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

### 未修改文件
- `backbones/`: 所有模型架构文件保持不变
- `utils/`: EMA 和工具函数保持不变
- `test.py`: 原推理脚本保留

## 10. 已知限制与注意事项

1. **2D vs 3D**: SynDiff 原生为2D架构，3D体积需逐切片处理再拼接。这可能导致切片间不一致性。

2. **成对翻译**: SynDiff 设计为两两模态翻译 (1→1)，处理多模态缺失需级联推理（选可用模态→生成缺失模态）。

3. **训练随机性**: 训练时随机采样源-目标对，模型需要学习所有12种有向模态对。200 epochs 可能不足以充分训练所有对。

4. **CUDA 自定义算子**: `utils/op/` 中的 CUDA 扩展可能需要根据 CUDA 版本重新编译。

5. **GPU 显存**: batch_size=1 适配 48G 显存，2D 切片处理相对轻量。

6. **依赖**: 需要 `nibabel`, `opencv-python`, `tensorboard`, `monai` (仅评估)，原 `tensorflow` 依赖已移除。

## 11. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `dataset.py` | 已修改 | MONAI 风格数据加载 |
| `train.py` | 已修改 | 单GPU + TensorBoard |
| `eval.py` | 新建 | 14模式推理评估 |
| `run_train.sh` | 新建 | 训练启动脚本 |
| `run_eval.sh` | 新建 | 评估启动脚本 |
| `summary.md` | 新建 | 本报告 |
| `git_learning.md` | 新建 | Git 命令记录 |
| `.gitignore` | 已修改 | 排除 results/ 等 |

---
*报告生成时间: 2026-06-04*
*分支: experiment/SynDiff_adapt*
