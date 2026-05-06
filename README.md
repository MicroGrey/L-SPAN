# L-SPAN 图像超分辨率项目

本项目是一个基于 BasicSR 训练框架整理的单图像超分辨率（Single Image Super-Resolution, SISR）实验仓库，主要实现并训练了 `LSPAN` 网络，用于将低分辨率图像放大到 2 倍分辨率。仓库中同时保留了一个轻量的 `SimpleCNN` baseline，便于调试训练流程和对比模型效果。

当前推荐使用的主模型配置为：

- 模型结构：`basicsr/archs/lspan_arch.py`
- 训练配置：`options/Train/train_LSPAN_x2.yml`
- 测试配置：`options/Test/benchmark_LSPAN_x2.yml`
- 放大倍率：`x2`
- 默认训练数据：DF2K
- 默认测试数据：Set5、Set14、Urban100
- 指标：PSNR、SSIM（Y 通道，`crop_border=2`）

> 注意：数据集、训练产物、测试结果和模型权重体积较大，默认不纳入 Git 版本管理。clone 仓库后需要自行准备 `datasets/` 和 `.pth` 权重文件。

## 项目结构

```text
.
├── basicsr/
│   ├── archs/                 # 网络结构定义
│   │   ├── lspan_arch.py      # LSPAN 主模型
│   │   ├── simple_arch.py     # SimpleCNN baseline
│   │   └── Upsamplers.py      # 上采样模块
│   ├── data/                  # 数据集、采样器、数据增强与 dataloader
│   ├── losses/                # 损失函数
│   ├── metrics/               # PSNR / SSIM 等评价指标
│   ├── models/                # SRModel、优化器、scheduler、保存与验证逻辑
│   ├── utils/                 # 日志、配置解析、图像 IO、分布式工具等
│   ├── train.py               # 训练入口
│   └── test.py                # 测试入口
├── options/
│   ├── Train/                 # 训练配置
│   └── Test/                  # 测试配置
├── utils/
│   └── make_lmdb.py           # 将 DF2K 数据制作成 LMDB 的辅助脚本
├── requirements.txt           # Python 依赖
├── setup.py                   # BasicSR 风格安装脚本
└── README.md
```

运行后会自动生成以下目录：

```text
datasets/       # 数据集，需自行准备
experiments/    # 训练日志、checkpoint、可视化结果
results/        # 测试日志、超分结果图、指标输出
tb_logger/      # TensorBoard 日志
```

这些目录已在 `.gitignore` 中排除。

## 环境准备

建议使用 Conda 创建独立环境。下面以 Python 3.8 为例：

```bash
conda create -n lspan python=3.8 -y
conda activate lspan
```

安装 PyTorch 时请根据本机 CUDA 版本选择对应命令。示例：

```bash
# 示例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

安装项目依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

如果只想直接运行源码，也可以先尝试只安装 `requirements.txt`。但推荐使用 `pip install -e .`，这样 `basicsr` 包会以可编辑模式注册到当前环境中，路径问题更少。

## 数据集准备

默认配置使用成对的 HR/LR 图像。请按如下目录组织数据：

```text
datasets/
├── DF2K/
│   ├── HR/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── LR_bicubic/
│       └── X2/
│           ├── 0001x2.png
│           ├── 0002x2.png
│           └── ...
└── benchmark/
    ├── Set5/
    │   ├── HR/
    │   └── LR_bicubic/X2/
    ├── Set14/
    │   ├── HR/
    │   └── LR_bicubic/X2/
    └── Urban100/
        ├── HR/
        └── LR_bicubic/X2/
```

文件命名规则由配置项 `filename_tmpl: '{}x2'` 决定：如果 HR 文件名是 `0001.png`，对应 LR 文件名应为 `0001x2.png`。

如果你的数据目录不同，需要同步修改配置文件中的：

- `datasets.train.dataroot_gt`
- `datasets.train.dataroot_lq`
- `datasets.val.dataroot_gt`
- `datasets.val.dataroot_lq`
- `datasets.test_*.dataroot_gt`
- `datasets.test_*.dataroot_lq`

## 模型说明

### LSPAN

`LSPAN` 定义在 `basicsr/archs/lspan_arch.py`，主要由以下组件组成：

- `PixelAttention`：像素注意力模块，通过 1x1 卷积和 Sigmoid 生成像素级权重。
- `BSConvU` / `BSConvS` / `DepthWiseConv`：轻量卷积模块，用于降低参数量和计算成本。
- `ESA`：Enhanced Spatial Attention，用于增强空间区域建模能力。
- `ESDB`：核心特征提取块，包含特征蒸馏、残差连接、ESA 与 Pixel Attention。
- `Upsamplers.PixelShuffleDirect`：默认上采样模块，将低分辨率特征映射到超分图像。

当前主配置 `train_LSPAN_x2.yml` 中的网络参数为：

```yaml
network_g:
  type: LSPAN
  num_in_ch: 3
  num_feat: 64
  num_block: 10
  num_out_ch: 3
  upscale: 2
  conv: BSConvU
```

### SimpleCNN Baseline

`SimpleCNN` 定义在 `basicsr/archs/simple_arch.py`，结构更简单，适合确认环境、数据路径、训练循环和指标计算是否正常。对应配置为：

- 训练：`options/Train/train_simpleCNN.yml`
- 测试：`options/Test/test_simpleCNN.yml`

## 训练

### 训练 LSPAN

```bash
python basicsr/train.py -opt options/Train/train_LSPAN_x2.yml
```

如果是从零开始训练，请先把 `options/Train/train_LSPAN_x2.yml` 中的续训路径改为空：

```yaml
path:
  resume_state: ~
```

当前配置保留了课程实验中的续训状态路径。如果本机没有对应的 `.state` 文件，直接运行会因为找不到 checkpoint 而失败。

训练配置中的关键参数：

- `name: test_LSPAN_x2_C64B10_L1_PixelAttention_500k`：实验名，也是输出目录名。
- `scale: 2`：超分倍率。
- `num_gpu: auto`：自动使用可见 GPU 数量。
- `gt_size: 256`：训练时 HR patch 尺寸。
- `batch_size_per_gpu: 8`：单卡 batch size。
- `total_iter: 1000000`：总训练迭代次数。
- `optim_g.type: Adam`：优化器。
- `scheduler.type: CosineAnnealingRestartLR`：学习率调度器。
- `pixel_opt.type: L1Loss`：像素级 L1 损失。
- `val.val_freq: 1000`：每 1000 iter 验证一次。
- `logger.save_checkpoint_freq: 5000`：每 5000 iter 保存一次模型和训练状态。

训练产物默认写入：

```text
experiments/test_LSPAN_x2_C64B10_L1_PixelAttention_500k/
├── models/             # net_g_*.pth
├── training_states/    # *.state，用于断点续训
├── visualization/      # 验证阶段保存的图像
└── train_*.log         # 训练日志
```

### 断点续训

如果要从指定状态续训，可以在配置文件中指定：

```yaml
path:
  resume_state: experiments/test_LSPAN_x2_C64B10_L1_PixelAttention_500k/training_states/375000.state
```

随后运行训练命令即可从该状态恢复。也可以使用自动续训：

```bash
python basicsr/train.py -opt options/Train/train_LSPAN_x2.yml --auto_resume
```

`--auto_resume` 会在当前实验目录的 `training_states/` 中寻找最新的 `.state` 文件。

### 训练 SimpleCNN

```bash
python basicsr/train.py -opt options/Train/train_simpleCNN.yml
```

如果只是检查环境是否可用，建议先跑 SimpleCNN，因为它更轻、更快。

## 测试

### 测试 LSPAN

测试前请先确认 `options/Test/benchmark_LSPAN_x2.yml` 中的权重路径：

```yaml
path:
  pretrain_network_g: experiments/test_LSPAN_x2_C64B10_L1_PixelAttention_500k/models/net_g_375000.pth
  strict_load_g: true
```

当前仓库中的测试配置保留了课程实验机器上的绝对路径。换机器运行时，应改成当前仓库下实际存在的 `.pth` 文件路径，例如上方这种相对路径。

确认权重和 benchmark 数据集都准备好后运行：

```bash
python basicsr/test.py -opt options/Test/benchmark_LSPAN_x2.yml
```

测试结果默认写入：

```text
results/test_LSPAN_x2_C64B10_L1_500k_best/
├── test_*.log
└── visualization/
    ├── Set5/
    ├── Set14/
    └── Urban100/
```

日志中会输出每个测试集的 PSNR / SSIM，超分图像会保存到对应数据集目录下。

### 测试 SimpleCNN

```bash
python basicsr/test.py -opt options/Test/test_simpleCNN.yml
```

同样需要先确认 `pretrain_network_g` 指向实际存在的权重文件。

## 配置文件说明

训练和测试主要通过 YAML 文件控制。常用字段如下：

| 字段 | 作用 |
| --- | --- |
| `name` | 实验名；决定 `experiments/` 或 `results/` 下的输出目录 |
| `model_type` | 模型封装类型；当前使用 `SRModel` |
| `scale` | 超分倍率 |
| `num_gpu` | 使用 GPU 数量；训练可设为 `auto`，CPU 可设为 `0` |
| `datasets` | 数据集路径、类型、增强、dataloader 参数 |
| `network_g` | 生成器网络结构及参数 |
| `path.pretrain_network_g` | 测试或 finetune 时加载的模型权重 |
| `path.resume_state` | 断点续训状态文件；从零训练时设为 `~` |
| `train.optim_g` | 优化器配置 |
| `train.scheduler` | 学习率调度器配置 |
| `train.pixel_opt` | 像素损失配置 |
| `val.metrics` | 验证或测试指标 |
| `logger` | 打印频率、checkpoint 保存频率、TensorBoard / wandb 配置 |

也可以用 `--force_yml` 在命令行临时覆盖已有配置项，例如：

```bash
python basicsr/train.py -opt options/Train/train_simpleCNN.yml --force_yml train:total_iter=1000 logger:print_freq=10
```

## LMDB 数据

仓库提供了 `utils/make_lmdb.py`，可将默认 DF2K 文件夹数据转换为 LMDB：

```bash
python utils/make_lmdb.py
```

脚本默认读取：

```text
datasets/DF2K/HR
datasets/DF2K/LR_bicubic/X2
```

并输出到：

```text
my_lmdbs/
├── DF2K_train_HR.lmdb
└── DF2K_train_LR_bicubic_X2.lmdb
```

LR 图像的 key 会自动去掉文件名末尾的 `x2`，以保证与 HR 图像 key 对齐。

如果要在训练中使用 LMDB，需要将对应 YAML 中的 `dataroot_gt`、`dataroot_lq` 指向 `.lmdb` 目录，并把 `io_backend.type` 改为 `lmdb`。

## 常见问题

### 1. 找不到数据集文件

请检查 YAML 中的 `dataroot_gt` 和 `dataroot_lq` 是否与本机目录一致。默认要求 HR 文件为 `0001.png`，LR 文件为 `0001x2.png`。

### 2. 找不到模型权重

`experiments/` 被 `.gitignore` 排除，clone 仓库后通常不会自带 `.pth` 文件。请先训练模型，或将已有权重放到配置文件指定的位置，并修改 `pretrain_network_g`。

### 3. CUDA 显存不足

可以尝试调小：

- `datasets.train.batch_size_per_gpu`
- `datasets.train.gt_size`
- `network_g.num_feat`
- `network_g.num_block`

测试大图时，可以考虑减小输入尺寸，或在模型测试逻辑中补充 chop / tile 形式的分块推理。

### 4. 想用 CPU 跑

将配置中的 `num_gpu` 改为：

```yaml
num_gpu: 0
```

CPU 可以用于小规模调试，但训练和 benchmark 测试会明显变慢。

### 5. TensorBoard 查看训练曲线

训练时如果 `logger.use_tb_logger: true`，日志会写入 `tb_logger/`：

```bash
tensorboard --logdir tb_logger
```

## 推荐工作流

1. 准备 Conda 环境并安装依赖。
2. 按目录结构准备 DF2K 和 benchmark 数据集。
3. 先运行 `SimpleCNN` 训练或测试，确认环境和数据路径正常。
4. 检查 `train_LSPAN_x2.yml` 中的训练参数和续训路径。
5. 运行 LSPAN 训练。
6. 将最佳或指定 checkpoint 路径写入 `benchmark_LSPAN_x2.yml`。
7. 运行 benchmark 测试，查看 `results/` 中的日志和可视化图像。

## 版本管理说明

当前 `.gitignore` 会排除以下大文件或生成文件：

- `datasets/`
- `experiments/`
- `results/`
- `tb_logger/`
- `wandb/`
- `*.pth`
- `*.png`、`*.jpg`、`*.jpeg` 等图像文件
- `*.lmdb`

因此，提交代码前通常只需要关注源码、配置文件和文档变化。若确实需要共享权重或结果图，建议使用网盘、Release 附件或课程平台单独提交。

## 许可证

本项目保留 BasicSR 风格的工程结构和安装脚本，仓库许可证见 `LICENSE`，为 Apache License 2.0。
