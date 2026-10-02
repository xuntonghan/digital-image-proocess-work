# 数字图像处理课程实习

本仓库用于完成课程实习中的三个任务：低光 RAW 图像增强、文档几何校正和图像压缩。当前首先实现图像增强模块。

## 当前范围

第一阶段提供：

- SIED 风格无头 `uint16` RAW 的严格读取和尺寸校验；
- 黑电平校正、归一化和 RGGB Bayer 四通道打包；
- 不依赖训练的传统 ISP 基线；
- 从四通道 RAW 到全分辨率 sRGB 的轻量 Residual U-Net；
- L1/SSIM/色度组合损失、Hann 羽化重叠分块推理；
- PSNR、SSIM 和可选 LPIPS；
- 数据审计、训练、推理及 CSV 指标导出命令；
- 验证集驱动的自动曝光/灰世界白平衡传统基线；
- RAW patch 内存映射及可选 GT 缓存，避免云端重复读取整张4K图；
- Canon 与 Sony 的独立 4090 训练配置。

项目不包含课程原始数据、模型权重或完整中间输出。它们分别放在 `data/`、`checkpoints/` 和 `outputs/`，这些目录不会提交到 Git。为便于验收，仓库保留最终逐图指标、汇总指标和 Canon/Sony 代表性四列对比图，见 `reports/assets/enhancement/`。

## 本机安装

本机开发使用 Python 3.10-3.12，不使用当前系统的 Python 3.14，也不安装官方 SIED 仓库中混合了 CUDA、TensorFlow 和旧版 NumPy 的完整依赖文件。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
pytest
```

需要在本机测试神经网络代码时：

```bash
python -m pip install '.[train,dev]'
pytest
```

## 数据清单

每行包含一个低光 RAW 相对路径和一个 GT RGB 相对路径，以空格分隔。路径含空格时使用 shell 风格引号：

```text
Canon/0.0001/1.raw Canon/gt/1.jpg
Canon/0.0001/2.raw Canon/gt/2.jpg
0.001/1.raw 'gt 2/1.jpg'
```

路径相对于数据根目录解析。训练集、验证集和测试集必须使用不同清单，避免评价泄漏。详见 `manifests/README.md`。

## 本机任务

数据先审计，不立即训练：

```bash
dip-audit \
  --data-root /path/to/SIED_dataset \
  --manifest /path/to/test_canon_0.0001.txt \
  --width 3840 --height 2160
```

如果老师没有提供固定划分，使用一次性、可复现的划分命令：

```bash
dip-split \
  --data-root /path/to/SIED_dataset \
  --manifest /path/to/all_canon.txt \
  --output-dir manifests/generated \
  --prefix canon --train-ratio 0.8 --val-ratio 0.1 --seed 20260930
```

划分结果和源清单 SHA256 会记录在 `canon_split.json`，之后训练始终使用生成的固定清单。

本课程下载的1-1000号子集能够完整映射到SIED官方训练/测试清单。本项目优先使用 `dip-official-split` 保留官方测试集，再从官方训练集固定抽取10%作为验证集。最终报告同时给出官方测试集指标和课件要求的1000张全量处理统计，并明确区分两者。

运行已经过验证集选参的自动传统基线（以下为 Canon）：

```bash
dip-classical \
  --data-root /path/to/SIED_dataset \
  --manifest /path/to/test_canon_0.0001.txt \
  --output-dir outputs/classical_canon \
  --auto --gamma 2.0 --auto-percentile 98 --auto-target 0.9
```

## 4090 云端训练

云机使用独立目录 `/root/autodl-tmp/cv-homework` 和独立虚拟环境，不复用 Gala 项目的 ComfyUI 环境。

```bash
dip-train \
  --config configs/enhancement/canon.toml \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --train-manifest /root/autodl-tmp/cv-homework/manifests/train_canon.txt \
  --val-manifest /root/autodl-tmp/cv-homework/manifests/val_canon.txt \
  --gt-cache-dir /root/autodl-tmp/cv-homework/cache/canon_gt
```

训练必须按顺序通过：数据审计、100 步冒烟测试、5000 步试验和验证集早停。若5000步内验证集已经回落，则采用验证损失最低的检查点，不为追求步数继续过拟合。

## 完整图像推理

```bash
dip-infer \
  --checkpoint checkpoints/canon/best.pt \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --manifest /root/autodl-tmp/cv-homework/manifests/test_canon.txt \
  --output-dir outputs/unet_canon \
  --tile-size 256 --overlap 32 --lpips
```

输出目录包含增强图像、逐图指标 `metrics.csv` 和汇总 `summary.json`。

最终实验结论见 `reports/enhancement_final_report.md`，本机基线记录见 `reports/enhancement_local_baseline.md`，完整4090执行顺序见 `cloud/runbook.md`。
