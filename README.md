# 数字图像处理课程实习

本仓库用于完成课程实习中的三个任务：低光 RAW 图像增强、文档几何校正和图像压缩。

## 综合报告与复现入口

- 最终方法、4090 结果、可视化和逐步复现：`reports/final_experiment_report.md`
- 前期完整实验记录：`reports/digital_image_processing_course_report.md`
- 三个项目的下一轮改进与 4090 训练方案：`reports/next_iteration_plan.md`
- 可编辑 Word 报告：`reports/deliverables/数字图像处理课程实习最终实验报告.docx`
- 可直接提交的 PDF 报告：`output/pdf/数字图像处理课程实习最终实验报告.pdf`
- 数据目录、版本和模型权重 SHA256：`DATA_AND_WEIGHTS.md`
- 权重分发边界：`docs/CHECKPOINTS.md`
- RTX 4090 环境与执行顺序：`cloud/runbook.md`

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
- ICCV 2025 SIED 作者预训练权重的盲推理适配器：只以低照 RAW 生成结果，GT 仅在预测完成后参与评价；
- 冻结 SIED 后的轻量曝光/色调精修器，支持全场景缓存训练与原尺寸推理；
- ΔE2000、亮度 MAE、饱和度 MAE、环形色相误差及前后五列对比图。
- Kodak 24 张无损 RGB 图像的固定上游版本、SHA256 清单与压缩实验数据目录。
- GeoCorrection 200 对图像/页面遮罩的严格审计和逐文件 SHA256 清单；
- 基于页面遮罩的四角单应性、边界展平与文本方向自动校正传统基线；
- PageMaskUNet 页面分割和预测遮罩驱动的完全自动几何纠正；
- 无平整真值场景下的页面边界直线度、文本行倾斜度和前后四列对比。
- DCTC v2 快速码流和 CompressAI Cheng2020 真实熵码流基准。

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

调用 SIED 作者模型时只安装最小依赖，不安装其混合了旧 CUDA、TensorFlow 与旧 NumPy 的完整 `requirements.txt`：

```bash
python -m pip install '.[sied]'
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

## ICCV 2025 SIED 官方模型

将作者仓库固定到已验证提交 `fb7af94ac831e91c647415bf1ca0b1a75931833a`，并使用与数据曝光档匹配的 `canon_0.0001.pt`。适配命令绕过作者 `sample()` 中不必要的 GT RGB 编码分支，因此推理输入严格只有低照 RAW：

```bash
dip-infer-official-sied \
  --official-repo /path/to/SIED \
  --checkpoint /path/to/canon_0.0001.pt \
  --data-root /path/to/SIED_dataset \
  --manifest manifests/enhancement/canon_test.txt \
  --output-dir outputs/official_sied_canon \
  --device cuda --precision fp16 --sampling-steps 20 \
  --ensemble-size 3 --seed 3407
```

检查点优先读取 EMA 权重。最终方案对三次独立 DDIM 输出做像素平均；第一个样本与 `--ensemble-size 1` 使用相同种子，便于公平消融。随机种子、集成数、精度、采样步数和实际权重来源均写入 `summary.json`。在 RTX 4090 上可以直接处理 3840×2160 图像，无需分块。

### 全场景曝光/色调精修

精修器仅预测受限的全局曝光、伽马、白平衡、3×3 色彩矩阵和平滑局部照度，初始是恒等映射，不生成额外高频细节。先用冻结 SIED 生成训练/验证缓存，再进行 BF16 全场景训练：

```bash
dip-train-sied-refiner \
  --official-repo /path/to/SIED \
  --checkpoint /path/to/canon_0.0001.pt \
  --data-root /path/to/SIED_dataset \
  --train-manifest manifests/enhancement/canon_train.txt \
  --val-manifest manifests/enhancement/canon_val.txt \
  --prediction-dir /path/to/cached_sied/images \
  --initial-checkpoint checkpoints/canon_sied_refiner_fullscene_bf16_best.pt \
  --output-dir outputs/canon_sied_refiner \
  --device cuda --precision bf16 --batch-size 8 \
  --max-steps 1500 --learning-rate 2e-5 --require-cuda
```

推理时在原 SIED 命令上增加 `--refiner-checkpoint /path/to/best.pt`。精修器只能使用训练集学习，通过验证门控后才允许在官方测试集报告一次冻结结果。

最终 1500 步精修器在 Canon 官方 76 图测试集达到 22.9980 dB、0.7868 SSIM 和 7.2288 ΔE2000。任务一提交时应以这一行作为最终平均 PSNR；Residual U-Net 的 16.1234 dB 只是课程规模基线，训练记录中的 27.1644 dB 是验证缓存数值。相对同一批三次采样未精修输出，PSNR 提升 0.8298 dB、SSIM 提升 0.005845、ΔE2000 降低 9.24%、亮度 MAE 降低 13.34%；饱和度 MAE 增加 0.70%，属于已记录的小幅取舍。

若推理时使用了 `--save-base-dir` 同时保存未精修输出，可在不重复扩散采样的情况下独立复算基线指标：

```bash
dip-evaluate-enhancement \
  --data-root /path/to/SIED_dataset \
  --manifest manifests/enhancement/canon_test.txt \
  --prediction-dir outputs/enhancement_final_test/base_images \
  --output-dir outputs/enhancement_final_test/base_evaluation \
  --method official_sied_ensemble3
```

### RAW-guided NAFNet 分阶段升级

该路线保留冻结 SIED 与受限光度分支，再用四通道 RGGB RAW 引导 NAFNet 残差恢复，并增加整图光度上下文一致性消融。4090或本次实测4080 SUPER可运行。以下流程使用全新实验目录；已有数据只审计复用，不重新执行准备脚本。准备脚本拒绝覆盖数据/权重，默认保留下载压缩包，因此须另预留压缩包空间。

```bash
bash cloud/prepare_sied_canon_assets.sh /root/autodl-tmp/sied30
bash cloud/run_sied30_phase1.sh /root/autodl-tmp/sied30
bash cloud/run_sied30_phase1_select.sh /root/autodl-tmp/sied30
bash cloud/run_sied30_phase2.sh /root/autodl-tmp/sied30
bash cloud/run_sied30_phase3_psnr.sh /root/autodl-tmp/sied30
bash cloud/run_sied30_phase4_cached_native.sh /root/autodl-tmp/sied30
bash cloud/run_sied30_phase5_context.sh /root/autodl-tmp/sied30
# 复核全部原生验证、权重哈希与HOLD_FINALIZE状态后，才执行最终验收。
bash cloud/run_sied30_finalize.sh /root/autodl-tmp/sied30
```

各阶段使用预设快速验证选择候选，再复核完整92张原生验证；Phase5直接用12张原生快速验证，并在无收益时早停。最终在完整92张上以Phase2作为固定参考，限制SSIM最大回摆0.002、各颜色误差最多增加2%，再选平均PSNR最高者，对76张官方测试图仅做本轮一次最终验收。测试清单、权重哈希与推理参数不可变绑定，禁止更换权重复用旧分数。92张验证场景已被官方SIED预训练见过，验证分数不是独立泛化成绩。完整新轮实验与负结果见[增强冲刺报告](reports/enhancement_sprint_20261024.md)。官方2.43GiB权重和数据集不提交到Git，推理需同时具备官方基础权重和本项目精修权重。

## Kodak 图像压缩

课件入口仓库将 Kodak 数据指向 `MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite`。本项目已固定上游提交，24 张 PNG 放在不提交到 Git 的 `data/compression/kodak/`；来源、图像规格和逐文件 SHA256 见 `manifests/compression/`。

标准 JPEG 基线已完成图像读取、压缩、解压缩和全量评价：

```bash
dip-compress-benchmark \
  --data-dir data/compression/kodak \
  --output-dir outputs/compression/jpeg_baseline \
  --qualities 30 50 70 90 \
  --subsampling 420 \
  --repeats 7 --warmup 2
```

命令会生成 96 个 JPEG、96 个解压 PNG、逐图 `metrics.csv`、汇总 `summary.json` 和代表图 `comparison.png`。汇总同时报告相对原 PNG 与原始 RGB 像素的压缩比、bpp、PSNR、SSIM 以及编码/解码时间，避免不同压缩比口径混用。

本机全量结果中，Q70 达到 10.535× PNG 文件压缩比、1.239 bpp、33.917 dB PSNR 和 0.9259 SSIM，是当前默认的质量/体积折中点。完整分析见 `reports/compression_jpeg_baseline.md`。

### 自研 DCTC v1 编码器

第二阶段实现了不依赖 JPEG 编码库的自定义 `.dctc` 格式，包含 RGB→YCbCr、4:2:0、8×8 DCT、质量缩放量化、Zigzag、DC 差分、AC 零游程和 DEFLATE 熵编码，并配有独立解码器：

```bash
dip-dct-benchmark \
  --data-dir data/compression/kodak \
  --output-dir outputs/compression/dctc_v1 \
  --qualities 30 50 70 90 \
  --subsampling 420 --entropy-level 9 \
  --repeats 3 --warmup 1 \
  --baseline-summary reports/assets/compression/jpeg_baseline/summary.json
```

DCTC Q70 在 Kodak 24 图上达到 8.965× PNG 文件压缩比、1.456 bpp、33.740 dB PSNR 和 0.9237 SSIM。它的率失真表现接近但仍弱于成熟 libjpeg，完整结构、结果和差异分析见 `reports/compression_dctc_v1.md`。

### DCTC v2 与神经压缩

DCTC v2 保留 v1 的颜色变换、DCT 和量化，新增 4:2:2、矢量化稀疏/截尾系数码流和快速 DEFLATE 配置。`trimmed + entropy-level 1` 是低延迟配置；它以较高码率换取约 8 ms 编码和约 4.5 ms 解码。v1 仍是自研编码器中的码率优先方案。

```bash
dip-dct-benchmark \
  --data-dir data/compression/kodak \
  --output-dir outputs/compression/dctc_v2_fast \
  --qualities 30 50 70 90 --subsampling 420 \
  --stream-format trimmed --entropy-level 1 \
  --repeats 3 --warmup 1 \
  --baseline-summary reports/assets/compression/jpeg_baseline/summary.json
```

机器学习路线使用 CompressAI 官方预训练的 Cheng2020 Anchor 模型。Kodak 始终只用于测试，不参与模型训练；bpp 由实际熵编码字节数计算，不使用概率估计值替代码流大小。

```bash
pip install -e '.[compression-ml]'
dip-learned-compression-benchmark \
  --data-dir data/compression/kodak \
  --output-dir outputs/compression/cheng2020_anchor \
  --architecture cheng2020-anchor --qualities 1 3 6 \
  --metric mse --device cuda --repeats 1 --warmup 0 \
  --baseline-summary reports/assets/compression/jpeg_baseline/summary.json
```

RTX 4090 的真实码流结果为：Q1 0.1196 bpp / 28.5804 dB，Q3 0.2709 bpp / 31.3405 dB，Q6 0.8056 bpp / 36.7047 dB。按未压缩 24-bit RGB 计算，三档压缩比分别为 200.746:1、88.598:1 和 29.790:1；相对 Kodak 原始 PNG 文件则为 109.155:1、48.175:1 和 16.198:1。Q3 相对 JPEG Q30 少用 58.94% bpp且高 0.849 dB；Q6 相对 JPEG Q70 少用 34.96% bpp且高 2.788 dB。Cheng2020 的率失真最优，但自回归熵解码约需 5.8 秒，实时部署仍应选 libjpeg。

图像增强最终结论见 `reports/enhancement_final_report.md`，压缩实验见 `reports/compression_jpeg_baseline.md` 和 `reports/compression_dctc_v1.md`，完整4090执行顺序见 `cloud/runbook.md`。

## 文档几何校正

先验证 200 对图像与二值页面遮罩：

```bash
dip-audit-geocorrection \
  --data-root data/geocorrection/GeoCorrection \
  --output-json manifests/geocorrection/dataset.json \
  --output-csv manifests/geocorrection/pairs.csv \
  --expected-count 200
```

运行无需训练的 oracle-mask 传统基线：

```bash
dip-rectify-classical \
  --data-root data/geocorrection/GeoCorrection \
  --output-dir outputs/geocorrection/classical_full \
  --max-side 1200 --expected-count 200
```

本机 200 张全部处理成功，页面边界归一化 RMSE 平均下降 95.64%，文本行平均倾斜角下降 29.61%。由于数据没有平整真值，项目不计算无意义的输入/输出 PSNR 或 SSIM。研究路线见 `reports/geocorrection_research_plan.md`，全量传统基线分析见 `reports/geocorrection_classical_baseline.md`。

4090 阶段已完成 UVDoc 零样本展平和“UVDoc + 传统边界精修”全量实验。混合方案将文本行平均倾斜角从 7.597° 降至 5.113°，40 张固定 OCR 子集的高置信字符数提升 22.03%。完整方法、版本、权重哈希和分类结果见 `reports/geocorrection_uvdoc_hybrid.md`。

### 端到端页面掩膜预测

原有传统/UVDoc 流程以数据集掩膜作为 oracle 上限，不能直接处理没有标注的新图像。最终版增加紧凑 U-Net，将 200 对图像按畸变类别固定划分为 156 张训练、22 张验证和 22 张测试；训练目标由 BCE、Dice 和边界损失组成。预测掩膜可直接接入原有透视和边界带纠正器：

```bash
dip-train-page-segmentation \
  --data-root data/geocorrection/GeoCorrection \
  --output-dir outputs/geocorrection/page_mask_unet \
  --input-size 512 --base-channels 32 --epochs 100 \
  --batch-size 8 --device cuda --precision bf16 --require-cuda

dip-infer-page-segmentation \
  --data-root data/geocorrection/GeoCorrection \
  --checkpoint outputs/geocorrection/page_mask_unet/best.pt \
  --output-dir outputs/geocorrection/predicted_masks \
  --split all --device cuda --precision bf16

dip-rectify-classical \
  --data-root data/geocorrection/GeoCorrection \
  --mask-subdir /absolute/path/to/predicted_masks/masks \
  --mask-source-label 'PageMaskUNet predictions' \
  --output-dir outputs/geocorrection/end_to_end
```

RTX 4090 固定测试集上，PageMaskUNet 达到 0.9461 IoU、0.9693 Dice 和 0.7880 boundary F1。预测掩膜驱动的 22 张全自动纠正把文本行平均倾斜从 9.994° 降至 7.104°，页面填充率达到 0.9829；同一测试集的 oracle-mask 上限为 6.097° 和 0.9998。

```bash
dip-infer-uvdoc \
  --data-root data/geocorrection/GeoCorrection \
  --output-dir outputs/geocorrection/uvdoc_full \
  --model-id PaddlePaddle/UVDoc_safetensors --device cuda

dip-rectify-classical \
  --data-root outputs/geocorrection/uvdoc_full \
  --image-subdir rectified --mask-subdir rectified_masks \
  --output-dir outputs/geocorrection/hybrid_full --max-side 1200

dip-eval-geocorrection-ocr \
  --data-root data/geocorrection/GeoCorrection \
  --keys-manifest manifests/geocorrection/ocr_test_40.txt \
  --classical-dir outputs/geocorrection/classical_full/corrected \
  --uvdoc-dir outputs/geocorrection/uvdoc_full/rectified \
  --hybrid-dir outputs/geocorrection/hybrid_full/corrected \
  --output-dir outputs/geocorrection/ocr_eval40 --device gpu
```
