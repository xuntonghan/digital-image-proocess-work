# AutoDL RTX 4090 执行手册

## 目录和容量

- 项目：`/root/autodl-tmp/cv-homework/project`
- 数据：`/root/autodl-tmp/cv-homework/data/SIED_dataset`
- 虚拟环境：`/root/autodl-tmp/venvs/cv-homework`
- GT 缓存：`/root/autodl-tmp/cv-homework/cache`
- 训练结果：`/root/autodl-tmp/cv-homework/runs`

只上传四个已解压目录：`0.0001`、`0.001`、`gt`、`gt 2`，合计约 34 GiB；不要重复上传四个 ZIP。两个 GT 的可选 uint8 缓存合计约 46.4 GiB。按 checkpoint 和最终输出估算，200 GiB 数据盘足够。

不得复用或修改 Gala 项目的 ComfyUI 目录、模型目录和虚拟环境。

## 第一阶段：环境与数据放行

```bash
cd /root/autodl-tmp/cv-homework/project
python3 -m venv --system-site-packages /root/autodl-tmp/venvs/cv-homework
source /root/autodl-tmp/venvs/cv-homework/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install '.[train]'

bash cloud/check_environment.sh
pytest -q

dip-audit --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --manifest manifests/source/canon_all.txt \
  --output-json /root/autodl-tmp/cv-homework/runs/canon_audit.json

dip-audit --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --manifest manifests/source/sony_all.txt \
  --output-json /root/autodl-tmp/cv-homework/runs/sony_audit.json
```

审计必须各为 1000 对、0 错误。上传后还要核对源清单 SHA256。

## 第二阶段：建立 GT 随机裁剪缓存

```bash
dip-prepare-gt-cache \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --manifest manifests/source/canon_all.txt \
  --output-dir /root/autodl-tmp/cv-homework/cache/canon_gt

dip-prepare-gt-cache \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --manifest manifests/source/sony_all.txt \
  --output-dir /root/autodl-tmp/cv-homework/cache/sony_gt
```

RAW 训练 patch 直接从原始文件内存映射读取；GT 缓存让随机 patch 不必在每一步重复解码整张 4K JPEG。

## 第三阶段：100 步冒烟测试

Canon 示例：

```bash
dip-train \
  --config configs/enhancement/canon.toml \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --train-manifest manifests/enhancement/canon_train.txt \
  --val-manifest manifests/enhancement/canon_val.txt \
  --gt-cache-dir /root/autodl-tmp/cv-homework/cache/canon_gt \
  --output-dir /root/autodl-tmp/cv-homework/runs/canon_smoke_100 \
  --max-steps 100 --require-cuda
```

Sony 使用 `sony.toml`、`sony_train.txt`、`sony_val.txt` 和 `sony_gt`。放行条件：无 NaN、无 OOM、输出尺寸正确、训练损失下降、GPU 利用率正常。

## 第四阶段：5000 步试验

- E1：分别使用 `canon_l1.toml`、`sony_l1.toml`，只跑 5000 步。
- E2：分别使用 `canon.toml`、`sony.toml`，先跑 5000 步；损失为 L1 + SSIM + 色度约束。
- 对比验证损失、验证 PSNR/SSIM 和样图。如果5000步内指标回落，直接使用 `best.pt` 早停；只有仍持续改善时才续训到 50,000 步。

完整训练不得在 100 步冒烟或 5000 步试验失败时继续。后续推理使用官方测试清单和 1000 张全量清单；原始尺寸计算 PSNR、SSIM、LPIPS，并下载 CSV、JSON、样图和 checkpoint 后再关机。

## 第五阶段：色彩保真升级

旧模型存在色调偏移与亮部过曝，新试验使用 HVI 色彩/亮度解耦、全局 RAW 统计条件、无 BatchNorm 网络、EMA 和颜色一致性损失。先运行 100 步冒烟，再分别运行 5000 步：

```bash
dip-train \
  --config configs/enhancement/canon_color_faithful.toml \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --train-manifest manifests/enhancement/canon_train.txt \
  --val-manifest manifests/enhancement/canon_val.txt \
  --gt-cache-dir /root/autodl-tmp/cv-homework/cache/canon_gt \
  --output-dir /root/autodl-tmp/cv-homework/runs/canon_color_faithful_5000 \
  --max-steps 5000 --require-cuda

dip-train \
  --config configs/enhancement/sony_color_faithful.toml \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --train-manifest manifests/enhancement/sony_train.txt \
  --val-manifest manifests/enhancement/sony_val.txt \
  --gt-cache-dir /root/autodl-tmp/cv-homework/cache/sony_gt \
  --output-dir /root/autodl-tmp/cv-homework/runs/sony_color_faithful_5000 \
  --max-steps 5000 --require-cuda
```

两个 5000 步试验都必须在对应 `*_visual_gate.txt` 上输出八张对比图，并与旧模型及传统 ISP 同时比较 ΔE2000、亮度 MAE、饱和度 MAE、色相误差、PSNR 和 SSIM。只有颜色指标和视觉门禁同时通过，才续训到 30,000 步。

## 第六阶段：SIED 官方强基线与最终替换方案

只安装作者代码实际推理所需的最小依赖，并将官方仓库固定到提交 `fb7af94ac831e91c647415bf1ca0b1a75931833a`。不要安装作者完整 `requirements.txt`，否则会把当前 PyTorch/CUDA 环境降级到旧版本。

```bash
python -m pip install '.[sied]'

dip-infer-official-sied \
  --official-repo /root/autodl-tmp/cv-homework/third_party/SIED-official \
  --checkpoint /root/autodl-tmp/cv-homework/official_sied/canon_0.0001.pt \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --manifest manifests/enhancement/canon_test.txt \
  --output-dir /root/autodl-tmp/cv-homework/outputs/official_sied/canon_test \
  --device cuda --precision fp16 --sampling-steps 20 \
  --ensemble-size 3 --seed 3407
```

本项目的适配器直接调用作者 Stage-2 EMA 网络的低照 RAW 分支，不向网络传入 GT RGB；参考图只在预测完成后读取并计算指标。最终方案平均三次独立 DDIM 预测以降低随机性；验证门禁显示它同时改善 PSNR、SSIM、ΔE2000、亮度和色相指标。最终汇报必须保留 `summary.json` 中的 `blind_inference: true`、`weight_source`、`ensemble_size`、采样步数和随机种子，以证明比较可复现且没有答案泄漏。
