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
