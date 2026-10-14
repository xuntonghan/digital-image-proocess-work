# 数据和模型权重复现说明

本仓库不提交原始课程数据、大型预训练权重或全量中间输出。这些内容需要按下述目录和哈希准备，然后再运行 `README.md` 或综合报告中的命令。

## SIED 数据

课程提供 Canon `0.0001`、Sony `0.001`、Canon GT 和 Sony GT 四个解压目录。放置为：

```text
data/SIED_dataset/
├── 0.0001/
├── 0.001/
├── gt/
└── gt 2/
```

`manifests/source/canon_all.txt` 和 `sony_all.txt` 各应解析为 1000 对。RAW 应为 3840×2160、`uint16`、每个文件 16,588,800 字节，黑/白电平为 512/16383。

```bash
dip-audit --data-root data/SIED_dataset \
  --manifest manifests/source/canon_all.txt
dip-audit --data-root data/SIED_dataset \
  --manifest manifests/source/sony_all.txt
```

## GeoCorrection 数据

放置为：

```text
data/geocorrection/GeoCorrection/
├── images/   # 200 JPG
└── mask/     # 200 PNG
```

数据集聚合 SHA256 为 `5523dfb7b4ad5f053b88ed5648e3f446bcb879a91c231b731d660bd572746050`。逐文件哈希和尺寸在 `manifests/geocorrection/pairs.csv`。

```bash
dip-audit-geocorrection \
  --data-root data/geocorrection/GeoCorrection \
  --output-json manifests/geocorrection/dataset.json \
  --output-csv manifests/geocorrection/pairs.csv \
  --expected-count 200
```

## Kodak 数据

使用 `MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite` 提交 `dd2e1105fe8d9afe1120cd7e4c0e5e41f509a81b`，将 24 张 PNG 放在 `data/compression/kodak/`。数据集聚合 SHA256 为 `b2194c51cff99d6f3cc14fa73c0a805817b1e705abc02cf94f89165cc5e6bcc5`。

```bash
cd data/compression/kodak
shasum -a 256 -c ../../../manifests/compression/kodak.sha256
```

## 模型权重

| 权重 | 建议路径 | SHA256 | 来源 |
|---|---|---|---|
| Canon Residual U-Net | `checkpoints/enhancement/canon/inference_best.pt` | `935465a3f2b185f159f7e4f96947f7d7aa8f3a8ed5d6f77cc7cec7213e4a7ee0` | 本项目训练 |
| Sony Residual U-Net | `checkpoints/enhancement/sony/inference_best.pt` | `225e3f845fb19998176c65275b853a1f3814a0e0ac9b7eab4cf499402e5777a2` | 本项目训练 |
| Canon SIED EMA | `checkpoints/official/canon_0.0001.pt` | 精简 EMA：`327813671119063b970ae1f26430e88a3d682a8e20c455b18aea102b0687c4e5` | SIED 作者发布，应遵守原许可条款 |
| Canon 曝光色调精修 | `checkpoints/canon_sied_refiner_fullscene_bf16_best.pt` | `a6fe63e6fa1fcc7cb244b03a644ae029c9ca9fb753b980f17750386246834b27` | 本项目训练 |
| UVDoc | Hugging Face 缓存 | `65f980e3083b4919d962d7b4c8ddb42a926dc8b36f4909944746cf2d7b7325dc` | `PaddlePaddle/UVDoc_safetensors` 修订 `bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4` |
| PP-OCRv6 medium | PaddleX 官方模型缓存 | 见 `reports/assets/geocorrection/ocr/model_sha256.txt` | PaddleOCR/PaddleX |

权重放置后可用以下命令核对：

```bash
shasum -a 256 \
  checkpoints/enhancement/canon/inference_best.pt \
  checkpoints/enhancement/sony/inference_best.pt \
  checkpoints/canon_sied_refiner_fullscene_bf16_best.pt
```

## 可复现边界

只克隆 Git 仓库可以重现代码结构、测试、压缩核心算法以及已提交的逐图指标和图表。要重新生成增强和几何校正的全量输出，还需要上述原始数据和模型权重。
