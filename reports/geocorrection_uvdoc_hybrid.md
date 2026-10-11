# GeoCorrection UVDoc 与混合几何校正

## 结论

在 RTX 4090 上对 200 张 GeoCorrection 图像完成了 UVDoc 零样本稠密展平和“UVDoc + 传统边界精修”全量实验。与原始页面裁剪相比：

- UVDoc 将文本行平均倾斜角从 7.597° 降到 5.328°，下降 29.87%；
- 混合方案进一步降到 **5.113°**，总降幅 32.70%；
- 混合方案将页面边界归一化 RMSE 从 0.011836 降到 **0.000714**，下降 93.96%；
- 固定 40 张分层 OCR 评价中，置信度不低于 0.8 的字符数由 35,226 增加到 **42,987**，提升 22.03%；33 张改善、1 张持平、6 张下降。

因此，项目的默认交付方案定为混合方案。传统方案仍是很好的边界基线，但不应作为文字识别主线：它在同一 OCR 子集上的高置信字符数下降了 29.43%。

## 方法设计

### 1. UVDoc 稠密展平

本项目采用 [UVDoc 作者仓库](https://github.com/tanguymagne/UVDoc) 对应的 Transformers 实现与 [PaddlePaddle/UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors) 预训练权重。模型以 712×488 输入预测 45×31 反向网格，然后将网格应用到高分辨率页面裁剪和同步遮罩上。详细接口与后处理参考 [Transformers UVDoc 文档](https://huggingface.co/docs/transformers/en/model_doc/uvdoc)。

本数据没有平整真值图，也没有反向映射真值，因此没有直接用 200 张图监督训练稠密展平网络，而是使用顶会预训练模型做迁移推理。数据集真值页面遮罩仅用于裁剪、同步变换和评价，所以该结果仍属于 **oracle-mask** 设置。

### 2. 混合几何精修

UVDoc 擅长恢复页内非线性弯曲，但零样本输出仍会有少量不规则外边界。混合方案将 UVDoc 输出和同步展平的遮罩作为新输入，再执行：

1. 轻量四角单应性矫正；
2. 上下边界逐列展平；
3. 左右边界逐行展平；
4. 文本主方向检测与 90° 自动旋转。

这一顺序先处理页内曲面，再整理页面外轮廓，避免仅靠外轮廓“拉直页面却伤害文字”。

## 实验设置与可复现性

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.19.0 |
| UVDoc 模型修订 | `bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4` |
| UVDoc 权重 SHA256 | `65f980e3083b4919d962d7b4c8ddb42a926dc8b36f4909944746cf2d7b7325dc` |
| OCR | PaddleOCR 3.7.0 / PaddleX 3.7.2 / PP-OCRv6 medium det+rec |
| 几何评价数量 | 200 |
| OCR 评价数量 | 按七个文件族分层固定 40 张 |
| 精修与 OCR 最长边 | 1200 |

UVDoc 模型预热后前向平均只需 4.82 ms；高分辨率重映射与遮罩后处理平均 1546.42 ms，完整 UVDoc 流程平均 2372.90 ms/张。混合精修额外平均需 1718.83 ms/张。这说明目前瓶颈不是 4090 模型前向，而是 CPU/高分辨率几何重映射。

## 全量几何结果

| 方法 | 边界 RMSE ↓ | 页面填充率 | 文本行倾斜角 ↓ |
|---|---:|---:|---:|
| 原始页面裁剪 | 0.011836 | 0.7554 | 7.597° |
| UVDoc | 0.005402 | 0.9778 | 5.328° |
| UVDoc + 边界精修 | **0.000714** | **0.9998** | **5.113°** |

UVDoc 的文本行倾斜角在 200 张中改善 169 张；混合方案相对原始页面改善 165 张，但均值更低。混合方案的边界 RMSE 在 200 张上全部优于原始页面。

| 文件族 | 原始 | UVDoc | 混合 |
|---|---:|---:|---:|
| 普通拍摄 | 5.499° | 3.446° | **3.421°** |
| 时间戳拍摄 | 8.968° | 5.737° | **5.059°** |
| 曲面 | 5.388° | **5.072°** | 5.461° |
| 折痕 | 5.797° | **4.933°** | 6.110° |
| 透视 | 12.705° | **6.981°** | 7.948° |
| 旋转 | 24.820° | 13.593° | **12.541°** |
| 书页 | 6.520° | **6.139°** | 6.264° |

精修对普通拍摄、时间戳拍摄和旋转类有明确增益；曲面、折痕和透视类的 Hough 倾斜指标反而略退化。这是下一阶段实现自适应路由、不对所有样本强制精修的主要依据。

## OCR 效果评价

因为数据没有人工转录，本实验使用同一个 PP-OCRv6 模型统计检测文本量、识别置信度和高置信字符数。这些是可读性诊断，**不是 OCR 准确率或 CER**。

| 方法 | 检测行数 | 高置信行数 | 高置信字符 | 平均行置信度 | 置信加权字符 |
|---|---:|---:|---:|---:|---:|
| 输入 | 2,679 | 2,171 | 35,226 | 0.8913 | 36,356.9 |
| 传统校正 | 2,292 | 1,530 | 24,860 | 0.8286 | 28,945.4 |
| UVDoc | **2,966** | **2,541** | 42,689 | **0.9135** | 43,765.3 |
| 混合 | 2,937 | 2,476 | **42,987** | 0.9061 | **43,911.1** |

UVDoc 的平均行置信度最高，混合方案的高置信字符总数和置信加权字符总数最高。两者差距很小：混合方案比 UVDoc 多 0.70% 高置信字符，但行置信度略低。就当前数据而言，混合方案适合作为默认输出，UVDoc 原始输出应同时保留作为对照和复杂曲面备选。

## 运行方法

```bash
pip install -e '.[geometry-ml,geometry-ocr]'

dip-infer-uvdoc \
  --data-root data/geocorrection/GeoCorrection \
  --output-dir outputs/geocorrection/uvdoc_full \
  --model-id PaddlePaddle/UVDoc_safetensors \
  --model-revision bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4 \
  --device cuda --expected-count 200

dip-rectify-classical \
  --data-root outputs/geocorrection/uvdoc_full \
  --image-subdir rectified --mask-subdir rectified_masks \
  --output-dir outputs/geocorrection/hybrid_full \
  --max-side 1200 --expected-count 200

dip-eval-geocorrection-ocr \
  --data-root data/geocorrection/GeoCorrection \
  --keys-manifest manifests/geocorrection/ocr_test_40.txt \
  --classical-dir outputs/geocorrection/classical_full/corrected \
  --uvdoc-dir outputs/geocorrection/uvdoc_full/rectified \
  --hybrid-dir outputs/geocorrection/hybrid_full/corrected \
  --output-dir outputs/geocorrection/ocr_eval40 \
  --device gpu --engine transformers --max-side 1200
```

## 限制与下一步

- 当前仍用真值页面遮罩裁剪，需训练轻量页面分割网络才能变成端到端系统；
- 无平整真值，不能合理报告 PSNR/SSIM；若要训练稠密展平，需引入 Doc3D/UVDoc 合成数据或手工采集平整参考；
- OCR 子集没有转录，下一步应人工标注 10–20 张，报告 CER 和词错误率；
- 混合精修对少数曲面/折痕样本有退化，可用几何指标与 OCR 分数做自适应选择，但选择规则应在独立验证集上冻结，不在当前 40 张评价子集上过拟合。

## 可复核文件

- `reports/assets/geocorrection/uvdoc/`：200 张 UVDoc 逐图指标和汇总；
- `reports/assets/geocorrection/hybrid/`：200 张混合方案指标、汇总和四列对比图；
- `reports/assets/geocorrection/ocr/`：40 张固定子集的 OCR 指标、逐图预测和权重哈希；
- `manifests/geocorrection/ocr_test_40.txt`：OCR 分层样本清单。
