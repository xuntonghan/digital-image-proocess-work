# 三个项目的下一轮改进计划

更新日期：2026-10-15

## 决策摘要

本机复现已经证明数据、权重、RAW 解码、分块推理和指标导出链路可靠。Canon 与 Sony 的 PSNR、SSIM、LPIPS 均与正式报告一致，因此下一轮不再重复基线实验，而是按以下优先级推进：

1. 立即在 RTX 4090 上继续微调 Canon SIED 色调精修器；
2. 随后为 GeoCorrection 训练页面分割器，逐步移除真值遮罩依赖；
3. 压缩项目先在 CPU 上改进 DCTC 熵编码和向量化，再决定是否引入学习式压缩。

## 1 图像增强

### 当前证据

- Canon U-Net 本机复现：16.1232 dB PSNR、0.6220 SSIM、0.6380 LPIPS；
- Sony U-Net 本机复现：17.0515 dB PSNR、0.6333 SSIM、0.6965 LPIPS；
- 当前最强 Canon SIED 三样本集成：22.1700 dB PSNR、0.7817 SSIM；
- 当前 SIED+Refiner：22.9060 dB PSNR、0.7861 SSIM，并降低平均色差和亮度误差。

### 下一轮实验

冻结 SIED，不重新训练大型扩散模型。从现有 1000 步精修权重继续低学习率微调：

- 初始权重：`checkpoints/canon_sied_refiner_fullscene_bf16_best.pt`；
- 使用 832 张训练缓存；
- 将验证从 8 张视觉门禁扩展到完整 92 张验证集；
- BF16、batch size 8、学习率 `2e-5`、1500 个新增优化步骤；
- 保留曝光、白平衡、3×3 色彩矩阵和低频照度场的受限参数化；
- 验证损失早停，测试集仅在方案冻结后评价一次。

放行条件：验证损失下降；PSNR、SSIM不回退；ΔE2000、亮度 MAE 和 P95 色差至少两项改善；极暗和高饱和失败样本没有明显恶化。

## 2 几何纠正

### 当前瓶颈

传统方法和 UVDoc 混合方案都使用数据集页面遮罩完成裁剪或纠正，因此尚不是任意照片输入的端到端系统。数据集又没有平整真值，不能用 PSNR 或 SSIM监督稠密展平。

### 下一轮实验

第一阶段训练轻量 U-Net 或 SegFormer 页面分割器：

- 对 200 张图做固定五折或 160/20/20 分层划分；
- 以 IoU、Dice、边界 F-score评价遮罩；
- 将预测遮罩送入现有传统、UVDoc和混合流程；
- 比较预测遮罩造成的边界误差和 OCR 变化。

第二阶段再引入 Doc3D/UVDoc 合成稠密映射真值微调展平网络。最终必须补 10 至 20 张人工 OCR 转录，报告 CER，而不是只报告高置信字符数量。

## 3 图像压缩

### 当前瓶颈

DCTC Q70 的 1.456 bpp、33.740 dB、0.9237 SSIM接近 JPEG，但编码和解码约慢两个数量级，码率也略差。主要瓶颈是 Python 逐块处理、固定量化表和通用 DEFLATE。

### 下一轮实验

优先完成无需 GPU 的改进：

1. 批量向量化 8×8 DCT、量化和 Zigzag；
2. 用训练图统计专用 Huffman 表或范围编码替代通用 DEFLATE；
3. 增加 4:4:4、4:2:2 和分量独立量化表；
4. 在相同 bpp 下比较率失真，而不是比较相同 Q 值。

若后续增加学习式压缩，应使用独立训练集训练超先验模型，Kodak 24 张只作为测试集，并同时报告模型大小、显存和编解码耗时。

## 4090 首轮命令

云端已有 SIED 训练缓存时，首轮继续微调命令为：

```bash
dip-train-sied-refiner \
  --official-repo /root/autodl-tmp/cv-homework/third_party/SIED-official \
  --checkpoint /root/autodl-tmp/cv-homework/official_sied/canon_0.0001.pt \
  --initial-checkpoint /root/autodl-tmp/cv-homework/project/checkpoints/canon_sied_refiner_fullscene_bf16_best.pt \
  --data-root /root/autodl-tmp/cv-homework/data/SIED_dataset \
  --train-manifest manifests/enhancement/canon_train.txt \
  --val-manifest manifests/enhancement/canon_val.txt \
  --prediction-dir /root/autodl-tmp/cv-homework/cache/official_sied/canon_train_val_768/images \
  --output-dir /root/autodl-tmp/cv-homework/runs/canon_sied_refiner_v2 \
  --device cuda --precision bf16 --batch-size 8 --num-workers 4 \
  --max-steps 1500 --learning-rate 2e-5 --weight-decay 1e-4 \
  --validation-interval 25 --checkpoint-interval 250 \
  --gradient-accumulation 1 --require-cuda
```

启动前必须确认缓存同时包含 832 张训练预测和 92 张验证预测；缺少验证缓存时先用盲推理模式补齐，不允许用测试集调参。
