# SIED 极暗 RAW 图像增强实验报告

更新时间：2026-10-01

## 1. 实验目标与数据

本实验完成课程第一部分要求：数据集读取、算法/模型设计、图像增强、客观效果评价，以及增强前后的可视化比较。处理对象为 SIED 数据集中的 Canon `0.0001` 与 Sony `0.001` 两个极暗 RAW 子集。

两个子集各包含 1000 对 `RAW + GT`。全部 RAW 均通过逐文件审计：3840×2160、`uint16`、黑电平 512、白电平 16383，文件尺寸与配对错误均为 0。读取过程为黑电平扣除、白电平归一化和 RGGB 四通道打包；不通过强制缩放掩盖尺寸问题。

使用 SIED 官方仓库提交 `fb7af94ac831e91c647415bf1ca0b1a75931833a` 的测试划分，并从官方训练部分按固定种子 `20260930` 抽取 10% 验证集：

| 子集 | 训练 | 验证 | 官方测试 | 合计 |
|---|---:|---:|---:|---:|
| Canon | 832 | 92 | 76 | 1000 |
| Sony | 822 | 91 | 87 | 1000 |

训练、验证和测试清单在 RAW 与 GT 路径上均互斥。验证集用于选择权重，测试集只用于最终评价。

## 2. 算法设计

### 2.1 传统自动 ISP 基线

处理链为：黑电平校正 → RGGB 去马赛克 → 灰世界白平衡 → 高分位自动曝光 → Gamma 映射 → `[0,1]` 裁剪。逐图曝光和白平衡只读取该图 RAW 的统计量，不读取 GT。全局参数仅在验证集选择并冻结：

- Canon：`gamma=2.0, percentile=98, target=0.9`；
- Sony：`gamma=1.8, percentile=99, target=0.9`。

### 2.2 学习型增强模型

主模型是约 1789 万参数的 Residual U-Net。输入为半分辨率的四通道打包 RAW，末端输出 12 通道并通过 PixelShuffle 2×恢复为原始分辨率 RGB。训练使用 256×256 RAW patch（对应 512×512 GT patch）、batch size 4、AdamW、学习率 `1e-4`、FP16 和固定随机种子。

主损失为：

```text
L = L1 + 0.2 × (1 - SSIM) + 1.0 × L_chroma
```

`L_chroma` 是 YCbCr 两个色度分量上的逐像素 L1，用于减少偏灰和错色。模型采用 BatchNorm；推理时使用固定运行统计，避免 GroupNorm 随每个图块改变统计量。4K 整图按 256×256 RAW 图块、32 像素重叠推理，并用二维 Hann 窗羽化融合，以消除拼接线。

## 3. 训练与模型选择

计算环境为单张 RTX 4090 24GB、PyTorch 2.8.0 + CUDA 12.8。Canon 和 Sony 独立训练，训练过程保存环境、清单哈希、逐步日志和可恢复检查点。

| 实验 | 关键差异 | 验证结果/结论 |
|---|---|---|
| Canon E1 | GroupNorm，纯 L1 | 5000 步可学习，但颜色与图块一致性较弱 |
| Canon E2 原型 | GroupNorm，L1+SSIM | 最佳验证 PSNR 19.83 dB；整图出现偏灰和横向分块痕迹 |
| Canon 主模型 | BatchNorm，增加色度损失与 Hann 融合 | 第 3000 步最优：PSNR 20.80 dB、SSIM 0.7649 |
| Canon 大块消融 | 无归一化，512 patch | 1000/2000 步约 13.24 dB，未达放行线，提前停止 |
| Sony 主模型 | 与 Canon 主模型相同 | 第 2000 步最优：PSNR 21.58 dB、SSIM 0.7624 |

Canon 在 5000 步时验证指标明显回落，Sony 在 2000 步后也未继续改善。因此依据事先规定的验证集早停规则，最终分别采用第 3000 步与第 2000 步的 `best.pt`，没有为了凑足 50,000 步继续过拟合。

## 4. 官方测试集结果

以下指标均在原始 3840×2160 RGB 图像上计算。PSNR/SSIM 使用 `data_range=1.0`；LPIPS 使用 AlexNet v0.1，并将输入从 `[0,1]` 映射到 `[-1,1]`。PSNR、SSIM 越高越好，LPIPS 越低越好。

| 方法 | 子集 | 数量 | PSNR / dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| 自动传统 ISP | Canon | 76 | 13.7400 | 0.1355 | 0.9164 |
| Residual U-Net 主模型 | Canon | 76 | **16.1234** | **0.6220** | **0.6380** |
| 自动传统 ISP | Sony | 87 | 14.2986 | 0.1642 | 0.8655 |
| Residual U-Net 主模型 | Sony | 87 | **17.0517** | **0.6332** | **0.6964** |

相对自动传统 ISP：

- Canon：PSNR `+2.3834 dB`、SSIM `+0.4865`、LPIPS `-0.2784`；
- Sony：PSNR `+2.7531 dB`、SSIM `+0.4691`、LPIPS `-0.1691`。

结果说明学习模型在结构保持和感知相似度上明显优于直接放大曝光的传统方法。传统方法会同步放大极暗 RAW 的噪声；学习模型能显著去噪并恢复轮廓，但部分场景仍有亮度偏差和色彩欠饱和。

## 5. 可视化与异常分析

最终四列对比图依次为：输入 RAW 预览、自动传统 ISP、学习模型、GT。Canon 示例使用测试集 ID 6/8/10/14；Sony 示例使用 ID 6/13/15/22。可视化表明 Hann 羽化后没有旧版本的横向拼接线，学习模型的噪声显著少于传统方法；同时强灯带附近仍可能过曝，商品颜色有时低于 GT 饱和度。

### Canon 对比

![Canon 输入、传统方法、学习模型与 GT 对比](assets/enhancement/canon/comparison.png)

### Sony 对比

![Sony 输入、传统方法、学习模型与 GT 对比](assets/enhancement/sony/comparison.png)

逐图指标未被均值隐藏。Canon 最低 PSNR 样本为 ID 35（3.23 dB），Sony 为 ID 61（4.77 dB）；这些极端异常主要表现为整体曝光判断失败。后续可采用全局亮度条件、曝光分组采样、EMA 和更丰富的颜色分布损失继续改善。

## 6. 结果文件

- Canon 学习模型汇总与逐图指标：`reports/assets/enhancement/canon/summary.json`、`metrics.csv`；
- Sony 学习模型汇总与逐图指标：`reports/assets/enhancement/sony/summary.json`、`metrics.csv`；
- Canon/Sony 传统基线：`reports/assets/enhancement/canon_classical/`、`sony_classical/`；
- 四列可视化：`reports/assets/enhancement/canon/comparison.png`、`sony/comparison.png`；
- 精简推理权重及完整训练日志保存在本机 `checkpoints/` 与 `outputs/`，不提交到 Git；
- 可复现实验配置：`configs/enhancement/`；
- 官方训练/验证/测试清单：`manifests/enhancement/`。

本报告中的汇总值均可由对应逐图 CSV 重新计算；训练配置、数据清单和权重相互独立保存，满足课程演示与复现实验需要。
