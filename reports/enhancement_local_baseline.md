# SIED 图像增强：本机阶段结果

> 本文保留为训练前的阶段记录。4090训练、LPIPS和最终官方测试结果见 `reports/enhancement_final_report.md`。

更新时间：2026-10-01

## 1. 数据读取与审计

课程指定的两个子集均已完成逐文件审计：

| 子集 | RAW/GT 对数 | RAW 尺寸与类型 | 黑/白电平 | 审计错误 |
|---|---:|---|---|---:|
| Canon 0.0001 | 1000 | 3840×2160, uint16 | 512 / 16383 | 0 |
| Sony 0.001 | 1000 | 3840×2160, uint16 | 512 / 16383 | 0 |

读取流程为黑电平扣除、白电平归一化和 RGGB 四通道打包。Sony GT 已通过压缩包文件哈希确认对应 `gt 2`，Canon GT 对应 `gt`。

## 2. 数据划分

课程的 1–1000 号子集可以完整映射到 SIED 官方仓库提交 `fb7af94ac831e91c647415bf1ca0b1a75931833a` 的训练/测试清单。项目保留官方测试集，并从官方训练部分以固定种子 `20260930` 抽取 10% 验证集：

| 子集 | 训练 | 验证 | 官方测试 | 合计 |
|---|---:|---:|---:|---:|
| Canon | 832 | 92 | 76 | 1000 |
| Sony | 822 | 91 | 87 | 1000 |

三个集合路径互斥。验证集只用于参数选择，官方测试集只用于最终评价。

## 3. 已实现算法

传统自动 ISP 基线包含：

1. RAW 黑电平校正与归一化；
2. RGGB 归一化卷积去马赛克；
3. 每张图使用灰世界法估计白平衡；
4. 根据亮度高分位数自动估计曝光；
5. Gamma 映射及 `[0,1]` 裁剪。

Canon 和 Sony 的全局超参数分别只在各自验证集上搜索。Canon 选择 `gamma=2.0, percentile=98, target=0.9`；Sony 选择 `gamma=1.8, percentile=99, target=0.9`。逐图曝光和白平衡只由该图 RAW 统计量计算，不读取 GT。

学习型主模型为约 1789 万参数的 Residual U-Net：四通道打包 RAW 在半分辨率空间处理，末端通过 PixelShuffle 2×输出全分辨率 RGB。主损失为 `L1 + 0.2 × (1-SSIM)`；另保留纯 L1 的 5000 步消融实验。完整训练需要 RTX 4090。

## 4. 本机正式测试结果

以下 PSNR、SSIM 均在原始 3840×2160 RGB 图像上计算。LPIPS 尚未填入，因为将与学习型模型一起在 4090 上使用相同实现和原始尺寸统一计算。

| 方法 | 子集 | 测试张数 | PSNR / dB | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| 固定曝光/白平衡 ISP | Canon | 76 | 12.7782 | 0.1332 | 待云端 |
| 自动曝光/白平衡 ISP | Canon | 76 | **13.7400** | **0.1355** | 待云端 |
| 固定曝光/白平衡 ISP | Sony | 87 | 13.3664 | 0.1472 | 待云端 |
| 自动曝光/白平衡 ISP | Sony | 87 | **14.2986** | **0.1642** | 待云端 |

自动算法相对固定算法在 Canon 上提高 0.9617 dB，在 Sony 上提高 0.9322 dB，并解决了 Sony 部分场景 RAW 通道比例反转导致的明显紫偏。传统方法仍会把极暗 RAW 的噪声一并放大，因此学习型联合曝光、去噪、去马赛克和颜色映射是下一步重点。

## 5. 结果位置

- 自动基线逐图指标：`outputs/enhancement/classical/canon_test_auto/metrics.csv`、`outputs/enhancement/classical/sony_test_auto/metrics.csv`
- 自动基线汇总：对应目录中的 `summary.json`
- 全分辨率增强结果：对应目录中的 `images/`
- 对比图：`outputs/enhancement/comparisons/canon_test_auto.png`、`outputs/enhancement/comparisons/sony_test_auto.png`
- 验证集选参记录：`outputs/enhancement/tuning/canon_auto.json`、`outputs/enhancement/tuning/sony_auto.json`

`outputs/` 是可再生成产物，默认不提交到 Git。

## 6. 当前结论与待完成项

本机阶段已完成数据读取、严格审计、无泄漏划分、传统算法设计、全分辨率增强、PSNR/SSIM 评价和可视化。代码回归检查共 24 项，全部通过。

4090 阶段现已完成：数据缓存与审计、训练门槛实验、验证集早停、官方测试集 PSNR/SSIM/LPIPS 统一评价，以及最终四列可视化。最终结果与局限性分析见 `reports/enhancement_final_report.md`。
