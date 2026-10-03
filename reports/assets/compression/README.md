# 图像压缩实验资料

`jpeg_baseline/` 保存 Kodak 24 图 JPEG 基线的轻量、可审计结果：

- `summary.json`：四个质量档位的全数据集汇总指标与运行环境；
- `metrics.csv`：每张图、每个质量档位的压缩率、bpp、PSNR、SSIM 和耗时；
- `comparison.png`：五张代表图的原图与 Q30/Q50/Q70/Q90 解压结果。

完整 JPEG 与解压 PNG 共 192 个文件，保存在不纳入 Git 的
`outputs/compression/jpeg_baseline/`，可通过仓库命令一键重建。
