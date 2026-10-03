# 图像增强实验资料

本目录保存课程第一部分可直接检查的轻量实验结果：

- `canon/`、`sony/`：Residual U-Net 在官方测试集上的逐图指标、汇总指标和四列对比图；
- `canon_classical/`、`sony_classical/`：自动传统 ISP 基线的逐图指标与汇总指标。
- `canon_official/`：SIED ICCV 2025 官方 EMA 模型的 Canon 色彩门禁与官方测试指标，以及“输入/传统/旧模型/SIED/GT”五列对比图。
- `canon_official_refiner/`：冻结 SIED 后的全场景曝光/色调精修训练记录、4K 门禁与官方测试指标，以及“输入/传统/SIED/精修/GT”五列对比图。

原报告四列图依次为输入 RAW 预览、自动传统 ISP、学习模型与 GT；两个 `visual_gate_comparison.png` 分别展示 Residual U-Net 到 SIED、以及 SIED 到全场景精修的直接对比。原始 SIED 数据、模型权重、训练检查点和全部增强图像体积较大，未提交到仓库。
