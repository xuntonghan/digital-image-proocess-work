# Canon SIED 全场景精修实验资料

- 基模型：SIED ICCV 2025 Canon `0.0001` EMA，20 步 DDIM，3 次采样平均。
- 精修器：恒等初始化的受限曝光/色调网络，BF16，batch size 8，1,000 步。
- 训练/验证：832 张 Canon 训练图，8 张独立视觉门禁图；清单不重叠。
- 精修权重 SHA256：`a6fe63e6fa1fcc7cb244b03a644ae029c9ca9fb753b980f17750386246834b27`。
- SIED EMA 精简权重 SHA256：`327813671119063b970ae1f26430e88a3d682a8e20c455b18aea102b0687c4e5`。

目录中的 `training_log.jsonl`、`training_summary.json` 和 `run_environment.json` 保留训练过程与环境；`visual_gate_*` 是 8 张 4K 门禁结果；`test_*` 是权重冻结后的 76 张官方测试结果。不在 Git 中保存原始数据、官方大权重和 76 张全分辨率预测图。
