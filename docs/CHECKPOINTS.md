# 模型权重与分发说明

项目默认忽略 `checkpoints/` 目录，避免将大型二进制权重直接推入 Git。

- 本项目训练的 Canon/Sony Residual U-Net 推理权重各约 68 MiB，可在课程许可的前提下放入最终作业压缩包。
- Canon SIED 权重由 SIED 作者发布。建议提供获取说明和 SHA256，不在本仓库重新分发未经确认的大型权重。
- 曝光色调精修权重约 189 KiB，由本项目训练，可作为作业附件。
- UVDoc 和 PP-OCRv6 由对应模型库自动缓存，应固定模型修订并核对哈希。

完整路径和 SHA256 见项目根目录的 `DATA_AND_WEIGHTS.md`。
