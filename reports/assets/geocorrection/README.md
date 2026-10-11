# GeoCorrection 可复核结果

`classical/` 保存传统 oracle-mask 几何校正基线的轻量验收材料：

- `metrics.csv`：200 张逐图指标；
- `summary.json`：全量与七类分组汇总；
- `comparison.png`：输入、数据集遮罩、单应性结果和最终边界展平结果。

`uvdoc/` 保存 200 张 UVDoc 零样本展平的逐图指标和汇总。

`hybrid/` 保存 200 张“UVDoc + 边界精修”指标、汇总与原图/传统/UVDoc/混合四列对比图。

`ocr/` 保存固定 40 张分层子集的 PP-OCRv6 指标、预测文本、汇总和模型文件 SHA256。由于没有人工转录，这些是识别量/置信度诊断，不是 CER。

完整的 200 张输出位于本机被 Git 忽略的 `outputs/geocorrection/classical_full/`，不提交到仓库。数据集和原始压缩包也不提交。
