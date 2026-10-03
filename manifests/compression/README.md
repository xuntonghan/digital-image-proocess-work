# Kodak 图像压缩数据

课件指定的入口仓库是 `wenwang123/Digital-Image-Processing`，其 README 将图像压缩数据指向 `MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite`。本项目固定后者的 `dd2e1105fe8d9afe1120cd7e4c0e5e41f509a81b` 提交，将 `PhotoCD_PCD0992/01.png` 至 `24.png` 放在：

```text
data/compression/kodak/
```

数据本体不提交到 Git。下载后可在项目根目录验证：

```bash
cd data/compression/kodak
shasum -a 256 -c ../../../manifests/compression/kodak.sha256
```

`kodak_dataset.json` 记录上游提交、图像规格和整体清单指纹。这 24 张图将作为同一压缩方案的完整测试集，不做训练/测试随机拆分。
