# 数据清单约定

清单使用 UTF-8 纯文本，每行两个字段：

```text
relative/path/to/low.raw relative/path/to/gt.jpg
```

空行和以 `#` 开头的行会被忽略。路径含空格时使用单引号或双引号。路径必须相对于 `--data-root`，禁止使用越过数据根目录的 `..` 路径。

建议每个相机维护三份清单：

```text
manifests/train_canon.txt
manifests/val_canon.txt
manifests/test_canon.txt
manifests/train_sony.txt
manifests/val_sony.txt
manifests/test_sony.txt
```

如果老师提供官方划分，直接使用官方划分；否则固定随机种子后一次性划分并保存清单，不在每次运行时重新随机划分。

## 图像压缩

Kodak 压缩测试集的上游来源、固定提交、图像规格和 SHA256 清单见 `compression/`。原始 PNG 放在被 Git 忽略的 `data/compression/kodak/`，不与仓库中的实验代码和指标结果混存。

## 几何校正

`geocorrection/dataset.json` 记录 GeoCorrection 的数量、分辨率、文件族、模式和全数据聚合 SHA256；`geocorrection/pairs.csv` 记录 200 对图像/遮罩相对路径、尺寸、前景比例以及各文件 SHA256。

原始 JPG、PNG 和压缩包均不提交到 Git。当前遮罩表示畸变输入中的页面区域，不是平整真值图或稠密形变场。
