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
