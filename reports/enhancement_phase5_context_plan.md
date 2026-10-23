# Phase 5：全场景光度上下文一致的原生分辨率精修

实验设计日期：2026-10-23。此文件记录待运行方案，不声明取得任何新实验分数。

## 已发现的瓶颈

1. Phase 4 在原生 512×512 随机裁剪上调用光度模型，最终推理的同一模型却读取整张 4K 图的低分辨率上下文。曝光、白平衡和局部照度因此出现输入分布差异。Phase 5 先在整图上生成光度校正结果，再裁剪训练；光度网络完全冻结。
2. Phase 4 的训练内最佳权重以 768 像素缩略图验证选出，只有最终一个权重执行原生验证。Phase 5 每 500 步直接在固定 12 张原生图上分块验证，用原生指标选权重，并对入选模型评价全部 92 张验证图。
3. NAF 模块具有全局平均池化，训练 512 块和推理 768 块的统计范围不同。本轮先统一训练与推理为 512 块、推理重叠 64，下一轮才以独立消融考虑 TLC 局部转换。
4. 官方 SIED 预训练权重曾使用官方训练集，而我们的 92 张精修验证图从该训练集中划出。因此精修验证集对新增分支独立，但对整个预训练流水线不是从未见过的场景；26.637 dB 验证分数不能当成冻结测试成绩或其无偏预测。

## 方法与数据边界

- 官方冻结测试 76 张不进入训练脚本、不进入缓存、不进入权重选择。
- 832 张训练图、92 张验证图分别建立缓存；检查 RAW 路径、GT 路径和图像 key 均无重复或交集。
- 输入缓存来自既有官方 SIED 三次盲预测平均的原生 4K PNG。整图光度缓存过程只读取预测、RAW 与模型，完全不读取 GT。
- 光度缓存为 `float16`、HWC `.npy`，可内存映射后裁剪。924 张约需 42.83 GiB；脚本检查未完成缓存所需空间，并保留至少 3 GiB 余量。源 PNG 的 key/尺寸/mtime、manifest SHA256 和光度权重 SHA256 绑定缓存，来源变化时拒绝混用。
- 输入 PNG 存在 1/255 量化，缓存还含 float16 舍入；最终盲推理读取新生成的浮点 SIED 预测，因此存在轻微精度差异，须在报告中保留。
- 每个原生 patch 的 RAW 增益根据全场景 RAW 均值计算；RAW、校正后 RGB 与 GT 在偶数坐标同步裁剪，再同步翻转。
- 训练直接调用结构残差分支，绝不对 patch 再次计算光度参数；最终推理仍为“整图光度校正一次 + 分块结构残差”。

## 预算、选择与复现

默认从 Phase 1/2/3/4 中已有完整 92 张原生验证且 PSNR 最高的检查点初始化。原生分辨率计分不与缩略图混比。

```bash
cd /root/autodl-tmp/sied30/project
bash cloud/run_sied30_phase5_context.sh /root/autodl-tmp/sied30
```

5,000 步预算试跑可设置 `PHASE5_MAX_STEPS=5000`；默认最多 10,000 步。使用单卡、FP16、512 RGB crop、batch size 1、AdamW、学习率 `2e-5`、余弦退火、EMA `0.999`。每 500 步验证与保存；2,500 步之后连续六次没有达到门槛的改善就早停。`--resume` 从本阶段最近的模型/优化器/调度器/scaler 状态继续；DataLoader 与随机 patch 序列不承诺逐位相同。

选择目标为逐图 PSNR 的平均值；提高至少 0.01 dB 才刷新最佳检查点。相对本阶段初始化的固定原生验证子集，SSIM 退化不得超过 0.002，平均/95 分位 ΔE2000、亮度 MAE、饱和度 MAE、色相误差退化均不得超过 2%。这些是有限回摆门槛，不应写成“所有指标都提高”。初始检查点也保存为候选，避免无收益训练替换更好的模型。LPIPS 在最终验收中计算，训练内未将其隐含宣称为已改善。

输出：

```text
runs/canon_raw_guided_naf_phase5_context/
  initial_selection.json
  initial_native_validation.json
  run_environment.json
  training_log.jsonl
  best.pt
  step_*.pt
  resume_state.pt
  summary.json
runs/canon_raw_guided_naf_phase5_val_native/
  metrics.json
  summary.json
```

若最终选择 Phase 5，完整验证与冻结测试必须使用 `--refiner-tile-size 512 --refiner-overlap 64`。其他阶段保留各自原评测设置。统一 finalizer 在全部设计决策结束后只验收冻结测试一次。

### 完整验证放行与冻结验收保护

最终选择由纯标准库模块 `dip_course.enhancement.finalization` 执行，训练中固定 12 张快速验证的门槛不能替代最终完整 92 张门槛。各候选必须包含完整 `count=92`、有限数值、存在的检查点；若验证摘要声明 `checkpoint_sha256` 则必须匹配实物，Phase 5 的摘要和完成记录必须明确提供同一检查点哈希。兼容此前未写哈希的旧阶段验证摘要，但最终选择必定计算并绑定哈希。

全量门槛固定参考 Phase 2 已完成的 92 张原生评价：SSIM 最多下降 0.002，参考中已有的平均与 P95 ΔE2000、亮度 MAE、饱和度 MAE、色相误差各最多增加 2%。缺失参考指标的候选不能绕过门槛。在合格候选中选择平均 PSNR 最高者，Phase 2 始终作为兜底。这是允许有限回摆的选择规则，不能描述成“所有指标均提高”。

`final/selection.json` 在首次验收前不可覆盖地写入，绑定所选精修权重、官方基础权重、冻结 76 张清单的 SHA256，以及种子、采样步数、集成数量、精度和 tile/overlap。精修权重先独立复制到不可覆盖的 `final/best.pt`，推理使用这个快照。`frozen_test/selection.json` 与 `evaluation_started.json` 保存推理前的绑定；若已启动却没有完整结果，后续调用直接失败，不自动重新测试。

完整结果复用前必须核对 76 张数量、逐图 key 与清单顺序、精度与随机参数、官方/精修权重路径及哈希、PSNR/SSIM/LPIPS 和颜色指标的有限性。通过后写入不可覆盖的 `verification.json`，继续绑定结果摘要和逐图指标的 SHA256。若已有测试结果与当前选择不同、缺失旧绑定或内容变化，程序失败并保留原文件，不能通过替换最终权重而复用旧分数。finalizer 以 Linux `flock` 互斥，原 `HOLD_FINALIZE` 继续阻止尚未审阅的验收。

上述保护不代替异机备份；同一云盘上的 `final/best.pt` 仍需在结束前下载并核对 SHA256。

## 相关一手论文与作者代码

- [SIED，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Learning_to_See_in_the_Extremely_Dark_ICCV_2025_paper.html)；[作者代码](https://github.com/JianghaiSCU/SIED)：本项目基础模型。其 Canon 最暗照度官方分数约 22.52 dB，30 dB 应视为研究目标而非可承诺结果。
- [NAFNet 作者实现，ECCV 2022](https://github.com/megvii-research/NAFNet)：采用简单门控与简化通道注意力的残差恢复骨干。
- [TLC 作者实现，ECCV 2022](https://github.com/megvii-research/TLC)：指出裁剪训练与整图推理中的全局聚合分布差异；支持本轮优先修正上下文一致性的判断。此轮统一 tile 大小，尚未声称实现完整 TLC。
- [DNF，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.html)；[作者代码](https://github.com/Srameo/DNF)：RAW 去噪与颜色映射分离及跨阶段特征反馈。本项目借鉴任务解耦原则，并非完整复现 DNF。
- [Why Low-Light Cameras Go Color Blind，ICCP 2026](https://arxiv.org/abs/2607.11090)：黑电平误差可能造成极暗 RAW 的色偏。可考虑只用训练集学习受限偏置，但本轮仍遵守 SIED 的固定黑电平 512。
- [MeanFlow for Unified RAW Restoration，2026 预印本](https://arxiv.org/html/2608.01720v1)：直接涉及 SIED，但主要 Sony RAW 目标与差异化 ISP 协议不能与本项目 Canon JPG 参考结果直接对比。此文为预印本，未将其标注成已确认的顶会论文。

## 验证

新增单测覆盖：结构 forward 不调用 patch 光度分支、全场景缓存与 RAW 裁剪对齐、缓存过程不读取 GT、缓存来源变化拒绝复用、缓存结构推理与最终整图光度推理等价、颜色/SSIM 验证门槛、GT 路径交叉检查。

```bash
python -m pytest tests/test_train_sied_restorer_context.py
ruff check src/dip_course/cli/train_sied_restorer_context.py tests/test_train_sied_restorer_context.py
bash -n cloud/run_sied30_phase5_context.sh
```
