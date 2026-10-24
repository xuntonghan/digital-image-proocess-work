# 项目一：Canon 1000张图像增强最终提交报告

完成日期：2026-10-25。范围仅为图像增强；几何纠正、压缩未在本轮继续训练或改动。

## 1. 提交结论

按课程要求，对原始Canon **全部1000张**逐图计算原生分辨率PSNR和SSIM，再取算术平均：

| 完整1000张方案 | 平均PSNR↑ / dB | 平均SSIM↑ |
|---|---:|---:|
| 冻结SIED＋Phase4 RAW引导NAF基线 | 26.554489664752047 | 0.8435530973672867 |
| 同一基线＋自适应3DLUT（最终冻结方案） | **26.707356438407885** | **0.8435157579481601** |
| 最终方案减基线 | +0.15286677365583756 | -0.00003733941912653638 |

这是**小幅提高PSNR、SSIM略有下降**的权衡，不是所有指标都改善，也未达到30dB。全1000包含832张训练图、92张被官方基础模型预训练见过的验证图，因此是课程指定的全数据集成绩，**不能写成独立测试或泛化成绩**。

最终候选在读取新最终测试分数之前，根据完整92val和预先规定的门槛冻结。即使后来发现留出测试退步，也没有根据测试分数重新选模型、删图或追加训练。旧基线、旧测试封存均保留。

提交依据：[1000行逐图指标](assets/enhancement/submission1000_20261024/metrics.csv)、[完整精度汇总](assets/enhancement/submission1000_20261024/summary.json)、[冻结绑定](assets/enhancement/submission1000_20261024/binding.json)、[云端校验记录](assets/enhancement/submission1000_20261024/verification.json)、[本机独立复核](assets/enhancement/submission1000_20261024/local_verification.json)。

## 2. 数据读取与评价定义

- 数据：Canon `0.0001`，1000对RAW与GT，固定划分为832train / 92val / 76test。Sony不混入这1000张统计。
- RAW：3840×2160、uint16；黑电平512、白电平16383；扣除黑电平、归一化并打包RGGB四通道。1000对审计0错误、0重复，检查18,669,533,292字节。
- PSNR、SSIM：在**PNG量化前、3840×2160原生浮点RGB**上计算，RGB范围0–1。每张图先算指标，然后1000个值求平均；不先合并MSE，也不把三个split的均值等权平均。
- 颜色指标沿用最长边960的既定协议，LPIPS保持既有实现。512×288缩略图仅用于LUT训练，不充当最终评价图。
- 推理不读GT；每图预测固定后才读取GT评分。不使用测试GT均值调整亮度、曝光或颜色，不筛掉困难图。
- 1000个唯一key、正确split与顺序、全部8项有限指标和单一模型绑定全部核对后，才生成总体。

## 3. 已实现方法及本轮改进

前期方法包括传统自动ISP、RAW→RGB Residual U-Net、SIED官方EMA恢复、全场景曝光/白平衡/颜色矩阵精修，以及RAW引导NAF残差精修。模型结构、损失和Phase1–5历史过程见[增强冲刺报告](enhancement_sprint_20261024.md)、[前期增强报告](enhancement_final_report.md)。这些旧方法并未都按本次协议重新评价1000张，不能把其历史76/87测试成绩填进本报告的1000张主表。

本轮固定SIED与Phase4 NAF，只研究低成本的附加模块：

1. **TLC512，无需训练的局部聚合对照**：仅将NAF结构分支的全局聚合改为局部均值，保留全图光度；采用FP32积分图。固定12val的PSNR从26.952234降为26.853515dB，未通过门槛，不扩大为92图试验，不进入最终方案。
2. **内容自适应3DLUT**：借鉴[图像自适应3DLUT作者代码](https://github.com/HuiZeng/Image-Adaptive-3DLUT)，使用17³格点、3基底、小CNN控制器、受限RGB残差0.12，共63,324参数。全场景area512→area128上下文只预测一次系数，全分辨率分块查询同一色表，避免逐块改变色调。恒等初始化，只训练832train上的FP32缩略对；FP32、fused AdamW、逐图RGB log-MSE及平滑/单调正则，最多1500步。按预设更新门槛选中1250步。
3. **ICCV启发的成对消融**：参考[SVDLUT论文](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Lightweight_and_Fast_Real-time_Image_Enhancement_via_Decomposition_of_the_ICCV_2025_paper.html)及[作者代码](https://github.com/WontaeaeKim/SVDLUT)，独立实现RG/RB/GB低秩表。global3d与pairwise rank4同控制器初值、私有batch随机序列、FP32、loss、优化器与1000步预算；仅改变表参数化。rank4为22,779参数，比63,324少约64%，但没有本次质量或速度优势。它**不是完整SVDLUT复现**：没有论文的空间/双边分支或自定义CUDA核。

所有新候选必须在原生验证上满足PSNR至少+0.02dB、SSIM退化不超过0.002、全部颜色与LPIPS误差增幅不超过2%。通过固定12val后才做完整92val。两组研究已结束，未追加第三组；GT-Mean Loss只阅读、没有加入本轮训练或推理。

### 完整92val的模型选择与消融

| 方案 | 所选步数 / 上限 | 参数量（附加模块） | PSNR / dB | SSIM |
|---|---:|---:|---:|---:|
| 冻结基线 | — | 0 | 26.63693109 | 0.85627188 |
| 第一组自适应3DLUT | 1250 / 1500 | 63,324 | 26.84298008 | 0.85626056 |
| 第二组global3d | 750 / 1000 | 63,324 | 26.79978926 | 0.85630565 |
| 第二组pairwise rank4 | 1000 / 1000 | 22,779 | 26.78292994 | 0.85619018 |

按完整val门槛后PSNR最高、精确同分以SSIM打破平局的既定规则，选择第一组1250步LUT。22:54:58冻结绑定，之后才开始新的1000张验收；没有依据1000/test重新挑选。第一组1500步与第二组1000步的批次/调度不同，不能将二者分数差异因果归于显存常驻或低秩。

## 4. 最终1000张结果与分项

| 指标 | 基线1000 | 最终1000 | 变化（最终－基线） |
|---|---:|---:|---:|
| PSNR↑ / dB | 26.554490 | 26.707356 | +0.152867 |
| SSIM↑ | 0.84355310 | 0.84351576 | -0.00003734 |
| DeltaE2000↓ | 4.595405 | 4.575388 | -0.020018 |
| 逐图DeltaE P95均值↓ | 8.165957 | 8.169354 | +0.003397 |
| 亮度MAE↓ | 4.458091 | 4.433654 | -0.024438 |
| 饱和度MAE↓ | 0.03768485 | 0.03725874 | -0.00042611 |
| 色相误差↓ / 度 | 11.926284 | 11.887847 | -0.038438 |
| LPIPS↓ | 0.32444085 | 0.32435590 | -0.00008495 |

总体PSNR、平均色差、亮度、饱和度、色相和LPIPS小幅改善；SSIM、色差P95略退。逐图PSNR有524张提高、476张下降，中位变化+0.105085dB，最差-1.752339dB、最好+4.137213dB；全部计入。SSIM有497张提高、503张下降。

| 分项 | 张数 | 基线PSNR | 最终PSNR | 基线SSIM | 最终SSIM |
|---|---:|---:|---:|---:|---:|
| train | 832 | 26.84302437 | 27.01191531 | 0.84742199 | 0.84743960 |
| val | 92 | 26.63693109 | 26.84298008 | 0.85627188 | 0.85626056 |
| test | 76 | 23.29599643 | 23.20906227 | 0.78580246 | 0.78513208 |

**留出76test的PSNR下降0.086934dB、SSIM下降约0.00067038；其中22张PSNR提高、54张下降。** 这说明本次色调校正在训练/验证分布上取得的平均收益没有转化为留出测试收益。验证集又存在官方预训练暴露，不能宣传“泛化增强成功”。保留旧基线作为对照；本轮最终冻结方案没有根据测试结果回退换表。各分项全部8项精确值在summary中。

新最终1000全在同一RTX4090上，以同一所选流水线评价。历史基线832train+92val来自4090，76test复用4080SUPER vGPU封存结果；比较存在硬件边界，因此不把极小差异都归于算法。最终92val逐图8项指标与冻结前验证完全一致。

## 5. 训练与运行效率：实测及边界

- 保存924张原样FP32盲扩散基础预测，并复用于研究/最终train与val，避免为了每个候选重复生成20步、3次采样的扩散结果。约86GiB基础缓存与约3GiB训练缩略对可重新生成，未下载到空间有限的Mac；逐图hash、seed与生成代码保留。
- 832张FP32训练对常驻GPU为2,944,401,408字节，预加载2.448222秒。同一global3d真实batch短测，pinned DataLoader为0.03469200秒/步，常驻0.01299559秒/步，约2.669521倍吞吐；首预测完全一致，20步后参数最大差1.0766e-6。**不包含完整原生验证和预加载摊销，不是整次实验快2.67倍。**
- 同预算1000次更新，global3d平均0.01239057秒/步，低秩0.01373014秒/步，低秩反而慢约10.81%；含周期验证阶段251.85396秒与252.78675秒，均不含之后的完整92val。
- NAF的channels-last＋FP16＋fused AdamW合成短测为1.168倍，单独channels-last反而更慢。它与LUT实数据短测不是同一模型或范围，不能相乘。
- 最终1000评价01:28:07结束。train/val从FP32盲基础缓存重放，test包含新扩散；不能把混合计时称为统一端到端推理延迟。原始时长逐图保留在CSV。

速度证据及一手论文适用边界见[效率与研究记录](enhancement_efficiency_research_20261024.md)。新增完整回归176项通过；最终1000约束相关20项针对性测试通过，JUnit均已归档。没有为凑分换精度、缩图评分或修改测试清单。

## 6. 文件与复现

### 6.1 本机无需GPU复算提交均值

在仓库根目录运行以下命令，只用Python标准库，不需要已删除的`.venv`或原始大数据：

```bash
python3 - <<'PY'
import csv, json, math, statistics
from pathlib import Path
p = Path('reports/assets/enhancement/submission1000_20261024')
with (p / 'metrics.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == len({r['key'] for r in rows}) == 1000
assert {s: sum(r['split'] == s for r in rows) for s in ('train','val','test')} == {'train':832,'val':92,'test':76}
result = {}
for column in ('psnr', 'ssim'):
    values = [float(r[column]) for r in rows]
    assert all(math.isfinite(v) for v in values)
    result[column + '_mean'] = statistics.fmean(values)
print(json.dumps(result, indent=2))
PY
```

这复现的是**已封存逐图指标的汇总**，不是重新推理。不能仅用8-bit展示PNG重新评分，再要求与量化前浮点分数逐位相同。

### 6.2 重新执行完整原生模型评价

本轮实际环境为Python3.12.3、Torch2.8.0+cu128、torchvision0.23.0+cu128、NumPy2.3.2、Pillow11.3.0、scikit-image0.26.0、LPIPS0.1.4；GPU RTX4090。原协议为FP16、20 DDIM步、ensemble3，NAF tile768/overlap96；第i张原split内图第j次种子为3407+i+j×1000003。LUT训练和查询保留既定FP32上下文。

重要文件：

| 内容 | 仓库文件或本机路径 |
|---|---|
| 固定基线评价/可再生缓存生成 | `cloud/evaluate_sied1000.py` |
| 第一组TLC/LUT研究 | `cloud/research_sied_color.py` |
| 第二组成对消融与真实批次基准 | `cloud/research_sied_iccv.py` |
| 最终单一流水线1000评价 | `cloud/finalize_sied_submission1000.py` |
| LUT模型 | `src/dip_course/enhancement/adaptive_lut.py`、`factorized_lut.py` |
| 基础恢复/精修/指标 | `src/dip_course/cli/infer_official_sied.py`、`enhancement/refiner.py`、`metrics.py` |
| 全部新1000展示PNG、逐图JSON、日志 | `outputs/sied1000_20261024/submission1000_native/`、同级`logs/` |
| 最终LUT权重及研究资料 | `outputs/sied1000_20261024/research/color_lut_tlc/best_lut.pt`及两个研究目录 |
| 官方SIED与冻结Phase4权重 | `checkpoints/sied30_resume_snapshot/official/canon_0.0001.pt`、`final/best.pt` |

完整推理还需要1000 RAW/GT、固定manifest、官方`third_party/SIED-official`代码、官方权重、Phase4权重、旧`final/selection.json`及测试封存、固定基线证据/逐图FP32缓存、两组研究result及完整92val指标和相应权重。**仅克隆GitHub不能自动获得大数据/权重/缓存。** 大模型和数据未塞进普通Git提交；唯一新模型、全部新图片和实验资料在本机备份。原数据与大缓存仍在已关闭实例的磁盘，未删除。

在已有本轮资料的云端副本中，先确保`project`位于所指定root下且安装相同依赖，然后执行：

```bash
cd /root/autodl-tmp/sied30_eval1000/project
/root/autodl-tmp/cv-homework/.venv/bin/python cloud/finalize_sied_submission1000.py \
  --root /root/autodl-tmp/sied30_eval1000
```

已有1000行时该命令校验并断点跳过，不意味着重新推理了一遍。要做独立重跑，应准备**另一个完整资料副本**并在其新输出目录执行，不能删除或覆盖本次封存。新机器需自行准备环境；跨GPU/CUDA版本不保证逐位相同。缓存缺失时须先按`evaluate_sied1000.py --cache-base-fp32 --cache-color-pairs`及历史基线的依赖/封存要求重建，不能用8-bit PNG冒充缓存。本次没有因此重新开机或开始新训练。

### 6.3 校验与可视化

本机独立解析1000行CSV/JSON，以`statistics.fmean`复算全部8项总体和分项均值，误差阈值1e-12；核对1000个唯一key、832/92/76清单、索引/种子/单一绑定、所有源码与三组主权重SHA。1000张PNG逐图SHA和3840×2160尺寸全部通过；完整提交目录rsync checksum dry-run零差异。

可以打开`outputs/sied1000_20261024/submission1000_native/images/`查看train、val、test最终增强图；同编号的基线展示位于`evaluation1000_native/images/`，历史test图保留在旧封存备份。课程早期RAW/ISP/GT对比及可视化方法见前期增强报告。本轮不从8-bit展示图倒推浮点成绩。

最终LUT SHA256：`2cfd72ec81addd4a091ea199c613c4bc47e27e562a098971db7302510a1db609`。CSV SHA256：`a99994aeeeae9f4c86da9074e776525f489b23d7cb79f2ef1800f77e43bc5041`。其余哈希在绑定与复核记录中。

## 7. 下一步方向（本轮未追加执行）

最需要解决的不是继续增加步数，而是**训练/验证收益与未见场景退化之间的差距**。后续应取得真正未被官方预训练见过的开发验证数据，在训练场景内检验全图色调偏移与局部纹理误差，单独消融损失、残差幅度和空间感知色彩映射；每次只变一个因素，并同时约束SSIM与颜色。现有76test不能继续当作调参验证集。

效率上优先保持已验证的FP32缓存复用/显存常驻，先做包含预载、数据读取与完整验证的端到端计时。低秩参数更少但本次更慢，不能只凭参数量决定部署。扩散少步蒸馏等方向需要新的受控实验与授权，不能直接照搬其他数据集30dB或速度。

## 8. 备份与关机

所有本轮GPU任务已结束，唯一权重、全部1000图片、逐图指标和日志已异机校验。37497于2026-10-25 **01:37:24**重新确认PID1184确为`/bin/supervisord -c /init/supervisor/supervisor.ini`且无GPU作业后正常关闭，两次SSH复查均离线。未调用会清Trash的shutdown包装，保留实例磁盘，不销毁实例、不删除原资料。旧18868先前已于19:12关闭，未重开。

定时监测`sied-1000`已删除；本轮临时防睡眠进程已停止，未修改系统电源设置。未独立检查平台控制台账单，不能据SSH离线保证无任何磁盘保留费。具体证据见[收尾记录](assets/enhancement/submission1000_20261024/closeout.json)。
