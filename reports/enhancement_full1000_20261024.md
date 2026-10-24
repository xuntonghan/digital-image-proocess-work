# Canon 1000张固定模型补充评价（已完成；改进研究继续）

## 用户范围与交付

> 2026-10-24最新补充授权：“好的继续，同时寻找新的提高方法，并提高训练效率”。在不改变下文冻结1000张基线评价的前提下，允许研究与效率优化，以及有严格预算、只用train/val的小规模对照。**不得在1000评价刚结束时无条件关掉仍有本轮研究作业的新机**；先读`reports/enhancement_efficiency_research_20261024.md`并核实全部授权作业是否结束。此更新覆盖下面旧的“不训练/重新训练先问”字样，仅不授权无上限重训、新租实例、扩容或改测试。

> 约19:49用户再次要求“继续改进算法，加快训练速度与训练效率，同时一步一步参考iccv相关顶刊做出提高”。本轮新增**一组**受控ICCV启发低秩LUT对照，排在原有TLC/LUT实验之后；具体预算见效率研究报告。不得只等第一组研究结束就关机，也不得据此无限追加训练。冻结1000评价代码、权重、seed、协议与76测试封存均不变。

仅项目一SIED增强；几何纠正与压缩继续暂停。用户在2026-10-24提供最新旧实例与新4090实例，用于迁移资料并补齐1000张整体评价。**没有授权新一轮盲目训练，不以训练集成绩替代独立测试。** 继续遵循此前“持续监测、完成备份后关闭云机”的要求。

固定模型1000张评价已于21:53:59完成，总体平均PSNR为**26.554489664752047dB**；本机随后独立复算并核对1000唯一key、832/92/76划分、全部8项逐图均值、76封存结果逐字段一致，以及924张新展示PNG的SHA256。**总体包含训练/预训练暴露，不能当作独立测试成绩。** 独立76测试PSNR仍为23.295996430040688dB。两组有界改进研究继续，尚未关闭新机。

## 实例与路径（无密码）

- 资料源：`ssh -i /Users/hanxuntong/.ssh/codex_gala_autodl -p 18868 -o BatchMode=yes root@connect.nmb1.seetacloud.com`，已于19:12关闭并验证SSH离线。其`/root/autodl-tmp/sied30`资料盘保留1000 RAW、1000 GT、924基础PNG；无需重新开机。旧45990也不需要开机。
- 新评价机：`ssh -i /Users/hanxuntong/.ssh/codex_gala_autodl -p 37497 -o BatchMode=yes root@connect.bjb2.seetacloud.com`，RTX4090、24564MiB显存、初始150GiB空闲数据盘。
- 新根目录：`/root/autodl-tmp/sied30_eval1000`；代码`project`，Python`/root/autodl-tmp/cv-homework/.venv/bin/python`（Python3.12.3，Torch2.8.0+cu128）。系统Python不在默认PATH，必须用上述绝对路径。
- 本机操作根：`/Users/hanxuntong/Documents/ChatGPT/cv宿題`。结果预定增量备份：`outputs/sied1000_20261024/`。
- 新机到旧机临时迁移密钥：新机`/root/.ssh/sied_eval20261024_ed25519`。旧机authorized_keys中新加且仅供迁移的一行comment为`sied-eval-20261024`；完成后可精确撤销这一行，不动其他密钥。密码不写入脚本或日志。

## 固定评价协议

不训练、不重新选模型。冻结昨天Phase4 refiner SHA256 `337acc11705e8ae1f170d22263161b210a0ed0d4272a647db4e82ccbb611f2d8`；官方权重SHA256 `40014f1f71dfdcd9a8969074097f675a54d2598aec26b183c0874bfcd5c26238`。

1. 832 train与92 val执行全新盲推理：FP16，原生3840×2160，20 DDIM steps，ensemble3，NAF tile768/overlap96。保留各原始split内索引，seed=`3407+split_index+j*1000003`，不改用全1000清单索引。
2. 76 test不再推理，复用已封存CSV的浮点指标。先校验selection、权重、CSV、summary和manifest哈希，再逐图核对key、顺序、seed及8项均值；任一不一致即失败。
3. 全部新指标来自PNG量化前的float预测；PSNR/SSIM在原生尺寸，颜色指标维持项目既定最长边960协议，LPIPS同既有实现。旧924缓存为8-bit PNG，不将其输入精修所得结果冒充原浮点流水线成绩；它们保留旧机，不为这次补评搬运。43GiB Phase5缓存也不搬运。
4. 按1000张逐图指标算术平均，同时报告832/92/76分项；必须核对三split精确覆盖canon_all且无重复/漏图。不能把三个split均值直接平均，也不能静默过滤困难图、错误或非有限值。
5. 披露训练/预训练暴露：832图用于精修训练，92验证图被官方基础模型预训练见过。全1000只是描述性总体，不是泛化成绩。新train/val在4090，历史test在4080SUPER vGPU，硬件差异披露；混合耗时不冒充单卡性能。

## 安全迁移与执行

代码以本机已提交HEAD的git archive部署，未搬运未提交的其他项目修改。新脚本：

- `cloud/migrate_sied_full_evaluation.sh`：仅迁移官方代码、权重、final、Phase4 val资料和1000对数据。无删除选项；每目录rsync checksum dry-run必须无差异才写`logs/MIGRATION_VERIFIED`。
- `cloud/migrate_sied_raw_compressed.sh`：并行流式zstd压缩RAW，节省无卡机跨区传输时间；与最终RAW rsync使用同一锁避免同时覆盖。中断后保留文件，最终rsync校验补齐。
- `cloud/evaluate_sied1000.py`：不可混用的binding、原子逐图JSON及PNG SHA256、断点续算、严格总体汇总；只有完整1000才写summary和verification。
- `cloud/run_sied_full_evaluation.sh`：等待迁移校验最多2小时，安装当前代码、运行针对性测试、全量1000对审计，3张pilot成功后继续924新推理+76复用。完成后等待本机备份，不自行关闭唯一产物所在机器。

后台screen：`sied_eval_migration`、`sied_eval_raw`、`sied_eval1000`。主迁移首次中断仅为移除不需要的缓存传输并加入RAW互斥，`--partial`保留断点；当前使用`logs/migration_resumed.log`。不要再启动旧three_task、phase1/2/4/5训练脚本。

观察日志：

- `logs/migration_resumed.log`、`logs/raw_stream.log`、`logs/verify_*.txt`。
- `logs/evaluation.log`、`logs/evaluation_status.txt`、`logs/data_audit.json`。
- `evaluation1000_native/progress.json`、`binding.json`、`rows/train/*.json`、`rows/val/*.json`、`summary.json`、`metrics.csv`、`verification.json`。

2026-10-24约18:30：18项针对性测试通过，ruff检查通过；部署与迁移运行中，评价wrapper等待校验标记，GPU尚未执行正式评价。环境从旧venv复制后原editable路径会暂时指向旧root，wrapper已强制重新`pip install --no-deps --no-build-isolation -e .`，避免ModuleNotFoundError。

约18:34已再次重装editable并确认`dip_course.__file__`指向新root；LPIPS AlexNet缓存已迁移，SHA256为`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`。本聊天10分钟heartbeat已成功创建，ID`sied-1000`，状态ACTIVE；旧automation-4没有重建，不存在两个增强监测重复执行。

最新效率优化：在任何1000评价binding生成前，已更新评价程序可同时保留924张**原始float32**基础盲预测（约85.7GiB），不改预测/指标数值，绝不缓存测试。这些文件用于后续训练/验证，避免重新生成DDIM；每图原子写入并记录SHA256，全场景RAW均值也记录于行数据。至少保留15GiB空余，预估150GiB数据盘可容纳本轮约113GiB总需求，不扩容。GPU基准测试与正式评价共用`logs/gpu_work.lock`互斥，不能并发抢卡。代码更新后28项针对性测试与ruff通过，等待迁移的wrapper已重新启动；已启动`sied_efficiency_bench`进行有界的合成数据速度基准，不改已选权重。具体结果待真实日志。

后续补充：已部署FP32成对area512色调训练缓存（约3GiB，仍仅train/val），含研究派生产物预计峰值约120GiB，不影响冻结评价的预测/指标。新LUT/TLC有界排程等待完整基线后运行。19:07已完成169项全套测试；基准实际结果与扩展范围见`enhancement_efficiency_research_20261024.md`。所有基线代码变更均在首个binding产生前完成，未更改任何已完成的基线行。

### 19:08–19:12迁移完成与正式评价开始

- 19:08:39生成`MIGRATION_VERIFIED`，六个目录的checksum dry-run记录均为0字节，1000 RAW与1000 GT已完整到新机。官方/精修模型SHA再次匹配冻结值。
- 全量数据审计`valid=true`、1000对、0错误、0重复，检查18,669,533,292字节。3张pilot通过后，924张正式train/val新推理已启动；约19:11已完成13张，步长约10–11秒/图，尚无全1000均值，不以部分图外推分数。
- 实物抽查：基础缓存`(2160,3840,3) float32`，色调缓存`(2,288,512,3) float32`；未缓存test。新结果已开始增量下载本机，明确排除`base_float32/`与`color_pairs/`大型可再生成中间文件。
- 旧18868确认只有平台Jupyter/tensorboard、无迁移或训练进程后，精确撤销本次临时迁移公钥行；19:12核对PID796的`/bin/supervisord -c /init/supervisor/supervisor.ini`身份后执行正常等效关机。保留旧数据盘和Trash、不销毁实例；SSH离线验证另记录，未独立核对控制台账单。

随后首次SSH复查返回连接关闭（exit255）；新机评价继续独立运行，无需源机在线。

### 19:58后续改进排程

实查新机已完成279/924新推理，日志持续更新，暂无1000均值。新增文件仅为独立低秩模型、实验、排程与测试，不修改`binding.json`登记的基线源码。新screen预定`sied_iccv_research`，日志`logs/iccv_research.log`，输出`research/iccv_pairwise/`；必须等`research/color_lut_tlc/result.json`存在并取得同一个GPU锁后才运行。最终收尾要同时检查这组新实验。原生FP32基础缓存仍不全量下载本机；新增模型、验证展示、逐图指标与日志必须备份校验。

20:56例行检查：云端603/924新推理、日志持续更新，两组研究等待进程均在；云盘剩余67GiB，仍正常。Mac电池降至26%、未接电，系统预计仅剩1小时12分，可能无法覆盖后续研究与备份关机。已向用户发出接电并保持应用/网络运行的提醒；没有擅自改电源设置或提前关云机。仅当电量继续降到紧急程度或状态有实质变化时再提醒，勿每10分钟重复通知。

## 固定1000张的最终统计与独立复核

以下为PNG量化前浮点预测的逐图指标算术平均。↑越大越好，↓越小越好；颜色指标沿用最长边960的既定协议，PSNR/SSIM在原生3840×2160计算。

| 指标 | 总体1000 | train 832 | val 92 | test 76（封存复用） |
|---|---:|---:|---:|---:|
| PSNR↑ / dB | 26.554490 | 26.843024 | 26.636931 | 23.295996 |
| SSIM↑ | 0.843553 | 0.847422 | 0.856272 | 0.785802 |
| DeltaE2000↓ | 4.595405 | 4.393579 | 4.496330 | 6.924813 |
| 逐图DeltaE P95均值↓ | 8.165957 | 7.792339 | 7.831748 | 12.660655 |
| 亮度MAE↓ | 4.458091 | 4.287994 | 4.455134 | 6.323792 |
| 饱和度MAE↓ | 0.037685 | 0.035325 | 0.033742 | 0.068297 |
| 色相误差↓ / 度 | 11.926284 | 10.831856 | 11.346907 | 24.608742 |
| LPIPS↓ | 0.324441 | 0.320581 | 0.313367 | 0.380104 |

可提交的轻量证据：[完整逐图CSV](assets/enhancement/full1000_20261024/metrics.csv)、[完整精度summary](assets/enhancement/full1000_20261024/summary.json)、[实验绑定](assets/enhancement/full1000_20261024/binding.json)、[云端封存receipt](assets/enhancement/full1000_20261024/verification.json)、[本机独立复核receipt](assets/enhancement/full1000_20261024/local_verification.json)。原生924张展示图与逐图JSON备份于本机`outputs/sied1000_20261024/evaluation1000_native/`；约86GiB可再生基础缓存不下载本机。

独立复核未调用评价程序的`summarize`：本机用标准库CSV/JSON解析、`statistics.fmean`复算全部8项均值（误差阈值1e-12），校验source与三个split的manifest SHA、key顺序、split-local索引与seed。76test的每个原始CSV字段（包括指标、种子、耗时与路径）与昨日封存文件逐字一致；新924行的JSON/CSV字段相符，924张PNG的SHA全部一致。CSV SHA256为`3b005a7a76b0b46629ce56196b126e1b84bdbac177f261f3314c776af85d6691`。

这仍未达到独立测试30dB目标。不能拿包含832训练图的26.55dB包装成“测试已提高到26.55”；也不能把新val与昨日val约0.000036dB的混合GPU数值差异当成算法收益。后续LUT/TLC研究与本固定基线分开报告，绝不回写此CSV或昨日76test。

Mac已接上AC电源并在充电（21:54复查），此前电量风险暂解除。云端第一组研究已开始，第二组仍等待；所有授权研究和新产物备份核验完成前不关闭37497。

后续22:17第一组LUT1250步通过完整92val：PSNR从26.636931提高至26.842980dB（+0.206049），但SSIM略降、色差P95略升，并非全部指标改善；50张PSNR提高、42张下降。数据与权重已异机复核备份，详见效率研究报告。该92val仍有预训练暴露，不是独立测试成绩。第二组对照仍在运行，以上1000固定统计和76封存结果绝不回写。

## 收尾要求

结合真实进程、日志更新时间、逐图数量、GPU与磁盘判断，不只看status字符串。迁移确认后及时关闭无卡旧机（保留数据盘，不销毁）；不要在仍传文件时关闭。新机评价完成后将完整新结果、逐图PNG、日志、代码与哈希增量备份本机，独立复算1000与三分项均值，确认test与原封存76逐项一致；再关闭37497新机。关机前确认无其他用户GPU作业，并校验supervisord身份；AutoDL shutdown包装会清Trash，不直接调用，不删除数据。不能只杀评价进程却让实例继续空转。

若评价错误，只修复本评价范围内的兼容性/断点问题，不更改模型、数据划分、原生指标协议或测试结果来凑分。需新付费、扩容、重新训练时先问用户。完整证据备份后更新本报告并依既有授权提交推送私有仓库，只提交本轮增强范围，不把其他项目未提交文件夹带进去。完成关机后停止本轮监测。不要自动开机或删除原始数据。
