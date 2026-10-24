# SIED新方法与训练效率研究（2026-10-24，进行中）

用户新增要求：继续1000张统计，同时寻找新的提高方法、提高训练效率。只做项目一，不动几何/压缩，不为30dB承诺结果或无限追加训练。保持昨天最终模型、76测试及本轮1000张基线统计不变。

## 已核实的实现问题与待验证假设

- 昨天Phase4/5后续训练没有有效原生PSNR收益，不能只加步数。现有`_loss_components`的PSNR profile仍计算Charbonnier、SSIM、edge、颜色矩、直方图、identity和参数惩罚。没有因果消融时，不能认定它们一定导致退化，但计算开销与目标失配值得测试。
- NAFNet作者实现`PSNRLoss`是每样本RGB MSE取对数再平均。它不同于先跨图平均MSE；本项目现在新增独立的FP32 RGB log-MSE损失。Patch目标仍只是全图平均PSNR的代理，不能宣传为完全等价。
- NAF的通道注意力全局均值在512训练块和768推理块上统计范围不同。Phase5统一512不是完整TLC。新增可选TLC式局部均值，只转换结构分支的NAFBlock，不转换全图光度。FP32积分图避免FP16大窗口溢出。现有baseline路径和权重文件不变。
- 采用TLC文献常用的1.5倍512窗口会得到768，恰好与现有768tile一样而退化为全局池化。因此本轮先单独验证明确的512窗口，并以大窗口等价单测作对照，不能把“换名字但数值没变”当提升。

## 已实现、正在实测的效率优化

1. 正式1000评价的924张train/val推理顺便原样缓存FP32 SIED基础输出，约85.7GiB；同一轮不再额外运行一次扩散器来建训练输入。每图带SHA、split-local seed与全场景RAW均值；永不读取GT生成基础预测，永不建立测试训练缓存。低于15GiB余量即停止，不擅自扩容。
2. 合成数据微基准`cloud/benchmark_refiner_efficiency.py`：同一Phase4权重副本，512 RGB patch、batch1、5步预热+20步计时，比较NCHW/FP16/defaultAdamW、channels_last/FP16/defaultAdamW、channels_last/FP16/fusedAdamW、channels_last/BF16/fusedAdamW。报告真实更新数、mean/median步时、峰值显存和初始数值差异。只用合成数据测核函数吞吐，**不宣称真实端到端训练速度或质量提高**。
3. GPU作业共享`logs/gpu_work.lock`；微基准与正式评价不同时运行。不因为CPU指标评价的间歇GPU0%而重启。
4. `src/dip_course/enhancement/efficiency.py`提供PSNR损失与独立TLC实现；六项新单测覆盖逐图log顺序、FP16零误差稳定性、滑窗等价/梯度、FP16前缀和不溢出、全局退化与保留光度/检查点key。包含其他增强相关测试共28项通过，ruff通过。

新机/root路径及备份要求同`enhancement_full1000_20261024.md`。微基准screen`sied_efficiency_bench`，日志`logs/efficiency_benchmark.log`，结果`research/efficiency/benchmark.json`，未完成先看`benchmark_partial.json`。单配置超过240秒失败停下，不无限跑。

## 小实验顺序与上限

先完成固定1000描述性评价并核验；再只用train/val的FP32缓存比较未改模型与TLC512，不重新运行SIED、不读取76测试GT、不根据全1000总体挑模型。优先固定12张验证pilot；无明确PSNR收益且颜色/SSIM不合格就记录负结果，不扩大。通过后才验证完整92张。纯PSNR损失的训练小实验需对齐全图光度与RAW增益，使用独立输出目录，单次上限1500步，不得自动扩大；比较对象必须同输入精度、tile、图像和验证协议。原模型保留作兜底。任何速度候选只有在真实数据数值稳定后才用于新训练，不修改冻结评价的内存布局或精度。

本文件目前记录实现/方案，不声明TLC或新损失已改善分数；训练小实验尚未启动。先实测速度与建立完整浮点缓存，避免无依据地继续烧卡。

## 本轮已部署的轻量色调小实验

最新实现基于TPAMI图像自适应3DLUT思想，**不声称完整复现原论文**：17³格点、3个基底、RGB残差上限0.12、内容自适应小CNN；共63,324可训练参数。从恒等映射初始化，旧SIED/NAF权重完全冻结。使用PyTorch原生5D grid_sample做三线性插值，不安装旧CUDA扩展，也不下载FiveK权重。

1000基线评价在原生浮点增强结果与GT上额外生成512×288 float32成对训练缩略图，仅832train/92val，约3GiB。这个尺寸只是色调模型训练输入，**绝不当作原生4K最终评分**。系数始终由全场景area512上下文预测，原生输出按行分块查询同一个LUT，避免每块改变色调。源预测仍先盲生成，缓存GT只在其后的评价/训练对准备中读取。

`cloud/research_sied_color.py`按固定seed20261024抽12张val：先做TLC512对照；再只用832train对训练LUT，batch16、FP32、fusedAdamW、RGB log-MSE加平滑/单调约束。每250步原生12图计算PSNR、SSIM、LPIPS和全部颜色指标；最多1500步，750步后连续3次无合格收益即停，含周期验证的训练阶段最多45分钟。合格要求PSNR至少+0.02dB、SSIM最多退化0.002、颜色/LPIPS各最多增加2%；通过pilot才评价完整92val。不会再跑76测试或改昨日最终权重。记录每步计算/含数据取用时长，不把合成吞吐当端到端训练时间。

部署排程：`sied_color_research`运行`cloud/run_sied_research_after_eval.sh`，等完整基线verification与完成状态后才开始；与基线共用GPU锁。仅运行这一轮，不自动扩大训练预算。输出`research/color_lut_tlc/`，包括实验绑定、代码hash、逐图验证、训练log、候选权重SHA、结果。没有收益则保留基线，不包装成成功。

## 已完成的效率测量（不是质量成绩）

同一4090、512块、batch1，5步预热+20步计时，各模式20次有效更新：

| 模式 | 平均秒/步 | 合成吞吐相对基线 | 初始输出最大绝对差 |
|---|---:|---:|---:|
| NCHW/FP16/default AdamW | 0.122135 | 1.000× | 0 |
| channels-last/FP16/default AdamW | 0.129088 | 0.946× | 0.0001831 |
| channels-last/FP16/fused AdamW | 0.104535 | 1.168× | 0.0001831 |
| channels-last/BF16/fused AdamW | 0.101450 | 1.204× | 0.0011597 |

FP16融合方案的计算吞吐约+16.84%，步时约-14.41%；单独换channels-last反而慢。BF16偏差更大，不因速度稍快就换入冻结评价。这是短合成微基准，排除DataLoader、EMA和验证，必须用真实数据再测，不能承诺整次训练快17%。原始结果已备份`outputs/sied1000_20261024/research/efficiency/benchmark.json`。

另在CUDA上完成新LUT的5步合成训练、梯度有限值和原生2160×3840查询烟雾测试：63,324参数，原生输出尺寸正确、无NaN。总耗时约0.447秒包含预热，**不把这个数字当4K单图推理延迟或真实训练效率**。结果在同目录`lut_cuda_smoke.json`。包含新方法的33项针对性测试通过；新机干净代码快照的全部测试也通过，最终数量另核对，既有Pillow弃用警告不影响本轮。

## 进一步阅读的低成本候选

- [图像自适应3DLUT，TPAMI2022，作者代码](https://github.com/HuiZeng/Image-Adaptive-3DLUT)：以低分辨率上下文学习全分辨率色调映射。只借鉴机制，FiveK上的分数和速度不作为本SIED项目实测。
- [SVDLUT，ICCV2025论文](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Lightweight_and_Fast_Real-time_Image_Enhancement_via_Decomposition_of_the_ICCV_2025_paper.html)、[作者代码](https://github.com/WontaeaeKim/SVDLUT)：用低维分解降低空间感知LUT复杂度。它解决全局LUT无法区分同色不同位置的限制，但带来更复杂特征和扩展依赖；本轮先做简化有界LUT，未宣称实现SVDLUT。

本机当前仅约19GiB可用：不会把约86GiB可再生成缓存下载本机。备份指标、逐图展示PNG、日志和唯一新模型；缓存保留逐图hash、seed和生成代码。禁止为腾空间擅自删原资料或其他项目文件。

### 代码验证与部署记录

截至19:07，新机干净代码与新增文件共**169项测试通过，0失败/0错误/0跳过**；[JUnit证据](assets/enhancement/efficiency_20261024/regression.xml)。既有Pillow弃用警告来自原压缩测试，本轮未改动该项目。新增缓存单测确认AMP预测先转FP32再缩略，避免FP16池化引入新的上下文偏差；新增研究门槛单测覆盖LPIPS、颜色、SSIM、非有限值、数量与低精度/错误尺寸缓存拒绝。

[合成微基准](assets/enhancement/efficiency_20261024/benchmark.json)、[LUT CUDA烟雾测试](assets/enhancement/efficiency_20261024/lut_cuda_smoke.json)随轻量资料保存。新研究会再次核对完整基线CSV/binding、split manifest与逐图缓存SHA，逐图基线重放PSNR偏差超过0.001dB即失败；不把推理精度/协议变化包装成新算法收益。预计云盘峰值约120GiB，运行时继续保留15GiB余量。

定时任务`sied-1000`已更新为“固定1000评价→有界TLC/LUT研究→异机备份校验→关机”，仍每10分钟，无实质变化保持安静。

## 一手依据与适用边界

- NAFNet（ECCV2022）：https://github.com/megvii-research/NAFNet ，作者PSNRLoss在`basicsr/models/losses/losses.py`，局部池化在`basicsr/models/archs/local_arch.py`。本项目自主实现相同数学思想，不安装旧BasicSR全家桶、不导入旧CUDA扩展。
- TLC（ECCV2022）：https://github.com/megvii-research/TLC ，支持无需再训练先解决全局聚合分布差异的方向，但论文中去模糊数据的收益不能转成SIED增益保证。
- ReDDiT（CVPR2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Lan_Efficient_Diffusion_as_Low_Light_Enhancer_CVPR_2025_paper.html ；作者代码https://github.com/lgz-0713/ReDDiT 。少步蒸馏是后续候选，但公开实现面向RGB LOL，与本项目Canon RAW→RGB、SIED潜空间不同，不能直接替换权重或把30dB跨数据集照搬；暂不另建大模型环境或下载新大权重。
- PyTorch官方channels_last说明：https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html ，以及AdamW fused选项：https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html 。这里只作为待测机制，不能拿官方其他网络的加速比当本项目实测。

## 收尾

更新同一个`sied-1000`监测，不新建重复任务。源18868校验完成后仍及时关机；新37497须确认固定评价、效率基准及已安排的有界验证/训练作业均结束、最新资料已异机校验后再关机。不能在baseline刚结束但GPU有本轮研究进程时提前关机，也不能以持续研究为名无限空转。没有明确已部署的新作业时不要凭本计划擅自启动长训。定时任务只在实质进展/异常/需用户处理/结束时通知。

## 19:49用户追加：按ICCV方向逐项改进（新增一组受控实验）

用户要求继续改进算法、速度与训练效率。保留原定第一组TLC/3DLUT实验不变，新增第二组，不重跑固定1000基线或76test。向用户说明预计在此前收尾估计上增加约30–60分钟，具体看验证耗时；不是承诺分数提高。

### 阅读、代码核对与选取理由

1. **SVDLUT，ICCV2025**：[作者代码](https://github.com/WontaeaeKim/SVDLUT)，核对HEAD `4759030cb6a51641a97f4031f56bc21e032e5cea`。已阅读README、完整`models.py`与`model_losses.py`。作者将3D变换分解成低维LUT，用低秩因子生成表，并包含空间/双边分支与自定义CUDA核。它的FiveK/PPR10K训练和耗时不是本SIED的实测成绩；本轮不安装其CUDA扩展、不加载跨数据集预训练权重。
2. **GT-Mean Loss，ICCV2025**：[作者代码](https://github.com/jingxiLiao/GT-mean-loss)，核对HEAD `1a09e2e077613c195ae7890005ffdfc8ce01dd4d`。已检查`GTmeanLoss`：以灰度均值的对称KL构造detached混合权重，在训练损失内融合原预测与GT均值归一化预测的L1。这**不是**简单加一个亮度均值惩罚，更不授权推理时读取测试GT调亮度。本轮先保留为后续独立损失消融候选，不与架构对照同时更改，避免把目标变化误归因于低秩表示。
3. **Retinexformer，ICCV2023**：[作者代码](https://github.com/caiyuanhao1998/Retinexformer)。阅读官方使用说明及架构；官方也明确警示使用测试GT均值调整评价的设置。当前任务不采用这种非盲亮度校正；不因为其他数据集的30dB而替换本项目Canon RAW权重。

### 第二组算法：只改色彩表的参数化

`src/dip_course/enhancement/factorized_lut.py`提供独立`PairwiseLowRankLUT`：将RGB残差表示为RG/RB/GB三个二维表的均值，各二维表在tanh限幅前采用rank4矩阵因子；仍为17格点、3基底、最大RGB残差0.12。复用相同的全场景area512→area128控制器，将表重建成小型3D LUT后调用同一个原生三线性查询，因此训练/推理上下文、原生分块协议一致。

这是**SVDLUT机制启发的低秩消融，不是完整论文复现**：没有空间/双边特征、没有论文自定义CUDA核；同色不同位置仍不能区别处理。低秩因子经tanh后最终矩阵不必保持严格rank4。参数预期从63,324降为22,779（约减少64%），但重建后仍查3D表，不据此宣称原生查询更快。训练吞吐、收敛与指标均需实测。

### 第二组效率：真实训练数据显存常驻

现有832张FP32训练对约2.74GiB，一次读取校验后搬入GPU。原像素、图序与dtype不变；不包含val/test，不重新缩放、不转FP16。检查显存余量（缓存外至少6GiB），不足则明确失败，不能偷偷换精度。记录一次性加载时间与字节量。

先进行4步预热+16步计时的真实train批次短对照：同一3DLUT、同一初始化/批次/loss/FP32/fusedAdamW，分别使用双worker pinned DataLoader与显存常驻；检查首个预测和20步更新后的参数偏差。另测低秩表示同等数据路径的步时。报告预加载成本、短测范围和测量边界，不将此短测称为包含全部验证的端到端加速，也不把它当质量收益。

### 受控训练与评价预算

- `cloud/research_sied_iccv.py`分别从恒等映射训练global3d与pairwise_rank4；相同控制器初值、固定私有batch RNG、FP32、fusedAdamW、lr3e-4、cosine、逐图log-MSE、同一重建3D表上的平滑/单调正则。只改变表参数化；两者均使用显存常驻数据。另有计算差异不能全部归为低秩本身。
- 每组最多1000步；每组训练含周期原生验证15分钟预算，在逐图边界检查（最多多出一张图的计算时间）。250步一次12图原生验证，750步后连续3次无合格收益早停。最多两组，不追加第三组，不扩容/新开机。
- pilot仍是固定seed20261024的相同12张val，原生PSNR至少+0.02dB、SSIM退化≤0.002、全部颜色误差和LPIPS增幅≤2%才保存候选；通过后再做完整92val并再次过门槛。两组都不合格则记录负结果，不替换旧模型。
- 复用第一组生成的原生pilot浮点缓存，并逐图核对冻结PSNR偏差≤0.001dB；记录缓存SHA。其余val从对应有SHA的基础FP32缓存和冻结NAF重放。不能用8-bit展示PNG、512缩略图或GT亮度归一化替代原生盲预测评分。
- `run_sied_iccv_after_research.sh`等待第一组`result.json`再取共享GPU锁；第一组消失且无结果则失败退出，不擅自重启。screen为`sied_iccv_research`，输出`research/iccv_pairwise/`，日志`logs/iccv_research.log`。首次启动即落盘实验声明/代码hash/训练key，存在声明则禁止自动覆盖。

当前仅代码/测试与排程阶段，**第二组尚无真实速度或质量成绩**。所有训练和92验证仍受官方预训练见过这些图的局限约束；最终验收76test不在本轮重新评价。实验结束后对两组正负结果分别写报告，备份新权重/PNG/指标/日志并核验，再关闭37497并停止监测。

20:02已部署并确认`sied_iccv_research`等待进程存在（尚未训练）。新增7项测试及既有回归共**176项通过，0失败/0错误/0跳过**，13.603秒；[新JUnit证据](assets/enhancement/efficiency_20261024/iccv_regression_full.xml)。18项针对性测试和ruff、脚本语法检查也通过。实查参数确为63,324与22,779；冻结评价登记的所有源码SHA均未变化。监测`sied-1000`已更新为检查两组研究后再备份关机，不新建重复监测。全量评价此时完成301/924新推理，继续正常运行，没有新质量均值。

## 21:54开始第一组研究：TLC负结果，LUT继续

固定1000张基线于21:53:59结束，总体PSNR26.554490dB（含训练，非独立测试）；924张新图已在本机备份并逐图SHA核验，76测试保持原始字段不变，详见全量报告。第一组`research_sied_color.py`已取得GPU锁，第二组成对实验继续等待，不提前关机。

同一固定12val、原生浮点对照：未转换模型PSNR为26.952233574933743dB、SSIM0.8674695044755936；TLC512为26.853514708264473dB、SSIM0.8671243141094843。PSNR约下降0.098719dB，未过收益门槛，**不扩大成92图TLC验证、不替换基线**。不是论文方向普遍无效，只说明这一冻结模型/窗口/协议下本次pilot为负结果。全部颜色和LPIPS逐图记录保存在`outputs/sied1000_20261024/research/color_lut_tlc/`，与后续LUT结果分开。

色调LUT训练按原1500步/45分钟上限继续，尚无最终92val候选成绩；ICCV低秩成对实验也尚无速度/质量结论。实时备份研究资料时只排除可再生成的`*.npy`缓存，**不能排除整个pilot_baseline目录**，该目录还包含必须保留的JSON指标。

22:03检查：第一组LUT已按上限完成1500步，开始完整92val原生验收。预设“新候选相较已保存模型至少再增0.01dB”的更新门槛选中1250步，pilot PSNR27.36835792669604dB，相对同12图基线约+0.416124dB；1500步pilot为27.372052397397578dB，但不足再增0.01dB，未替换已选检查点。不能因此把1500步分数配给1250步权重，亦不能当作完整92图或独立test收益。已选`best_lut.pt`的SHA为`2cfd72ec81addd4a091ea199c613c4bc47e27e562a098971db7302510a1db609`，已在Mac备份并与云端核对。第二组仍等待，尚无最终速度/质量结论。

## 22:17第一组完成：92val通过，第二组已启动

第一组于22:17:44正常结束。1250步候选在完整92张原生val上通过预先约定门槛；本机独立复算8项逐图均值、核对92个key及顺序、检查点SHA，并对排除可再生NPY的整个第一组目录做rsync checksum dry-run，零差异。新模型与92张展示图已异机备份。原始76test与1000固定基线不变，**尚未对新模型进行独立test验收，也不覆盖原final**。

| 指标 | 冻结基线92val | LUT1250步92val | 变化 |
|---|---:|---:|---:|
| PSNR↑ / dB | 26.636931 | 26.842980 | +0.206049 |
| SSIM↑ | 0.85627188 | 0.85626056 | -0.00001132 |
| DeltaE2000↓ | 4.496330 | 4.473077 | -0.023252 |
| 逐图DeltaE P95均值↓ | 7.831748 | 7.851426 | +0.019678 |
| 亮度MAE↓ | 4.455134 | 4.419811 | -0.035323 |
| 饱和度MAE↓ | 0.03374167 | 0.03310769 | -0.00063398 |
| 色相误差↓ / 度 | 11.346907 | 11.199809 | -0.147098 |
| LPIPS↓ | 0.31336739 | 0.31313677 | -0.00023061 |

不能称“所有指标改善”：SSIM略降，色差P95约增加0.2513%，都在预设容忍范围内。逐图PSNR为**50图提高、42图下降**，中位变化+0.318271dB，最差-1.104357dB、最好+2.114446dB；所有图均计入，未删困难图。平均收益只是小幅进展，不是30dB目标已经实现。验证图仍有官方基础模型预训练暴露。

证据：[第一组结果](assets/enhancement/research_20261024/color_lut_tlc/result.json)、[92图逐图指标](assets/enhancement/research_20261024/color_lut_tlc/lut_full_val/metrics.json)、[92图summary](assets/enhancement/research_20261024/color_lut_tlc/lut_full_val/summary.json)、[本机复核](assets/enhancement/research_20261024/color_lut_tlc/local_verification.json)。模型在本机`outputs/sied1000_20261024/research/color_lut_tlc/best_lut.pt`，SHA见上。轻量指标已归档，展示PNG与模型不重复夹入报告目录。

### 已完成的真实训练批次短测

第二组已开始并通过数据路径等价检查。FP32、batch16、同批次/控制器初值/loss/fusedAdamW，4步预热+16步计时：

| 模式 | 数据+更新平均秒/步 | 参数量 |
|---|---:|---:|
| global3D / pinned DataLoader | 0.03469200 | 63,324 |
| global3D / 显存常驻 | 0.01299559 | 63,324 |
| pairwise rank4 / 显存常驻 | 0.01302303 | 22,779 |

同一个global3D模型的短测吞吐比为**2.669521×**；首个预测完全一致，20步后最大参数偏差`1.0766088962554932e-6`，低于预设`1e-5`。常驻数据2,944,401,408字节，预加载2.448222秒，需单独计入总成本。原始证据：[real_data_benchmark.json](assets/enhancement/research_20261024/iccv_pairwise/real_data_benchmark.json)。

这是**真实训练批次的短测**，不包含原生验证和预加载摊销，不等于整次实验快2.67倍。低秩模型的本次步时与global3D基本相同，甚至略慢约0.2%，不能宣称低秩架构本身加速；其参数减少64%的质量/正则化效果仍等待完整对照。先前NAF合成1.168×与此LUT短测属于不同模型/范围，不能相乘或混报。

22:23第二组global3d已完成1000步，按同一更新门槛选中750步（1000步pilot虽略高但增益不足0.01dB），正在完整92val验证。之后仍需pairwise_rank4的既定有界训练/验证，因此尚不关机，不追加第三组。
