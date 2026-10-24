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
