# 4090 云机约定

本目录只保存云端执行说明，不保存密码、SSH 私钥、数据或权重。

## 隔离原则

- 工作目录：`/root/autodl-tmp/cv-homework`
- 虚拟环境：`/root/autodl-tmp/venvs/cv-homework`
- 不修改 Gala 项目的 ComfyUI、LoRA、模型和虚拟环境。
- 正式运行前记录 GPU、磁盘、Python、PyTorch、CUDA 和 Git 版本。
- 正式运行包和配置生成 SHA256 清单，上传后再次校验。
- 先做100步冒烟测试，再做5000步试验，最后才允许完整训练。
- 每次训练保存 `run_environment.json`、日志和可恢复 checkpoint。
- 下载并校验结果后，确认没有训练进程，再关闭实例。

## 资源门槛

- RTX 4090 24GB；
- 数据盘至少剩余100GB，完整数据规模确认后重新计算；
- CUDA 可用且显存启动前基本空闲；
- 数据清单审计通过，训练、验证、测试之间没有重复路径。
