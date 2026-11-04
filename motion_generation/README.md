> **新版：论文 99 维、无地形条件**见 [paper/README.md](paper/README.md)。使用 `--profile paper` 和 `dataset_paper99.npz`；下文旧 38 维模型仍保留。

# G1 Motion generation

从项目根目录运行。新实现采用 MDM 风格 x0 去噪、G1 可微分 FK 几何损失和重叠前缀 inpainting。

## 新版训练

接触数据已整理为 `acc_data/motion_generation/dataset_contacts.npz`。如需重建：

```bash
python -m motion_generation.prepare --input retarget/output \
  --output acc_data/motion_generation/dataset_contacts.npz --with-contacts
```

```bash
python -m motion_generation.train --profile mdm \
  --dataset acc_data/motion_generation/dataset_contacts.npz \
  --output acc_data/motion_generation/mdm_100epochs \
  --epochs 100 --checkpoint-every-epochs 10 \
  --batch-size 256 --horizon 25 --stride 5 \
  --layers 6 --width 256 --heads 4 --ffn-dim 1024 \
  --lr 1e-4 --weight-decay 1e-4 --device cuda --amp bf16
```

必须指定 `--profile mdm` 才会启用新版。默认 `legacy` 供旧权重/旧训练命令兼容。
新版有新增的条件编码参数，需从头训练；不能直接用旧 checkpoint 续训新版。

## 连续生成

```bash
python -m motion_generation.sample \
  --checkpoint acc_data/motion_generation/mdm_100epochs/best.pt \
  --seed-motion retarget/output/B2A_35_01_stageii.pkl \
  --seconds 5 --heading-deg 0 --device cuda \
  --commit-frames 12 --overlap-frames 4 \
  --output acc_data/motion_generation/mdm_forward.pkl
```

新版 checkpoint 默认保留 4 帧重叠前缀，旧版默认 0，且旧权重不能开启 inpainting。
已知帧在每步去噪时注入对应噪声水平，输出时精确恢复已知前缀；剩余帧由网络预测。
根位置和朝向在世界/局部坐标间一致转换。PKL 记录 `generation` 中的拼接参数。
这不是对整个生成片段做低通滤波，也不保证新生成后缀一定平滑。

## 本次具体实现

- `model.py`：时序 Transformer、历史和方向条件、新增已知帧/掩码条件。
- `diffusion.py`：cosine schedule、DDIM、噪声一致的已知帧约束。
- `data.py` / `prepare.py`：原始左右脚二值接触标签的最近邻重采样，窗口不跨动作。
- `kinematics.py`：从项目 URDF 构建双腿 FK，输出脚底四个球体中心；与 MuJoCo 对照验证。
- `losses.py`：实际时间尺度的速度匹配、加速度匹配（以 100 为尺度）、未知后缀接缝速度、
  支撑脚水平速度、最低脚底接触高度、所有脚底点穿地、FK 参考跟踪和关节限位软损失。
- `train.py`：随机长度的已知前缀训练，未知部分重建损失，AMP/epoch/checkpoint/恢复支持。
- `sample.py`：缓存未提交的世界坐标预测，下一窗口提取重叠前缀进行约束采样。

几何损失用反归一化后的 FP32 状态和固定标签计算，不允许模型通过预测“无接触”逃避惩罚。
地面假设为 z=0 平地。原始接触标签是上游数据，尚未完成质量筛选，不能当成实测接触力。
四脚底点约束是简化接触模型，不覆盖复杂脚掌滚动和其他身体部位碰撞。
每项权重可通过 `--geometry-velocity/acceleration/boundary/skating/contact/penetration/fk/limits` 设置。
日志的 `geometry_last_validation_batch` 是最后一个验证 batch 的分项值，不是全验证集均值。
训练仍包含旧的归一化帧间增量项，权重是 `--velocity-weight`；它与物理单位速度匹配是不同项。

## 参考来源与边界

以下参考用于理解方法。本项目代码是面向现有 G1 数据的独立实现，没有复制这些仓库的完整代码或权重：

- [MDM 官方代码](https://github.com/GuyTevet/motion-diffusion-model)：x0 motion diffusion、几何损失和 motion inpainting；核对了 `diffusion/gaussian_diffusion.py`。
- [Motion-Diffusion-From-Scratch](https://github.com/AmirMalekhosseini/Motion-Diffusion-From-Scratch)：轻量 Transformer、velocity/foot-contact 损失的参考。
- [Kimodo](https://github.com/nv-tlabs/kimodo)：已有 G1 模型及独立动作表示。本次未移植其表示、文本编码器和预训练权重。
- [CMDM](https://github.com/lycorp-jp/CMDM)：因果自回归生成研究参考。本次使用前缀补全；未实现 MAC-VAE、Causal-DiT 或逐帧 diffusion forcing，不是 CMDM 复现。

## 验证与限制

```bash
python -m unittest discover -s tests -p 'test_motion_g*.py' -v
```

验证覆盖独立 MuJoCo FK 对照、FK 梯度、脚滑/穿地损失响应、无接触时不惩罚脚滑、
前缀精确保留及条件梯度，以及原有旋转/窗口测试。
短训练只验证链路，尚未证明新版在长 rollout 上减少脚滑或跳变。正式训练后应使用同样种子、方向、
时长与旧版比较 `audit` 指标，并另行验证 Fabric/WBC 跟踪。损失和前缀约束均不构成动力学安全保证。

旧基线记录见 [历史文档](../docs/g1_motion_diffusion.md)。

## 本机 SIGILL / SIGBUS 运行时修复（2026-09-16）

原环境为 CPython 3.11.11、PyTorch 2.7.0+cu126。用户日志出现 SIGILL；GDB 对照运行
在累计第 14 epoch 后捕获 SIGBUS，停止在 `_PyEval_EvalFrameDefault`。
模型及 AdamW 状态均有限值。这定位到解释器/原生运行时层，尚不足以证明是某一条 CPython 缺陷，
也没有证据说明 BF16 或开头的 pkg_resources 弃用警告是根因。

替换 Python 及其基础运行库的对照测试，仍使用原环境的 Torch/NumPy 和相同 BF16 模型配置：

- Python 3.11.15 从第 11 epoch 连续训练到第 25 epoch（14 个完整 epoch，605 秒），正常结束。
- 固定安装后，通过原训练入口再完成第 26 epoch（44.6 秒），正常结束。
- 12 项运动生成/运行时测试通过。没有对完整 100 epoch 作无崩溃保证。

原 Isaac/Fabric 环境固定 GCC 12，而新版 Python 需要 GCC >=14，因此没有升级原环境。
已在项目 `.runtime/motion_generation/` 安装验证过的 Python 3.11.15 基础环境。
`runtime.py` 根据本机 `.runtime/training-runtime.json`，仅在原受影响的 Python 可执行文件启动
`python -m motion_generation.train` 时，在加载 Torch 前切换到新解释器。保留 CLI 参数，
通过 PYTHONPATH 使用原环境 CPython 3.11 的 Torch/NumPy 包；其他环境没有本机配置时不会切换。
训练配置的 `runtime` 字段记录实际执行版本。没有修改原环境包，且不再依赖 `/tmp` 环境。

本机运行时目录不纳入 Git；在另一台机器运行时不会自动下载或创建它。
使用环境的原命令即可自动切换；控制台会打印 `[training runtime] Switching ...`。

从已完成的第 26 epoch 继续到 100 epoch：

```bash
python -m motion_generation.train --profile mdm \
  --dataset acc_data/motion_generation/dataset_contacts.npz \
  --output acc_data/motion_generation/mdm_runtime_fixed \
  --resume acc_data/motion_generation/mdm_runtime_fixed/last.pt \
  --epochs 100 --checkpoint-every-epochs 10 \
  --batch-size 256 --horizon 25 --stride 5 \
  --layers 6 --width 256 --heads 4 --ffn-dim 1024 \
  --lr 1e-4 --weight-decay 1e-4 --device cuda --amp bf16
```

诊断证据保存在 `acc_data/motion_generation/sigill_gdb.log`、
`python31115_validation.log` 和 `mdm_runtime_fixed.log`。GDB 在正常退出后查询寄存器会显示
`No registers / No stack`，应以其 `exited normally` 和完整训练日志判断该测试是否成功。
