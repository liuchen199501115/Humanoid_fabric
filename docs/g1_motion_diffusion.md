# G1：状态与方向条件的 motion diffusion

> 新版 MDM 几何损失与重叠前缀实现、训练命令见 [motion_generation/README.md](../motion_generation/README.md)。以下保留旧基线配置与实验记录。

本模块把多个重定向 PKL 作为同一训练集，学习 `p(未来动作 | 过去两帧, 目标方向)`。
输入数据为 29-DOF G1，不依赖 Isaac Sim 或 Warp。需要项目环境中的 PyTorch、NumPy。

## 范围

- 50 FPS，2 帧历史预测 25 帧未来。
- 每帧 38 维：root position 3、root rotation 6D 6、29 个关节角。
- MDM 风格时序 Transformer，预测干净样本 x0；cosine diffusion schedule，DDIM 采样。
- 条件为历史状态和水平目标单位方向。没有地形条件、文本条件、速度指令或动作类别。
- 损失为归一化特征重建和帧间增量（包括历史/未来接缝）。未实现 FK、接触或地形损失。
- 这是论文生成器思路的简化实现，不是原论文复现；原论文的 23 关节、link 特征、RL 数据修正和闭环 RL 微调均不同。

## 1. 数据集

在仓库根目录运行：

```bash
conda activate humanoid_fabric
python -m motion_generation.prepare \
  --input retarget/output \
  --output acc_data/motion_generation/dataset.npz
```

只对可信来源的 PKL 使用此命令，pickle 加载会执行其中的 Python 内容。
递归读取输入目录，按 URDF 关节名重排，重采样至 50 FPS；root 四元数采用最短弧 SLERP。
重复动作和不足 27 帧的动作排除，原因写入同名 JSON。
按完整动作随机划分 90% 训练、10% 验证，随后在各自的片段内部取窗口，窗口不跨片段。
精确相同的重采样数据去重；同一次采集的不同裁剪仍可能相关，正式实验应额外按采集来源划分。

NPZ 中的 `states` 是存储上连接的大数组；`offsets` 保留动作边界，`split` 标记划分，
`paths` 保留来源，另存 fps、关节名和关节限位。不能把整个 states 当作一段连续动作。

窗口统一到最后历史帧的水平位置和 yaw 坐标系，高度保留。方向由未来终点相对当前的水平位移提取，
不足 2 cm 时回退到未来终点朝向。这个规则不表达停止指令；原地动作与走路仍可能有多种预测。
归一化均值/标准差只用训练窗口估计并随模型保存。

原始 retarget 数据不等于动力学可执行数据。当前数据包含不同类型的人体动作，并未进行行走类别筛选或物理可行性筛选。
如需稳定的 locomotion 模型，建议在独立输入目录中选出走路/转向片段，再准备对应数据集。

## 2. 训练

短训练只验证链路，不能证明生成质量：

```bash
python -m motion_generation.train \
  --dataset acc_data/motion_generation/dataset.npz \
  --output acc_data/motion_generation/smoke \
  --steps 200 --eval-every 50 --batch-size 32 --device cpu
```

正式训练的起始配置（训练步数不代表已收敛）：

```bash
python -m motion_generation.train \
  --dataset acc_data/motion_generation/dataset.npz \
  --output acc_data/motion_generation/run01 \
  --steps 10000 --batch-size 64 --device cuda
```

产物为 `config.json`、`metrics.jsonl`、`last.pt` 和验证损失最低的 `best.pt`。
验证使用固定噪声以方便比较；损失不能代替多步自回归动作和物理跟踪评估。
数据/网络和优化器状态保存在 checkpoint 中；当前 CLI 每次启动为新训练，不自动续训。

## 3. 生成 PKL

```bash
python -m motion_generation.sample \
  --checkpoint acc_data/motion_generation/smoke/best.pt \
  --seed-motion retarget/output/B2A_01_01_stageii.pkl \
  --seconds 5 --heading-deg 0 --sampling-steps 50 \
  --output acc_data/motion_generation/generated.pkl
```

`--heading-deg` 是世界坐标方向：0 为 +X，90 为 +Y，**不是相对初始朝向**。
`--seed-frame` 是重采样后的帧索引，默认使用前两帧。输出含两帧种子，总帧数约为 seconds×fps。
每次预测 25 帧、默认提交前 12 帧，然后用已提交的最后两帧重新生成。只追加提交的前缀，不重复保存重叠预测。
每一轮都重新变换方向到历史局部系，并将预测恢复到世界坐标。

输出与 `examples/test_g1.py` 兼容：fps、root_pos、xyzw root_rot、dof_pos、dof_names。
root 旋转会正交化并转回单位四元数；生成关节角按 URDF 限位裁剪，统计裁剪比例。
同名 JSON 记录速度极值、root 高度范围、checkpoint 步数，并明确标记尚未物理验证。
速度与接缝没有硬约束，不保证不脚滑、不穿地或不倒地；不要把短训练输出当作可执行动作。
输出不复制种子文件中的 contact_sequence，因为生成动作的接触时序已经改变。

先检查动作质量，再使用已有入口进行仿真验证：

```bash
PYTHONPATH=src python examples/test_g1.py \
  --task-profile dynamics-aware \
  --motion-pkl acc_data/motion_generation/generated.pkl \
  --diagnose-com
```

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_motion_generation.py -v
```

覆盖四元数符号/旋转 180°、SLERP 时间和关节顺序、局部/世界坐标往返、片段边界与数据划分、
模型反传以及改变方向条件确实改变采样输出。后者只验证条件通路，不证明语义方向跟随已学会。

## 留出动作的生成评估

```bash
python -m motion_generation.evaluate \
  --dataset acc_data/motion_generation/dataset.npz \
  --checkpoint acc_data/motion_generation/run01/best.pt \
  --output acc_data/motion_generation/run01/evaluation.json
```

固定随机种子，从验证集抽取最多 128 个窗口，实际从噪声采样未来动作。
报告关节/root 误差、运动方向余弦、接缝速度与关节越界比例，同时给出“保持最后一帧”基线。
方向余弦仅统计预测水平位移超过 2 cm 的窗口，需与移动预测比例一起看。
这里每个窗口只采一个未来，无法完整评估多模态生成；也不代替仿真跟踪。

## 本次基线训练结果

实际使用 `retarget/output` 的 1,983 个文件，接受 1,960 段，排除 23 段（短片段或精确重复）；
50 FPS 共 1,623,118 帧，约 9.02 小时。训练/验证为 1,764/196 段，窗口为 285,999/29,143 个。

`acc_data/motion_generation/run01` 使用 CPU、batch 64、默认 0.84M 参数模型训练 10,000 步，
约 421 秒；最佳验证去噪损失为 0.06159，位于第 10,000 步。

128 个留出窗口、每个窗口 50 步 DDIM 采样的结果：

| 指标 | 结果 |
|---|---:|
| 生成未来的归一化 MSE | 0.4484 |
| 保持最后一帧的归一化 MSE | 0.3871 |
| 关节 RMSE | 0.1871 rad |
| root 位置逐坐标 RMSE | 0.0642 m |
| 移动预测比例 | 89.1% |
| 移动预测的平均方向余弦 | 0.8921 |
| 平均接缝关节速度绝对值 | 1.1682 rad/s |
| 未裁剪预测关节越界比例 | 0.435% |

生成误差仍高于保持姿态基线，不能声称已得到高质量动作模型。需要进一步筛选 locomotion 数据、
改进训练与接触/运动学约束，并验证闭环跟踪。训练去噪损失与从纯噪声采样的 MSE 是不同指标。

`acc_data/motion_generation/generated.pkl` 为 5 秒、250 帧示例，种子为 `B2A_35_01_stageii.pkl`，
世界方向 +X。其最大 root 速度约 2.52 m/s，最大关节速度约 7.65 rad/s；
约 0.862% 生成关节值被限位裁剪。已通过现有 `load_motion_trajectory()` 读取检查，尚未进行物理跟踪验收。

使用本次实际训练权重重新生成：

```bash
python -m motion_generation.sample \
  --checkpoint acc_data/motion_generation/run01/best.pt \
  --seed-motion retarget/output/B2A_35_01_stageii.pkl \
  --seconds 5 --heading-deg 0 \
  --output acc_data/motion_generation/generated.pkl
```

## 100 epoch / 256 hidden / AMP 配置

```bash
python -m motion_generation.train \
  --dataset acc_data/motion_generation/dataset.npz \
  --output acc_data/motion_generation/run_gpu_100epochs \
  --epochs 100 --checkpoint-every-epochs 10 \
  --batch-size 256 --horizon 25 --stride 5 \
  --layers 6 --width 256 --heads 4 --ffn-dim 1024 \
  --lr 1e-4 --weight-decay 1e-4 \
  --device cuda --amp bf16
```

该网络约 4.90M 参数；输入 38 维，历史固定为 2 帧。帧率来自数据集（当前 50 FPS）。
AdamW 优化器；BF16 使用 autocast，FP16（`--amp fp16`）另外启用 GradScaler，损失计算保持 FP32。
RTX 4090 D 上两种精度均已通过 batch 256 的前向、反向和验证短测。

epoch 是完整遍历训练窗口，末批不丢弃：ceil(285999/256)=1118 步/epoch，100 epoch=111800 步。
`--epochs` 和 `--steps` 互斥。epoch 模式每轮验证并更新 `last.pt`、`best.pt`；
每 10 轮额外保留 `epoch_0010.pt`、`epoch_0020.pt` 等独立快照。验证仍抽取固定的最多 512 个留出窗口。
使用新网络从头训练；CLI 暂不支持续训。旧 checkpoint 的网络配置仍能加载。

## 崩溃后的续训

训练入口现支持 `--resume /path/to/last.pt`。当前仅支持完整 epoch 边界恢复；
网络、diffusion、batch size、stride、seed、统计窗口数和数据集应与原训练一致。
恢复模型、AdamW 状态及归一化统计量；FP16 模式同时恢复可用的 GradScaler 状态。
`--epochs 100` 表示累计训练到 100 epoch，不是再训练 100 epoch。
旧 checkpoint 未保存随机数和数据迭代器状态，因此恢复不会逐位重现崩溃前的数据/噪声顺序。
保存采用临时文件替换，减少保存中断造成已有 checkpoint 损坏的风险。
默认启用 Python faulthandler；若再出现 SIGILL，保留完整终端日志中的线程堆栈。
这些改动提供恢复与诊断能力，不代表已修复 SIGILL 的底层原因。
