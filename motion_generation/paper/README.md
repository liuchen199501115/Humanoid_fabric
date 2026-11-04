# 论文结构：99 维、无地形条件

本版本按用户提供的论文段落实作 MDM 风格的 x0 diffusion，并按用户要求去掉地形高度条件。入口仍在顶层 `motion_generation/`，与 `retarget/` 同级。

## 表示与条件

- 50 FPS；前 2 帧作为条件，预测后续 25 帧。
- 每帧 99 维：根位置 3 + 根四元数 xyzw 4 + 关节角 23 + 身体位置 23×3。
- 使用仓库 `g1_23dof.urdf` 的 23 个转动关节。身体位置选各转动关节的子 link 原点；这是本实现明确采用的顺序，论文截图没有给出具体 link 名称。
- 根位置与身体位置转换到最后一帧历史的水平朝向坐标系；保留世界高度。四元数在窗口内保持符号连续。
- 目标 heading 从窗口内根位置位移计算并归一化；静止窗口用最后一帧根朝向。无速度指令。
- 不输入高度图、不添加地形噪声。训练时扰动历史条件，归一化其四元数并重新计算身体位置。
- 每个窗口来自同一个动作文件；保留原有按动作划分的训练/验证集。

## 损失

`L = L_reconstruction + 0.1 L_velocity + 1.0 L_joint_consistency + 10.0 L_flat_penetration`

- reconstruction：标准化特征的 x0 重建 MSE；预测四元数先归一化并与目标符号对齐。
- velocity：标准化特征的帧差误差，包括历史和未来的接缝。四元数帧差是旋转变化的近似，而非精确角速度。
- joint consistency：身体位置输出与预测根姿态、关节角经可微 FK 得到的位置之间的误差，两侧均有梯度。
- flat penetration：脚底球底部和 FK link 原点低于 `z=0` 的软惩罚。它不是完整碰撞检测。

权重分别由 `--velocity-weight`、`--geometry-fk`、`--geometry-penetration` 控制。其他 `--geometry-*` 参数只作用于旧 `mdm` profile。以上权重、link 选择和数值细节属于实现选择，不能称为作者代码的逐项复现。

此 profile 不额外使用旧 `mdm` 的接触标签、滑脚损失或前缀 inpainting。软损失不能保证完全不滑脚、不跳变或动态稳定；长时间 rollout 与 tracking 仍需要验证。

## 数据转换

已生成 `acc_data/motion_generation/dataset_paper99.npz`：1,960 段，1,623,118 帧，约 9.017 小时。需要重建时：

```bash
python -m motion_generation.paper.prepare \
  --input acc_data/motion_generation/dataset.npz \
  --output acc_data/motion_generation/dataset_paper99.npz
```

原 29 DOF 数据中未使用的腰 roll/pitch、双腕 pitch/yaw 共 6 个角不进入 23 DOF 特征。身体位置按 23 DOF URDF 重新计算，因此不能保留这些额外关节造成的姿态变化。

## GPU 训练（从头训练）

在项目根目录、`humanoid_fabric` 环境运行：

```bash
python -m motion_generation.train \
  --profile paper \
  --dataset acc_data/motion_generation/dataset_paper99.npz \
  --output acc_data/motion_generation/paper99_100epochs \
  --epochs 100 --checkpoint-every-epochs 10 \
  --batch-size 256 --horizon 25 --stride 5 \
  --width 256 --layers 6 --heads 4 --ffn-dim 1024 \
  --diffusion-steps 1000 --lr 1e-4 --weight-decay 1e-4 \
  --device cuda --amp bf16
```

当前机器训练入口会自动切换到已配置的 Python 3.11.15 runtime。旧 38 维 checkpoint 与新模型不兼容。

输出目录中：`best.pt` 为验证损失最低模型，`last.pt` 为最近一次验证保存的模型，`epoch_0010.pt` 等为定期保存。100 epochs 共 111,800 steps。没有因实现本功能而启动正式 100 epochs 训练。

## 生成向前 PKL

训练完后，替换种子动作路径：

```bash
python -m motion_generation.paper.sample \
  --checkpoint acc_data/motion_generation/paper99_100epochs/best.pt \
  --seed-motion retarget/output/YOUR_MOTION.pkl \
  --output acc_data/motion_generation/forward_paper99.pkl \
  --seconds 5 --heading-deg 0 --commit-frames 12 \
  --sampling-steps 50 --device cuda
```

`0` 表示世界 +X 方向。每次预测 25 帧、采用前 12 帧，再以最近 2 帧继续预测。导出兼容现有播放代码的 29 DOF PKL，缺少的 6 个关节固定为零。23 DOF 与 29 DOF 模型几何存在差异，导出兼容不等于原 29 DOF 姿态完全重建。导出前归一化根四元数、裁剪关节限位，并通过 FK 修正下一窗口的身体位置条件；JSON 保留修正前位置输出与 FK 的误差。

评估可继续使用 `python -m motion_generation.evaluate`，同时传入新的 dataset/checkpoint；报告包含身体位置输出的 FK 一致性误差。

## 验证范围

- `tests/test_motion_paper.py`：窗口边界、坐标往返、与独立 MuJoCo 23 DOF 模型对照 FK、损失梯度、反向传播。
- 全部 motion 测试 16 项。
- CUDA/BF16、batch 256、6 层×256 的 20 步短训练通过；只用于检查流程，不代表收敛或长训练稳定性。
- 短训练模型导出 50 帧 PKL，通过现有 `load_clip` 读取检查。此文件不是可用步态质量的验证结果。

## 2026-09-16 原生段错误的处理

一次正式训练在完成第 42 个 epoch 后，Python 栈停在逐窗口 `quat_matrix` 调用并发生 SIGSEGV。该次已使用 Python 3.11.15，因此更换解释器不足以证明问题解决。现场没有可用的 core 文件，尚不能确认底层根因；Python 栈中的行号也不等于非法内存访问的来源。

本次将 DataLoader 的批量获取改为 `PaperWindows.__getitems__`：用 Torch 批量完成坐标变换、四元数符号连续化和 heading 计算，避开每个训练窗口重复调用 NumPy 旋转函数的路径。模型、损失和 checkpoint 兼容性不变。新增与原 NumPy 实现逐项对照及 Subset/DataLoader 集成测试，全部 motion 测试共 17 项通过。全量数据检查未发现 NaN/Inf、异常四元数范数或错误动作边界。

从原第 42 个 epoch checkpoint 的验证续训写入独立目录 `acc_data/motion_generation/paper99_batch_validation/`。这是绕开故障路径的缓解措施；短时间续训成功不能证明底层原生崩溃已彻底消除。续训时需保持 batch size、stride、seed、stats-windows 和网络配置与原运行一致。

验证结果：CUDA/BF16 连续完成第 43–45 个 epoch（3,354 steps，68.66 秒），正常退出并保存 `paper99_batch_validation/last.pt`；验证损失在 0.0237–0.0250。该目录的 `best.pt` 保存这次续训中改善后的最佳模型。
