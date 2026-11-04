# B2A_131_02 PKL 闭环跟踪实验

最新定位见 [起步失稳根因检查](g1_tracking_root_cause.md)：确认支撑 wrench 几何失配，
以及实测关节状态缺少边界保护。脚底几何对照消除了早期大幅振荡，但整段跟踪仍未通过。

使用 `retarget/output/B2A_131_02_stageii.pkl`：1247 帧、120 Hz、约 10.38 秒，含接触序列。
运行 `dynamics-aware` profile，RTX 4090 D / Isaac Sim，50 Hz 控制、500 Hz 物理和显式 PD。
保留自碰撞、原始参考和原有验收阈值。

## 结果

| 实验 | 结果 | 主要指标 |
|---|---|---|
| 初始姿态保持 2 秒 | 通过现有静态参考验收 | 最终 CoM 误差 9.44 mm；servo RMSE 0.0226 rad；相对 PKL 的关节 RMSE 0.0444 rad |
| 1 秒预热 + 3 秒动作播放 | 未完成；1.38 秒停止 | 已播放约 0.38 秒；最后成功记录为参考帧 43.2；下一次求解失败 |

动作实验在成功记录的区间内，CoM 最大误差为 0.0551 m，相对 PKL 的关节 RMSE 峰值为
0.473 rad。失败求解时模式为双支撑，但脚部测得法向力分别为 0 N 和 2337 N；
接触加速度残差为 0.690，摩擦不等式违反量为 2.714。控制器保存失败快照后停止。

这里的 servo RMSE 指实际关节相对积分命令的误差，与实际关节相对 PKL 参考的误差不同。
静态测试通过不代表整段 PKL 跟踪成功。接触模式与实测载荷的不一致是失败时的观测，
尚不能据此把所有动态误差都归因于状态切换；动作开始后、首次单支撑切换前，跟踪误差已增大。

![保持与动作跟踪对比](../acc_data/dynamics_tracking_13102/tracking_comparison.png)

## 本次排查并修复的模型问题

1. `urdfpy` 为没有 `<inertial>` 的 link 自动补入单位质量与惯量。
   四个 IMU/相机坐标系因此虚增 4 kg。导入器现依据原始 XML 判断是否存在惯性定义，
   无定义的坐标系保持零质量/零惯量。实际模型总质量从 37.341 kg 修正为 33.341 kg。
2. PhysX 的广义根部速度、质量矩阵以及 link Jacobian 使用刚体质心；Fabric 使用 link 原点。
   新增 `rigid_body_frames.py`，同时转换根部速度、加速度、质量矩阵、偏置力和足部 Jacobian。
   根部加速度转换包含 `omega × (omega × r)`，并非只变换质量矩阵。

转换定义：`v_COM = T v_origin`，`a_COM = T a_origin + beta`，因此
`M_origin = Tᵀ M_COM T`，`h_origin = Tᵀ(h_COM + M_COM beta)`。
足部 Jacobian 同时变换输入根坐标与输出作用点，使接触 wrench 仍关于 ankle link 原点。

对真实 Isaac 初始状态核验后，归一化质量矩阵最大误差为 `1.46e-6`，
左右足部 Jacobian 最大误差分别为 `5.36e-7` 和 `3.73e-7`。
没有放宽模型检查阈值。55 项测试全部通过；四种接触模式的离线 QP 验证已用修正质量重跑。

## 复现

在能访问 GPU 的 `humanoid_fabric` 环境中，从项目根目录运行：

```bash
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --motion-pkl retarget/output/B2A_131_02_stageii.pkl --duration 4 --motion-warmup-duration 1 --diagnose-com --output-path /tmp/g1_tracking.npz
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --motion-pkl retarget/output/B2A_131_02_stageii.pkl --duration 2 --motion-warmup-duration 5 --diagnose-com --output-path /tmp/g1_hold.npz
```

## 数据

- [汇总 JSON](../acc_data/dynamics_tracking_13102/summary.json)：包括 PKL SHA-256、实验参数及数值结果。
- [动作跟踪日志](../acc_data/dynamics_tracking_13102/tracking.log) / [失败快照](../acc_data/dynamics_tracking_13102/tracking.failure.npz)。
- [保持日志](../acc_data/dynamics_tracking_13102/hold.log) / [保持结果](../acc_data/dynamics_tracking_13102/hold.npz)。
- [模型比对](../acc_data/dynamics_tracking_13102/model_comparison.json) / [原始 Isaac 惯量快照](../acc_data/dynamics_tracking_13102/raw_isaac_inertia.npz)。

## 后续跟踪排查

修复运动起步时 CoM 目标二次差分产生的加速度尖峰：新 profile 使用有界二阶参考更新。
足部接触偏置改为当前状态中心差分 `Jdot*v`，避免历史滤波滞后；增加足部线/角加速度差分测试。
新增的预测加速度限幅和名义目标选项仅用于消融，默认关闭。

以下为顺序修改代码的对照实验，不能只凭 CLI 重现早期版本；每组快照、日志和说明均已保存。
全部动态实验都未完成，没有放宽验收阈值或关闭自碰撞。

| 设置 | 停止时仿真时间 / s | 推进的 PKL 时间 / s | 结果 |
|---|---:|---:|---|
| 二阶 CoM 参考 | 1.78 | 0.780 | QP 验证失败 |
| 再加承重门控 | 2.26 | 0.627 | QP 验证失败 |
| 再限预测加速度 | 1.94 | 0.267 | QP 验证失败 |
| 接触姿态恢复 kp=20、kd=8 | 0.80 | 0 | 预热期间失败 |
| 四分之一播放速度 | 2.72 | 0.270 | QP 验证失败 |
| 承重门控 + 当前接触偏置 | 2.24 | 0.647 | 命令边界冲突 |
| 再改名义加速度目标 | 2.24 | 0.627 | 接触/力矩约束超限 |
| 当前接触偏置 + 更快命令同步 | 2.16 | 0.647 | 接触/力矩约束超限 |

“推进的 PKL 时间”由失败快照中的参考帧位置除以 120 Hz 得到；预热和等待承重不算动作进度。
当前接触偏置加承重门控实验，在成功记录的历史中 CoM 峰值误差为 0.0268 m，
相对 PKL 的关节 RMSE 峰值为 0.3225 rad。更长的仿真时间并不等于完成更多动作。

独立 OSQP 重放当前接触偏置实验返回 `primal infeasible inaccurate`（20000 次迭代）。
仅在离线诊断中移除命令加速度边界后可以求解：右踝 roll 所需命令加速度为
`+5.399 rad/s²`，而积分命令状态要求其处于 `[-150, -48.746] rad/s²`。
该结果支持命令积分边界与物理约束冲突的判断，但尚不能证明其是全部失稳的唯一原因。
更快同步也未消除失败，不能把取消边界作为修复。

当前默认修改后的静态参考回归通过：2 秒末 CoM 误差 9.96 mm，相对 PKL 初始姿态的
关节 RMSE 为 0.0477 rad；57 项测试通过。静态回归没有播放动作。

当前版本复现：

```bash
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --motion-pkl retarget/output/B2A_131_02_stageii.pkl --duration 6 --motion-warmup-duration 1 --require-support-load --diagnose-com --output-path /tmp/g1_tracking_current_bias.npz
PYTHONPATH=src python examples/compare_g1_qp.py /tmp/g1_tracking_current_bias.failure.npz --diagnose-constraints
```

[后续实验汇总与参数](../acc_data/dynamics_tracking_13102_followup/summary.json)、
[独立 QP 重放日志](../acc_data/dynamics_tracking_13102_followup/current_bias_replay.log)、
[测试日志](../acc_data/dynamics_tracking_13102_followup/tests.log)。同目录保存全部失败快照与静态回归结果。

尚待解决：在保留接触、摩擦、力矩和关节限制的前提下，协调命令积分状态与可实现加速度，
并验证运动参考的支撑阶段是否动态可行。整段 PKL 跟踪仍未通过。
