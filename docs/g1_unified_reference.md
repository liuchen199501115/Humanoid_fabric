# G1 统一全身参考

`examples/test_g1.py` 对 PKL 跟踪默认使用 `--reference-mode unified`，
以 `pchip-analytic` 插值取得同一源帧上的全身姿态、广义速度和加速度。
动作时钟的速度与加速度共同进入插值导数；暂停时所有运动参考一起暂停。
PCHIP 是 C1 插值，加速度在节点处仍可能跳变。

## 单一数据来源

`src/fabrics_sim/reference/whole_body.py` 的 `WholeBodyReference` 持有独立的
`q[B,36]`、`v[B,35]`、`a[B,35]`。`apply()` 一次性设置：

- 29 维关节位置、速度、加速度；
- 基座和躯干姿态及其运动导数；
- 双脚、双手的位置、姿态、速度及加速度；
- 质心位置、速度 `Jv`、加速度 `Ja + Jdot*v`；
- 全身角动量与变化率。

统一路径不再独立覆盖屈膝、摆脚位置、质心、手部或骨盆目标。
需要修改动作时，应先生成一套修正后的全身 q/v/a，再调用 `apply()`。
实测接触锚点、摩擦和执行器约束继续描述物理执行条件，不属于运动参考。
点/线接触释放的旋转任务也从同一足端目标提取，不另造压平或抬脚跟参考。

## 运行

```bash
cd /home/sky/Humanoid_fabric
bash examples/run_lafan_tracking.sh --show-isaac-window
```

保留旧实验作为对照：

```bash
bash examples/run_lafan_tracking_legacy.sh --show-isaac-window
```

统一模式拒绝 `--dynamic-reference`、`--source-swing-reference`、
`--align-balance-pelvis` 等独立参考生成选项，防止悄悄覆盖全身参考。
原始关节姿态不会因启用 ReferenceManager 而自动增加屈膝。
现有控制权重、接触门控仍需进一步物理验证，不能将目标一致等同于能稳定走完。

## 验证和记录

相关单元回归 81 项通过，覆盖运动参考零反馈误差、停止后导数清零、
时钟重定时、输入隔离、点/线接触自由坐标及旧接触/管理器逻辑。

日志新增 `history_whole_body_reference_q/v/a` 和 `history_reference_mode`。
这些记录与已有的实际使用关节、足端和质心目标可用于逐控制步核对。
短时仿真保留全段验收：指定短 `--duration` 时，未覆盖完整源帧会报告失败，
不能把这种运行称为完整跟踪通过。

两秒 Isaac Sim 验证（warmup=0.2 s）完成 1000 次 QP/扭矩更新，源帧推进
至 50.25，900 个控制步具有非零参考速度。逐步复核关节、双脚、骨盆、
质心位置及质心速度/加速度，与保存的全身 q/v/a 重算结果最大误差均为 0。
该片段仍为双支撑，未验证换脚。峰值关节 RMSE 0.129 rad、根位置误差
0.00954 m、根姿态误差 0.150 rad，片段精度通过，完整覆盖不通过。
QP/扭矩按仿真 2 ms 更新；QP P99 约 3.20 ms，尚不满足墙钟实时 500 Hz。

- [参考一致性审计](../acc_data/lafan_unified_reference_motion_20260928.reference_consistency.json)
- [片段跟踪审计](../acc_data/lafan_unified_reference_motion_20260928.tracking.json)

## 首次承重转移与抬脚诊断

新增独立入口 `bash examples/run_lafan_transfer.sh --show-isaac-window`，
从首次起步前的停止状态生成协调的全身转移与抬脚参考。
范围、验收和结果见 [全身承重转移验证](g1_unified_transfer.md)。
该诊断不接管完整 PKL 跟踪入口。
