# PKL 起步失稳定位（2026-09-15）

后续修复：脚底几何已默认开启；已增加实测关节制动约束和意外承重落足处理。
见文末结果，整段跟踪仍未通过。下文的“尚未实现”描述的是本报告定位时的代码状态。

本次定位出两处控制模型缺口。此前观察到的“命令积分边界冲突”是最后停止的表现，
不能单独解释最早的动态失稳。输入仍为 `B2A_131_02_stageii.pkl`，自碰撞与验收阈值不变。

## 1. 支撑 wrench 范围超出实际脚底，且缺少作用点平移

`build_single_support_wrench_constraints()` 原来使用半宽 0.04 m、后跟长度 0.06 m，
仅将世界系 wrench 旋转到脚坐标系，直接对 ankle 原点的力矩应用压力中心边界。

展开 USD 实例代理检查实际碰撞体后发现：

- 后跟球心为 `(-.05, ±.025, -.03)` m，前侧球心为 `(.12, ±.03, -.03)` m，半径 .005 m。
- 平放时这些球的最低接触平面位于 ankle 下方约 .035 m；其接触点凸包并不是半宽 .04 m 的矩形。
- 另有较高的脚底 box 和前端球；不能把它们的投影都当成平放时已经承重的接触面。

正确作用点转换为 `m_sole = m_ankle - r_sole × f`，其中 `r_sole = [0,0,-.035]`。
因此 `mx_sole = mx_ankle - .035*fy`，`my_sole = my_ankle + .035*fx`。

原配置在 1.12 s 请求的两脚压力中心，经平移后 y 分别为 -43.9、-39.2 mm，
比平放接触点凸包的对应边界超出约 16.8、13.2 mm。
这违反了保持平脚的几何前提，但原 QP 仍允许该解。

1.14 s 的单个 2 ms 物理步检查：

- 实际发送力矩与预测总力矩最大差仅 `4.77e-7 N·m`，排除该时刻的力矩发送/排序错误。
- 右脚实测 roll 角加速度约 `188 rad/s²`，与硬约束假设的固定脚不一致。
- 1.18 s 右踝 roll 已到 `0.2616 rad`，接近上限 `0.2618 rad`，PKL 对应参考仅约 `0.0116 rad`。

加入 `--use-sole-wrench-geometry` 作受控对照：采用平放接触凸包内的保守矩形
（半宽 .025、后跟 .05、前侧 .12 m），并正确平移力矩。原始 PKL、任务权重、PD 和容差不变。

| 1.12 ≤ t < 1.30 s | 原配置 | 脚底几何对照 |
|---|---:|---:|
| 实测关节速度峰值 / rad/s | 16.86 | 0.402 |
| CoM 加速度预测与区间实测差的 RMS / m/s² | 3.363 | 0.0235 |

该对照支持支撑 wrench 几何失配是早期翻脚失稳的重要原因。由于同时修正了尺寸与作用点，
没有进一步将两者的贡献分开。瞬时预测和区间平均实测也不是完全等价的量。

## 2. 关节边界只约束命令积分状态，没有约束实测状态的可停止性

`test_g1.py` 调用 `integration_acceleration_bounds()` 时传入的是
`joint_position_desired` 与 `joint_velocity_desired`。它能限制积分命令，
但没有给 `predicted_acc` 添加基于实测 `q, qd` 的位置/速度/制动边界。
现有可选预测加速度幅值限制也不等价于实测状态的关节边界保护。

修正脚底几何后的实验，在 1.82 s 时：

- 左踝实测位置 `-0.26149 rad`，距下限约 `0.00031 rad`，速度 `-0.3187 rad/s`。
- 此时积分种子位置仍为 `-0.19896 rad`，命令边界看起来尚有余量。
- QP 预测该关节加速度约 `+1.32 rad/s²`，不足以在剩余距离内停止。

随后仍出现关节限位与接触响应失配；最终在 2.34 s、右脚单支撑时 QP 验证失败，
参考推进到 87.2 帧（约 .727 s 动作）。因此脚底几何修正没有解决整个跟踪问题。
这些证据也不能证明原 PKL 整段动力学不可行。

## 代码与验证

- `AccelerationQP.py` 新增可选 `sole_depth`，默认 0 保持原调用行为。
- `test_g1.py` 新增 `--use-sole-wrench-geometry`，目前为默认关闭的诊断选项。
- 新增点力在 ankle/sole 两作用点下的 wrench 残差一致性测试；全部 58 项测试通过。
- 尚未实现实测关节状态的预测约束。下一步应在保留命令积分边界的同时加入这类约束，
  并检查支撑转移请求在关节可行域内是否仍能实现。

复现几何对照：

```bash
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --motion-pkl retarget/output/B2A_131_02_stageii.pkl --duration 6 --motion-warmup-duration 1 --require-support-load --diagnose-com --use-sole-wrench-geometry --output-path /tmp/g1_sole_geometry.npz
```

[数值汇总](../acc_data/dynamics_tracking_onset/summary.json)、
[单步归因日志](../acc_data/dynamics_tracking_onset/g1_onset_attribution.log)、
[USD 几何日志](../acc_data/dynamics_tracking_onset/g1_collision_geometry.log)。
同目录保存原配置与几何对照结果、单步 QP/响应快照及测试日志。

## 后续修复验证

新增 `predicted_acc_lower/upper`，约束包含 PD 响应且排除命令同步修正的物理加速度，
并纳入求解后的边界验证。实测关节速度上下界来自 `sqrt(2*b*remaining_distance)`，
再转换成下一个控制周期的加速度上下界；`b=5 rad/s²`，关节预留 .01 rad。
这些限制不能保证接触冲击发生时模型预测仍然成立。

最初试验的二阶位置 barrier 在 0.04 s 落地阶段不可行，已弃用。
剩余行程制动版本推进到约 0.707 s 动作后，在总时间 2.34 s 失败。
失败时计划仍为右单支撑，而左右脚实测载荷约 287/302 N，暴露意外落足未被接纳的问题。
补充强承重触发后，该时刻正确切换到双支撑，但仍因冲击状态下的接触/动力学残差停止。
没有放宽验收阈值、关闭自碰撞或删除命令边界。

61 项测试通过，包括非对称物理边界中的 PD/同步修正、关节近限位制动和意外承重落足。
本轮日志和快照位于 `acc_data/dynamics_tracking_fixes/`。
2 秒静态保持回归通过：末端 CoM 误差 9.30 mm，相对初始 PKL 的关节 RMSE 0.0448 rad。
下一项尚未解决的是接触冲击状态与加速度级刚性接触模型的衔接；当前修改不代表 PKL 跟踪完成。
