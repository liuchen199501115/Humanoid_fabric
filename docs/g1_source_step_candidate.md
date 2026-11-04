# 从首次抬脚衔接源动作：离线候选检查

当前 Isaac Sim 已验证的是首次左脚支撑、右脚抬起并保持。连续左右换脚、原始动作落地及完整 PKL 跟踪尚未实现。本次新增源动作第一步的离线候选生成与必要可行性检查，没有把未通过的候选下发给控制器。

## 新增代码

- `src/fabrics_sim/reference/source_step.py`：`SourceStepCandidate` 联合采样重心、双足、源上身动作，求解一套全身 q/v/a。支撑脚保持实测姿态，摆动脚终点保留 PKL 的世界 XY/yaw，落地姿态放平到地面。摆动途中不是原始关节序列逐帧复制。
- `examples/plan_unified_source_step.py`：从成功的首次转移记录生成离线候选，并检查全身角动量对应的单脚支撑力、压力中心和关节速度。
- `tests/test_source_step_candidate.py`：验证重定时导数、静态压力中心以及角动量引起的支撑需求变化；相关 17 项测试通过。

所有输出明确标记 `ready_for_tracking=false`、`simulation_validated=false`，并禁止候选越过落地终点继续采样。即使必要检查都通过，也还需要扭矩、碰撞、接触过渡与闭环仿真验证。

## 当前结果

输入：`retarget/lafan/walk1_g1.pkl` 的第 114–129 帧，以及成功转移记录的停止参考。第一步目标世界 XY 为 `[1.504914, 4.080635] m`，距当前右脚约 50 cm。

固定重心时，现有几何求解不能完成整段摆动。经过时序与高度调整，0.8 秒候选将重心从约 0.661 m 降到 0.620 m，让摆脚随重心前移推进，401 个 500 Hz 采样点全部通过关节限位内的姿态求解。最大关节速度约为 URDF 限制的 54.7%。

但计入由这套全身 q/v/a 产生的角动量变化后，只有 63.8% 的点通过保守支撑区域检查；约 33.9% 的点要求压力中心位于实际足底矩形之外。因此不能把这一候选直接交给现有控制器执行。这里的失败只针对当前候选和固定平足单支撑假设，不证明原 PKL 的所有执行方案都不可行。

落地时重心速度约 `[-0.322, -0.980, 0] m/s`。落地后直接暂停参考会产生新的速度跳变；需要规划落地承重以及下一次换脚，并将全身角动量产生的支撑需求加入规划约束。

- [候选报告](../acc_data/lafan_unified_source_step_20260930.json)
- [候选 q/v/a 数据（不可直接播放为控制指令）](../acc_data/lafan_unified_source_step_20260930.npz)

## 复现离线检查

在仓库根目录、已配置依赖的 Python 环境中：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:examples:.runtime/qp_solvers \
python examples/plan_unified_source_step.py \
  --recording acc_data/lafan_unified_transfer_margin_20260929.npz \
  --motion-pkl retarget/lafan/walk1_g1.pkl \
  --start-frame 114 --end-frame 129 \
  --duration 0.8 --landing-height 0.62 \
  --output acc_data/lafan_unified_source_step_candidate.npz
```

该命令只计算候选，不启动 Isaac Sim，不控制机器人。终端命令成功结束表示报告已生成，候选是否通过必须看 JSON 中各项检查；`ready_for_tracking` 目前始终为 `false`。
