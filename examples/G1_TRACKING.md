## 当前选用：投影加速度基线

按要求回到已验证的 3/29 步基线：Fabric → 加权接触投影 → 单次 QP → 力矩。
默认 `amp-full` 和 `run_amp_full_motion.sh` 均使用该配置；
`qp_com_feedback_scale=0`、`qp_fabric_metric=False`，两条新实验路径关闭。
实验代码及以下测试记录保留，后续对比以此基线为准。

## 原始质心目标与 Fabric 度量传递（2026-11-23）

新增两个单次 QP 目标选项，仅用于 `--tracking-profile amp-full`：

- `--qp-com-feedback-scale s`：在原加速度代价上加入原始 COM 前馈＋PD
  残差 `s/2 * ||J_com*qdd + bias - a_des||²_Wcom`。
  `a_des`、`bias`、`Wcom` 均直接取自当周期 Fabric COM 任务，目标不经过接触投影。
  通过配方合并成一个加速度矩阵和中心，不增加 QP 次数。
  `--qp-com-feedback-start` / `--qp-com-feedback-ramp` 可指定仿真时钟上的启用时刻和
  五次平滑过渡时长；默认均为 0。
- `--qp-fabric-metric`：将完整合成矩阵 `H_fabric` 和未投影的
  `qdd_fabric` 一起传给 QP。`H_fabric` 包含各任务的 `JᵀWJ` 和原有正则项，
  因此受约束优化仍使用原任务的取舍标准；投影结果继续用于诊断。
  可叠加上一个选项以额外强调 COM，但不允许叠加投影 COM 保持权重或基座平移权重覆盖。

两种路径均保留全部原始物理约束和动作验收，不缓存 `last_requests`，不增加优先级阶段。
原始目标新增记录为 `history_com_acc_original_desired`；
`history_com_acc_qp_reference` 仍表示合成后的广义加速度中心映射出的 COM 加速度，
两者含义不同。`history_qp_com_feedback_scale` 和 `history_qp_fabric_metric` 记录所用目标。
COM 的原有反馈限幅及参考轨迹不变。

复现实验与审计位于 `acc_data/g1_com_feedback_20261123/`。单状态可行不代表
整段闭环可行，是否提升默认配置以闭环落脚、共同时间段误差和原始动作保真度为准。

已完成 1 组基线及 6 组候选的闭环验证，**没有候选整体优于基线，因此默认仍为
投影加速度目标，两个新选项均未默认启用**。原始默认基线的状态、速度、QP 加速度及
发送力矩与历史记录逐元素一致；延迟组在 0.7 s（含）之前也逐元素复现基线。

| 目标配置 | 有效落脚 / 29 | 停止时间 / s | 停止原因 |
| --- | ---: | ---: | --- |
| 原投影加速度基线 | 3 | 1.356 | QP 校验失败 |
| 原 COM 代价 ×0.25，从起步启用 | 1 | 0.716 | 原始动作误差超限 |
| 原 COM 代价 ×1，从起步启用 | 2 | 0.768 | 原始动作误差超限 |
| 原 COM 代价 ×4，从起步启用 | 0 | 0.244 | 原始动作误差超限 |
| 完整 Fabric 度量，从起步启用 | 1 | 0.334 | DAQP 返回解未通过校验 |
| 原 COM 代价 ×1，0.7 s 起、0.1 s 过渡 | 3 | 1.180 | 原始动作误差超限 |
| 原 COM 代价 ×4，0.7 s 起、0.1 s 过渡 | 2 | 0.946 | 原始动作误差超限 |

最后一组在相同的 0.946 s 时刻将世界 Y 质心位置误差从基线 **5.03 cm 降至
3.40 cm**，但有效落脚减少且提前停止，不能视为整体 tracking 改善。
延迟 ×1 在 1.18 s 的横向误差为 11.63 cm，基线同期为 10.80 cm。
完整度量组的 `DAQP 1; rejected` 仅表示校验未通过，不能据此断言约束数学不可行。
×4 起步组曾在 Isaac 初始化停顿，随后完成实际仿真；最初三次参数名冲突导致的
启动失败没有计入以上闭环试验。

56 项相关测试通过，包括：完整度量与任务目标在约束下等价、原始 COM 目标可与
投影方向相反且仍遵守力矩上限、每周期单次原生 DAQP、缓存关闭时保留必要矩阵、
原始目标与合成中心的日志区分。所有有效闭环记录均通过单次求解、无任务缓存、
发送力矩等于 QP 力矩和延迟前缀审计。结果见
[results.json](../acc_data/g1_com_feedback_20261123/results.json)，测试记录见
[tests.log](../acc_data/g1_com_feedback_20261123/tests.log)。

复现延迟 ×4 对照（实验配置，尚未优于默认）：

```bash
bash examples/run_amp_full_motion.sh --qp-com-feedback-scale 4 \
  --qp-com-feedback-start .7 --qp-com-feedback-ramp .1 \
  --output-path acc_data/g1_com_feedback_20261123/manual.npz
```

### 两条路径的独立复测（2026-11-23）

按要求再次从头运行基线、原始 COM ×1、延迟 COM ×4、完整 Fabric 度量，共四组。
全部完成实际仿真，未发生超时；状态、速度、QP 加速度及实际力矩均与各自上一轮
记录逐元素一致，分别为 3 步 / 1.356 s、2 步 / 0.768 s、2 步 / 0.946 s、
1 步 / 0.334 s。四组使用同一份控制器代码，未修改物理约束、验收阈值或默认配置。
独立记录及审计见 [复测结果](../acc_data/g1_both_paths_20261123/results.json)。

完整 Fabric 度量的失败状态已额外复算：线性可行性检查通过；原始单次 DAQP
的约束残差约 1.24e-7、驻点残差约 1.39e-12，均在原容差内，但互补性残差
4.04e-7 超过 1e-7 门槛，因而拒绝。仅在离线启用活跃集数值精修后，同一 QP
通过原有全部校验，约束残差约 4.55e-13；这没有用于闭环，也没有放宽容差。
该停止点应先处理数值求解/认证问题，不能直接解释成物理不可行。
细节见 [失败状态复算](../acc_data/g1_both_paths_20261123/fabric_failure_check.json)。

## 投影加速度基线：单次 QP 继续跟踪至 3 步（2026-11-14）

当前默认仍为 Fabric → 接触投影 → 一次 QP → 力矩，不缓存 `last_requests`。
此前 15 组有效闭环候选中，原速的 `projection1_attitude` 确认 **3/29 次落脚**，
分别为 0.176、0.412、0.934 s；落脚位置误差为 1.23、3.39、1.76 cm。
它保留全部原物理约束、辅助跟踪约束和原始动作验收。
1.356 s 时 QP 未通过校验而停止；已执行到 1.354 s 的前缀关节 RMSE 峰值
0.17518 rad，仍在原 0.2 rad 阈值内。尚未完成完整 tracking。

默认参数采用该已测候选：Fabric 关节姿态 metric ×10、关节反馈上限 150 rad/s²，
骨盆姿态 metric ×10、Fabric 角反馈上限 100 rad/s²；QP 基座加速度权重 100。
投影增加关节修正代价 1：使用 `W=M+SᵀS`，接触修正公式中的逆质量度量改为
`W⁻¹`。这是加权接触投影；接触约束及后续 QP 中的真实物理质量矩阵不变。
QP 依然只跟踪投影后的加速度，不重新引入任务列表或优先级阶段。

离线线性可行性检查显示，1.356 s 完整约束不可行，去掉辅助关节跟踪屏障后可行。
预先关闭该辅助行的独立试验只延长到 1.378 s，随后原始源关节 RMSE 超限，
仍为 3 步，因此默认保留辅助屏障。其他候选没有增加有效步数；半速结果与原速
分开记录。两次停在仿真初始化阶段的运行不计作有效闭环试验。

默认输出改为 `acc_data/g1_projected_acc_qp_20261114/continued_tracking.*`。
启用参数取自已完成的 `tuning/projection1_attitude` 试验；GPU 恢复后的独立闭环
复现与该试验的状态、QP 加速度及发送力矩逐元素一致。可选质心加速度保留代价
在权重 100、1000 的闭环测试中均只完成 1 步，默认权重继续为 0。
此前 73 项相关测试通过，并检查了默认参数与选中候选一致；全部有效候选的记录
均确认每周期一次 QP、无任务缓存、QP 参考等于投影输出，验收阈值未改变。

[全部有效候选与单次求解审计](../acc_data/g1_projected_acc_qp_20261114/tuning/results.json) ·
[3 步落脚结果](../acc_data/g1_projected_acc_qp_20261114/tuning/projection1_attitude.steps.json) ·
[失败状态可行性检查](../acc_data/g1_projected_acc_qp_20261114/tuning/projection1_attitude_feasibility.json)

### GPU 恢复后的闭环继续验证

Warp 初始化及 CUDA 张量运算通过后，完成 10 组新参数试验和 2 次默认配置复现。
结果保存在 `acc_data/g1_projected_acc_qp_20261114/continue_20261114_0338/`。
当前最好结果仍为 3/29 步；新参数未超过默认配置，因此没有修改控制器默认值。
两次基线的状态、速度、QP 加速度和发送力矩均与此前 3 步记录逐元素一致，
其中一次额外保存完整 QP 快照，也未改变轨迹。

| 修改项（其余沿用当前默认） | 落脚数 | 停止时间 / s | 停止原因 |
| --- | ---: | ---: | --- |
| 默认配置，两次复现 | 3 | 1.356 | QP 校验失败 |
| 质心加速度保留权重 100 | 1 | 0.656 | 源关节 RMSE 超限 |
| 质心加速度保留权重 1000 | 1 | 0.224 | 源根姿态误差超限 |
| QP 基座权重 10 | 1 | 0.660 | 原始源误差超限 |
| QP 基座权重 30 | 1 | 0.706 | 原始源误差超限 |
| Fabric 反馈倍数 18（默认 9） | 1 | 0.810 | 原始源误差超限 |
| 从源时钟 0 s 启用跟踪屏障 | 1 | 0.358 | QP 校验失败 |
| 从源时钟 0.9 s 启用跟踪屏障 | 3 | 1.352 | QP 校验失败 |
| 从源时钟 1.1 s 启用跟踪屏障 | 3 | 1.342 | QP 校验失败 |
| 投影关节修正权重 0.3 | 1 | 0.608 | 原始源误差超限 |
| 投影关节修正权重 3 | 2 | 1.352 | QP 校验失败 |

全部记录通过单次求解审计：每周期一次原生 DAQP、无任务缓存和优先级阶段，
QP 参考逐元素等于投影输出，按关节顺序对齐后的发送力矩逐元素等于 QP 力矩。
伺服增益记录仍非零，但在当前 500 Hz 测量状态同步配置中实际 PD 增量为零；
不能仅凭增益字段推断额外施加了反馈力矩。原始源误差、落脚位置、离地高度和
落脚保持时间阈值均保持不变。

对 1.2、1.3、1.34、1.354 s 的完整 QP 做离线线性规划，最小化所有广义加速度
的最大绝对值，得到下界 2.369、51.898、187.265、1491.075。后两时刻仅对关节
加速度取最大值也得到相同下界，单位为 rad/s²。1.354 s 的真实 QP 峰值为
2061.582 rad/s²；该快照仅在离线诊断中去掉辅助关节屏障，下界降到 317.151。
所有 LP 解对原矩阵的独立约束残差检查均不超过 1e-6。这证明末段已进入必须
使用极大加速度才能满足当前约束的状态；调目标权重不能消除这个状态的下界。
去掉屏障的历史闭环也仍只有 3 步，因此没有据此移除运行时约束。

[本轮全部闭环结果及审计](../acc_data/g1_projected_acc_qp_20261114/continue_20261114_0338/results.json) ·
[约束所需加速度下界](../acc_data/g1_projected_acc_qp_20261114/continue_20261114_0338/acceleration_bounds.json) ·
[速度误差与各级加速度](../acc_data/g1_projected_acc_qp_20261114/continue_20261114_0338/velocity_audit.json)

### 支撑膝与平移代价对照（2026-11-14 运行，2026-11-17 完成审计）

`continue_knee_20261114/` 完成 12 组新候选及两次基线复现，最好仍为 **3/29**。
两次基线均在 1.356 s 停止；最终代码与旧基线的位置、速度、QP 加速度、
QP 参考及发送力矩逐元素一致。全部记录通过单次求解、无任务缓存、投影输出
作为 QP 参考及原始验收阈值审计。12 项相关测试通过。

原始源根位置的横向误差在 1.2 s 已达 0.12533 m，1.3 s 达 0.16976 m。
1.2 s 投影请求横向加速度 -1.80889 m/s²，QP 给出 +0.62648 m/s²。
同状态提高平移代价可改善这一分量，但不能据此推断闭环会改善。
两组支撑膝屏障使 0.9 s 后支撑膝最小角度从基线 0.06652 rad 提升至
0.21455 / 0.22238 rad，仍只有 3 步，根位置漂移也未解决。

| 候选 | 落脚数 | 停止时间 / s |
| --- | ---: | ---: |
| 支撑膝阈值 0.15 / 0.25 rad | 3 / 3 | 1.354 / 1.342 |
| 延后接触速度反馈 | 3 | 1.358 |
| 源时钟保持速度 1 | 3 | 1.358 |
| 平移权重 300 / 1000 / 3000，从起步启用 | 2 / 1 / 1 | 0.586 / 0.570 / 0.198 |
| 关节跟踪屏障带宽 20 / 40 | 3 / 3 | 1.360 / 1.370 |
| 0.7 s 起平滑提高平移权重至 300 / 1000 | 3 / 3 | 1.350 / 1.086 |
| 1.0 s 起平滑提高平移权重至 300 | 3 | 1.352 |

新增可选 `--qp-base-linear-acceleration-weight`、`--qp-base-linear-weight-start`
和 `--qp-base-linear-weight-ramp`，仅调整 QP 平移加速度跟踪代价；起始时间使用
仿真时钟，过渡使用五次曲线。默认不覆盖原权重，不改变投影参考、旋转/关节代价
或物理约束。实测过渡前轨迹与基线逐元素一致，逐周期权重记录与公式一致。
候选没有提高有效落脚数，因此没有调整默认配置。

[全部结果与求解审计](../acc_data/g1_projected_acc_qp_20261114/continue_knee_20261114/results.json) ·
[权重过渡及最终基线审计](../acc_data/g1_projected_acc_qp_20261114/continue_knee_20261114/weight_transition_audit.json) ·
[根位置与支撑膝对照图](../acc_data/g1_projected_acc_qp_20261114/continue_knee_20261114/drift_comparison.png)

### 历史环境阻塞：NVIDIA UVM 初始化

本轮输出保存在 `acc_data/g1_projected_acc_qp_20261114/continue_20261114_0330/`。
宿主机 `nvidia-smi` 曾正常识别 4090 D，但随后仿真卡在 `wp.init()` 的
`self.core.init()`；进程等待点为 `uvm_gpu_retain_by_uuid`，GPU 查询进程
等待 `uvm_va_space_destroy`。内核日志报告查询进程阻塞超过 122 秒。
本轮两个启动尝试均未进入控制循环，不计为闭环试验；质心权重候选未启动。
已终止试验队列并向本轮启动的阻塞进程发送 SIGKILL；内核阻塞进程可能仍需
等待驱动恢复才能退出。当前默认配置及已验证的 3/29 步结果未改变。

CPU 离线重放准确重建了 1.356 s 失败快照的三条跟踪屏障。关节屏障当前要求
`C_joint qdd <= -42.02724`，而其余约束允许的最小值为 `-7.37128`；带宽
4、10、15、20、30、40 均不能恢复该状态的可行性，每个离线候选仅调用一次 DAQP。
屏障右端的相对速度项为 -40.66177，参考加速度项仅 -0.37713；关节速度相对
屏障参考的 RMS 误差为 4.50898 rad/s，左髋 yaw 误差为 12.74116 rad/s。
上一周期 QP 对该关节速度的预测为 12.09694 rad/s，实测为 12.10310 rad/s。
这表明需要进一步检查高速度如何在此前控制过程中增长，不能只调失败时刻的屏障。
这些离线结果不代表新增落脚或完成 tracking。

[本轮离线诊断](../acc_data/g1_projected_acc_qp_20261114/continue_20261114_0330/audit_tube.json) ·
[初始化调用栈](../acc_data/g1_projected_acc_qp_20261114/continue_20261114_0330/baseline_traced.log) ·
[GPU 内核日志](../acc_data/g1_projected_acc_qp_20261114/continue_20261114_0330/gpu_kernel.log)

## 历史记录：Fabric → 原质量度量接触投影 → 一次 QP（2026-11-14）

按用户补充要求，`amp-full` 的初始化和控制循环均将 Fabric 合成加速度先进行
动力学一致的接触投影，再将 `output.acc_dc` 作为 QP 的 `acc_reference`。
初始化先设置有效接触状态再计算 Fabric 加速度，避免投影旧的接触状态。
COM 诊断使用实际传入 QP 的投影后加速度，同时保留投影前后的加速度记录。

仍不缓存 `last_requests`，不向 QP 传任务列表，不启用优先级阶段或失败重试。
QP 在物理约束下跟踪 `qdd_dc`；命令同步修正为零，电机接收 QP 力矩。
接触投影是矩阵运算，不增加一次 QP 求解。
25 项相关测试通过。当前结果写到 `acc_data/g1_projected_acc_qp_20261114/`。
闭环的 108 个控制状态均只求解一次 QP、无任务缓存；快照及 COM 映射确认 QP
跟踪投影后的加速度。共发送 107 次力矩，确认 1/29 次落脚，0.214 s 时原始源
关节 RMSE 达到 0.200677454 rad，超过 0.2 rad 阈值而停止。
见 [投影后加速度跟踪校验](../acc_data/g1_projected_acc_qp_20261114/verification.json)。

## 历史记录：Fabric 加速度直接进入一次 QP（2026-11-14）

按用户要求，`amp-full` 不再保存内部任务列表：
`fabric.dynamics_tasks.cache_task_requests=False`，每次合成后 `last_requests=None`。
任务仍作为 Fabric 函数内部的临时数据参与合成，但外部只消费返回的 `qdd_fabric`。
当前数据流为：实测状态与参考 → Fabric 合成加速度 → 一次动力学 QP → 关节力矩。

初始化和每个控制周期均使用 `acc_reference=qdd_fabric`、`task_requests=None`，
优先级参数清空，命令同步修正为零。QP 跟踪广义加速度误差，同时保留接触力目标、
正则项以及动力学、接触、摩擦、力矩、速度、碰撞和已配置的跟踪约束。
接触投影仍用于生成接触约束及稳定项，但其投影加速度不再作为 QP 跟踪目标。
单次原生 DAQP 求解和失败即停止的规则不变。

依赖 `last_requests` 的后处理在当前路径中不再执行，包括额外支撑姿态任务、
关节软限位追加任务，以及合成后才进行的反馈修改和任务调权；Fabric 内部的原始
反馈、避碰计算和物理约束仍然执行。COM 诊断改为映射实际的合成加速度目标，
不再将单个任务或旧 attractor 的请求误记成 QP 目标。
其他实验入口可继续使用原任务缓存接口。

70 项相关测试通过，包括关闭缓存前后合成加速度一致、清除旧缓存、直接跟踪给定
加速度、失败不重试和 COM 诊断。当前输出保存至
`acc_data/g1_fabric_acc_qp_20261114/full_motion.*`；历史结果不覆盖。

闭环验证：167 个已执行控制周期及 0.334 s 的失败周期均只调用一次 DAQP，
`last_requests` 全程未缓存，命令加速度同步修正为零。0.2 s 快照验证加速度目标为
`H_acc=(1+1e-6)I`、`h_acc=-qdd_fabric`，没有任务目标覆盖，也没有任务列表快照。
本次确认落脚 **1/29**；0.334 s 因 QP 校验失败停止，未重试、未发送失败力矩。
已执行前缀的原始源误差验收通过，但没有完成完整 tracking。
见 [加速度直入 QP 校验](../acc_data/g1_fabric_acc_qp_20261114/verification.json)。

## 历史记录：每周期一次任务 QP（2026-11-14）

上一版本的 `examples/test_g1.py --tracking-profile amp-full` 将完整任务列表
按现有 metric 放进同一个 QP；不再先求优先任务，也不添加第一阶段结果锁定约束。
保留动力学、接触、摩擦、力矩、速度、碰撞及当前跟踪误差约束。
Fabric 的 `qdd_fabric` 仍用于接触投影，QP 的目标仍来自修改后的 `last_requests`。

初始化检查求解一次；之后每个控制周期只执行一次 DAQP 原生求解。
关闭关节/姿态 tracking tube 失败重试、肩部保护候选求解，以及 DAQP 的冷启动重试、
proximal 重试和额外 active-set 修复。每次唯一求解仍执行原来的可行性和 KKT 校验；
失败时停止，不使用未通过校验的力矩。`amp-full` 会拒绝启用这些额外求解选项，
并要求 DAQP 后端。其他入口的求解器默认重试行为不变。

默认输出改为 `acc_data/g1_single_qp_20261114/full_motion.*`，避免覆盖历史 12 步结果。
记录中的 `history_qp_native_solve_count` 给出各周期的原生求解次数；
`history_priority_active` 应始终为 false。历史两阶段版本的 12 步成绩不能作为
当前单次求解版本的成绩。单次求解、热启动、失败不重试、物理约束和原优先级路径
相关的 39 项测试已通过。

闭环实测：115 个控制状态的原生求解次数均为 1，优先级阶段和 tube 恢复均未启用；
共发送 114 次力矩。0.228 s 时原始源关节 RMSE 达到 0.200026929 rad，超过
0.2 rad 阈值而停止，确认落脚 **0/29**。这是任务合并后按现有权重运行的结果，
没有保留原两阶段版本的 12 步表现，也没有放宽验收阈值。
见 [单次求解校验](../acc_data/g1_single_qp_20261114/verification.json) 和
[原始源跟踪报告](../acc_data/g1_single_qp_20261114/full_motion.tracking.json)。

## 已撤回的关节角度硬约束试验（2026-11-13）

以下描述的是恢复 12 步基线前的试验版本，当前入口不再提供
`--joint-position-limits` / `--no-joint-position-limits`。该版本及后续注释清理版
已分别保存在 `acc_data/g1_restore_12step_20261113/backup/` 中的
`test_g1.before_cleanup.py` 和 `test_g1.py`；
原始基线快照见下节；当前入口在该快照上增加了上节的单次 QP 修改。

试验版本默认启用关节姿态 task，并在 `dynamics-aware` 模式下启用
`joint_limit_avoidance` task（权重 20），接近上下限时请求向内制动。
该试验的 `--joint-position-limits` 默认开启：从仿真模型读取 29 个关节的角度范围（弧度），
按关节名对齐，在初始求解和控制循环中约束预测物理加速度，使下一步角度保留
0.01 rad 余量，并按剩余距离制动。它独立于软件加速度上限开关。
试验中的软 task 可用 `--joint-limit-avoidance-weight 0` 关闭；角度约束可用
`--no-joint-position-limits` 关闭，但显式启用的其他制动选项仍然有效。
这些是模型预测约束；接触冲击和模型误差下的实测效果仍需闭环仿真验证。
新硬约束版本在 0.264 s 因原始 PKL 关节 RMSE 超限停止；注释清理前后结果一致。

## 历史恢复记录：原 12 步基线（2026-11-13）

当时按用户要求，`examples/test_g1.py` 恢复到质心速度软约束接入前的已验证快照
（SHA256 `e738f5460f70f46040887b1f30168220badff1239c6646a8a142342661d9a7a9`）。
运行 `examples/run_amp_full_motion.sh` 使用 `amp-full`，肩权重默认 1、保护调权默认关闭，
质心速度软约束已从入口移除。该基线最近验证为 12/29 步、4.300 s 根位置误差超限停止。
实验模块、代码快照和结果继续保留；下文软约束参数属于历史实验，不适用于当前入口。
本次恢复也撤回了仅允许 500 Hz、最长 18 s 的清理版入口限制，并归档了对应测试。
QP 动力学求解器、运动源文件及记录中校验的辅助模块与原基线哈希一致，无需替换。
本次复现结果保存至 `acc_data/g1_restore_12step_20261113/restored.*`，不覆盖历史结果。
复现确认 **12/29 次落脚、4.300 s 停止**，根位置误差为 0.201005965 m，超过原始
0.2 m 阈值。2150 次实际力矩更新，以及位置、速度、源帧、QP 力矩和末状态，
均与保存的 12 步基线逐元素一致；79 项相关测试通过。
见 [本次恢复校验](../acc_data/g1_restore_12step_20261113/verification.json)。

## WBC 任务冲突诊断（2026-11-13，无 MPC）

新基线重跑仍为 12/29 步、4.300 s 停止，实测位置/速度/源帧与历史基线逐元素一致。
采集 3.48–4.28 s 的 7 个状态，对腰部、双臂、腰部＋双臂分别降权至 0.9/0.75/0.5，
共 70 组双阶段 QP（含各状态基线）均通过原约束校验。COM 加速度残差最大改善仅
2.74%，更强降权还影响骨盆、关节或摆脚残差；没有候选进入闭环，主控制器未改。

3.58 s 在当前完整约束下独立最大化 COM 竖直加速度，得到 +0.943 m/s²，上游请求却为
+10.080 m/s²；右踝 pitch 的 20 N·m 预算是该极值对力矩界最敏感的一项。
3.96 s 可精确满足 COM 请求，但躯干加速度残差从 23.10 增至 56.97 rad/s²。
4.216 s 右脚法向力达到 800 N 的 QP 上限，多关节接近预算；4.28 s 还存在很大的
预测/实测加速度差。因此不能将问题简单归结为腰部或上肢权重过高。

[完整诊断及限制说明](../acc_data/g1_wbc_conflict_20261113/REPORT.md) ·
[同状态结果](../acc_data/g1_wbc_conflict_20261113/conflicts.json) ·
[可行加速度区间](../acc_data/g1_wbc_conflict_20261113/com_feasible_ranges.png)

## 质心速度软约束试验（2026-11-12，已撤出主入口）

新增默认关闭的 `--com-velocity-soft-weight`：用显式非负松弛变量，软约束
50–100 ms 恒加速度外推后的逐轴质心速度误差，有限惩罚加入现有第一阶段 QP。
保持全部物理约束与原始 PKL 验收，不叠加肩部调权。

**本轮 4 组有效候选均只通过 9/29 次落脚，退化于原基线的 12/29 次。**
权重/预测时间为 1/100 ms、0.1/100 ms、0.1/50 ms、0.01/100 ms，
分别在 3.574、3.748、3.682、3.720 s 因关节 RMSE 超限停止。
后三组第 10 步带载落脚位置误差分别为 14.89、8.52、12.91 cm，均超过 4 cm。
速度状态触发首次激活在 2.562 s，此前实测轨迹与基线逐元素一致。

36 项测试通过。首次尝试的混合精度适配错误已修复并从零重跑，原中断记录保留，
不计作有效候选。没有改善跟踪步数，因此不替换默认配置。

[完整报告与复现命令](../acc_data/g1_shoulder_roll_20261112/com_velocity/REPORT.md) ·
[独立落脚审计](../acc_data/g1_shoulder_roll_20261112/com_velocity/results.json) ·
[速度与软约束诊断](../acc_data/g1_shoulder_roll_20261112/com_velocity/velocity_audit.json)

## 肩 roll 同状态保护试验（2026-11-12）

新增可选 `--shoulder-roll-guard`：同一状态先求原权重基线，再尝试肩权重增益，
检查 COM、骨盆/躯干、左右脚及其余关节的加速度残差，拒绝候选后恢复基线求解。
该检查只约束当前任务折中，不保证后续落脚或稳定性。

本轮 1 组基线和 3 组候选的 GPU 闭环结果：3.3 s 早启用仅通过 9 步；
3.58 s 晚启用两组均保留 12 步。较宽保护余量将 3.8–4.2 s 双肩 roll RMSE
从 0.23449 降至 0.18504 rad（降低 21.1%），全身 RMSE 仅降低 0.9%。
两组晚启用候选在 4.248/4.240 s 因关节误差超限停止，早于基线 4.300 s。
**仍为 12/29 次落脚，第 13 步未通过，未替换默认配置。**

28 项测试通过；启用前轨迹与基线逐元素一致，原始 PKL 和验收阈值未改。
保护版本求解耗时中位数约 8.5–11.1 ms，尚不满足 2 ms 实时预算。

[保护试验完整报告及复现命令](../acc_data/g1_shoulder_roll_20261112/guarded/REPORT.md) ·
[独立审计](../acc_data/g1_shoulder_roll_20261112/guarded/results.json) ·
[实际误差曲线](../acc_data/g1_shoulder_roll_20261112/guarded/tracking_comparison.png)

## 肩 roll 调权闭环对照（2026-11-12）

新增 8 组从第 0 帧开始的物理试验，重新加载原始 PKL 并独立检查实测落脚。
最佳仍为 **12/29 次落脚，第 13 步未通过**，没有替换默认 `amp-full`。

`--shoulder-roll-weight-start` 已修正为仿真/控制时间；此前沿用源时钟的两组候选
实际在 2.93 s 开始调权，完整证据保留。修正后的 3.3 s 启动 ×2/×4 试验均退化为
9 步。改在基线第 10 次落脚时刻 3.58 s 后平滑启用 ×2/×4/×8，均保留前 12 步，
且调权前的实测轨迹与基线逐元素一致。

3.8–4.2 s 双肩 roll 的实际角度 RMSE 从基线 0.23449 rad 分别降至
0.17136/0.12291/0.08455 rad；全身 RMSE 不随之单调下降。
×4 在第 12 次落脚时将关节 RMSE 从 0.14348 降至 0.12552 rad，但根姿态误差
从 0.20614 增至 0.27633 rad，最终在 4.304 s 因关节 RMSE 超限停止。
×8 在 4.314 s 因根姿态超限停止；这些都不构成完整 tracking。

沙箱外 GPU 可用，先前沙箱内的 NVIDIA 探测失败不是主机驱动故障。
时钟修正后的 22 项回归通过；默认肩权重仍为 1，实验参数默认关闭。

[完整闭环结果与复现命令](../acc_data/g1_shoulder_roll_20261112/closed_loop/REPORT.md) ·
[原始 PKL 独立审计](../acc_data/g1_shoulder_roll_20261112/closed_loop/results.json) ·
[实际误差对照图](../acc_data/g1_shoulder_roll_20261112/closed_loop/tracking_comparison.png)

## 第 10 步起的偏差追查与修改（2026-11-07）

本轮从第 10 步前后重新采集 QP 快照，比较任务要求、预测加速度、实际响应和
原始 PKL 误差。新增可选的质心状态调权，入口为 `test_g1_2.py --tracking-profile drift-wbc`，
没有替换 `test_g1.py` 或改变原始源文件及验收阈值。

**已验证的局部改善：**三轴状态调权保留前 10 步轨迹，3.646 s 开始激活，
3.648 s 首次出现实测状态差异。第 12 次落脚时，根位置误差由 0.1761 m 降至
0.1379 m，根姿态误差由 0.2061 rad 降至 0.1018 rad，关节 RMSE 由 0.1435 rad
降至 0.1070 rad。最终仍在第 13 步完成前因姿态误差超限停止，不能宣称完整跟踪。

### 定位到的限制

- 3.58 s 第 10 次落脚时，根位置误差仅约 0.0159 m，但质心竖直速度误差已为
  -0.559 m/s。该时刻期望质心竖直加速度约 +10.08 m/s²，QP 预测约 -5.19 m/s²，
  实测约 -5.14 m/s²。位置验收通过没有排除较大的速度偏差。
- 3.65 s，质心期望加速度约 `[5.46, 6.55, 4.58]` m/s²，QP 预测约
  `[-0.61, 1.70, 8.39]`，实测约 `[-0.62, 1.61, 8.38]`。该快照的主要缺口
  已在 QP 输出中出现；这不代表所有接触冲击区间都具有同样准确的响应模型。
- 独立固定状态检查中，3.48、3.58、3.65、3.75 s 的原质心加速度目标均无法
  在完整已存约束下精确实现。提高质心权重能改善部分水平纠偏，但不能保证目标
  全部实现。单独降低支撑腿姿态任务权重也没有稳定的离线收益。
- 3.58 s 右踝 pitch 已达到旧软件预算 20 N·m；URDF 和仿真的共同限值为
  50 N·m，原流程稍后才切到其 80% 预算 40 N·m。离线使用 40 N·m 能减轻
  竖直加速度缺口，但闭环中单独提前切换只通过 10 次落脚。因此没有将其作为
  默认修复，也没有提高模型或仿真的物理限值。

以上说明当前问题包含接触/执行能力限制和任务取舍，尚未证明某一个因素是唯一
根因。离线约束移除仅用于诊断，不能将数值求解失败视为不可行证明；报告中
HiGHS 状态 4 表示数值问题，状态 2 才是其不可行判定。

### 新增公式

在 [`g1_response_feedback.py`](g1_response_feedback.py) 的
`com_tracking_metric()` 中，用当前控制参考的质心误差和速度误差构造：

$$
r=\|e_{xy}+T\dot e_{xy}\|,\qquad
s=\operatorname{clip}\left(\frac{r-r_0}{\Delta r},0,1\right),\qquad
w=1+(w_{max}-1)(6s^5-15s^4+10s^3).
$$

本轮取 $T=0.1$ s、$r_0=0.04$ m、$\Delta r=0.04$ m、$w_{max}=4$。
默认三轴使用 $W_{COM}^{new}=wW_{COM}$；可选 `xy` 通过对角矩阵左右缩放
保留度量半正定性。此线性外推只是状态触发器，不是未来接触可行性预测。
参考位置/速度/加速度、原物理约束、源时钟和原始 PKL 审计均保留。

复现已验证的三轴状态调权：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/test_g1_2.py \
  --tracking-profile drift-wbc
```

`drift-wbc` 等价于 `response-wbc --response-gain 0 --state-com-maximum 4 --state-com-axes xyz`；
参数一致性有专门测试。其他既有配置不默认启用该状态调权。仅水平调权退化为 10 次
落脚；不能将相同倍数在不同任务方向上的效果等同。固定源时刻提前升权的对照
在第 12 次落脚时根位置误差更低，但第 13 步离地不足，也没有通过该步。

| 配置 | 已验证落脚前缀 | 停止时刻 | 停止原因 |
| --- | --- | --- | --- |
| 原基线 | 12/29 | 4.300 s | 根位置误差超限 |
| 固定时刻质心权重 ×4 | 12/29 | 4.464 s | 根姿态误差超限 |
| 状态调权，三轴 ×1–4 | 12/29 | 4.378 s | 根姿态误差超限 |
| 状态调权，仅水平 ×1–4 | 10/29 | 3.992 s | 关节 RMSE 超限 |
| 第 10 步前平滑减慢源时钟 | 9/29 | 3.584 s | 关节 RMSE 超限 |
| 提前使用模型力矩预算 | 10/29 | 3.996 s | 关节 RMSE 超限 |
| 固定时刻质心升权＋提前力矩预算 | 10/29 | 3.978 s | 关节 RMSE 超限 |

状态公式、配置、原有完整跟踪及模型力矩回归共 **45 项通过**。
共完成 7 组从第 0 帧开始的物理试验；首次诊断启动因重复时间参数写法错误未进入
仿真，已修正并重跑，不计入 7 组物理结果。

- [全部试验与原始 PKL 独立审计](../acc_data/g1_2_tracking/step10_diagnosis/results.json)
- [逐样本前缀及落脚误差对比](../acc_data/g1_2_tracking/step10_diagnosis/prefix_comparison.json)
- [质心请求／预测／实测分析](../acc_data/g1_2_tracking/step10_diagnosis/response_zero.drift.json)
- [同状态任务权重检查](../acc_data/g1_2_tracking/step10_diagnosis/priority_audit.json)
- [约束分组可行性检查](../acc_data/g1_2_tracking/step10_diagnosis/constraint_audit.json)
- [力矩预算离线检查](../acc_data/g1_2_tracking/step10_diagnosis/effort_audit.json)
- [源帧对齐的对比曲线](../acc_data/g1_2_tracking/step10_diagnosis/tracking_comparison.png)
- [45 项测试日志](../acc_data/g1_2_tracking/step10_diagnosis/tests.log)

## 从十二步记录提炼的状态公式（2026-11-07）

已在 `test_g1_2.py` 接入可选 `response-wbc` 实验配置，公式实现位于
[`g1_response_feedback.py`](g1_response_feedback.py)。本轮 17 组从第 0 帧开始的
完整源文件跟踪尝试，独立审计后最好仍为 **12/29 次落脚，第 13 步未通过**。
没有实现完整 2000 帧 tracking，也没有替换 `test_g1.py` 的默认基线。

这里能提炼的是状态反馈候选公式，不能仅凭前十二步推导出保证全程成功的控制律。
状态层保留基线的历史时序、参考预处理和源时钟；未实现未来接触可行性预测。
原始 PKL、关节 RMSE 0.2 rad、根位置 0.2 m、根姿态 0.35 rad 及落脚验收保持不变。

### 公式与代码对应

对关节当前位置 $q_i$、速度 $v_i$，取运动方向上的剩余限位距离：

$$
d_i=\begin{cases}q_i^{max}-q_i-\delta,&v_i\ge0,\\q_i-q_i^{min}-\delta,&v_i<0,\end{cases}
\qquad b_i=\frac{v_i^2}{2\max(d_i,\epsilon)}.
$$

使用 $S(s)=6s^5-15s^4+10s^3$，$s\in[0,1]$，构造平滑制动任务：

$$
g_i=S\!\left(\operatorname{clip}\left(\frac{b_i-b_0}{b_0},0,1\right)\right),
\quad a_i^{brake}=-\operatorname{sign}(v_i)\min(b_i,a_{max}),
\quad W_i=w g_i.
$$

实现为 `stopping_distance_request`。默认余量 $\delta=0.03$ rad，
$\epsilon=0.001$ rad，$a_{max}=200$ rad/s²；`--state-braking-budget 25`
是经验激活门限，不是电机保证可提供的减速度。基线 0–4.1 s 稳定区间腰部 roll
最大制动需求约 21.9 rad/s²，4.20 s 已约 52.4 rad/s²。此结论不代表整个十二步
前缀都处在门限以内。软任务可能被其他目标压过；可选 `--state-braking-hard`
增加激活关节的加速度不等式，但可能使 QP 不可行。

根位置恢复使用原始源参考 $p_r,v_r,a_r$，按实测误差激活：

$$
e_p=p-p_r,\quad
\gamma=S\!\left(\operatorname{clip}\left(\frac{\|e_p\|-e_0}{0.05},0,1\right)\right),
\quad a_p^*=a_r+\operatorname{clip}(-K_p e_p-K_d(v-v_r),-30,30),
\quad W_p=w_p\gamma I.
$$

实现为 `root_recovery_request`，逐分量限幅，默认 $K_p=100,K_d=20,e_0=0.1$ m。
可选 `--state-root-angular-scale` 同时提升已有源姿态任务的权重。

另提供预测响应残差补偿，使用已完成时间段和已通过校验的 QP 预测：

$$
r_k=\frac{v_k-v_{k-1}}{\Delta t}-\hat a_{k-1},\quad
\hat d_k=(1-\alpha)\hat d_{k-1}+\alpha\operatorname{clip}(r_k),
\quad\alpha=1-e^{-\Delta t/\tau},\quad
 a_{task}^{new}=a_{task}-\eta\gamma A\hat d_k.
$$

实现为 `ResponseFeedback`。接触模式变化重置估计，接触力突变时暂停学习并衰减
估计；默认不补偿关节通道。持续开启该补偿会破坏较早的落脚，当前没有优于基线
的证据，因此下方代表性制动试验设 `--response-gain 0`。

### 实测结果

| 代表试验 | 已验证落脚前缀 | 停止时刻 | 结果 |
| --- | --- | --- | --- |
| 零补偿基线 `response_zero` | 12/29 | 4.300 s | 根位置超限 |
| 持续残差补偿 0.25 / 0.75 | 1/29 / 1/29 | 2.398 / 0.890 s | 关节 RMSE 超限 |
| 软制动 `stop25_320` | 12/29 | 4.300 s | 根位置超限 |
| 软制动＋根位置恢复 `stop25_root500` | 12/29 | 4.314 s | 根姿态超限 |
| 硬制动 `stop25_hard` | 12/29 | 4.262 s | QP 校验失败 |
| 硬制动＋辅助约束恢复 `stop25_hard_recovery` | 12/29 | 4.268 s | 关节 RMSE 超限 |
| 硬制动＋位姿恢复＋辅助约束恢复 | 11/29 | 4.278 s | 关节 RMSE 超限 |

`stop25_root500` 在相同落脚前缀下推进到源帧 992.36；这不构成第十三步或完整
文件成功。更早激活的另一组虽推进到 997.64 帧，却只验证 11 次连续落脚，不能
按帧号判定改善。

两组硬制动失败快照的独立 HiGHS 检查表明：原约束不可行；仅移除辅助姿态管道
一行后可行，最大约束残差分别约 4.95e-11、4.06e-10。这只是对应快照的证据。
`--state-soft-orientation` 在原恢复流程失败后，将辅助关节/姿态管道改为软任务，
保留根位置管道、原物理约束和新增硬制动行；实际原始 PKL 误差仍逐步验收。
此恢复跨过了原 QP 失败点，但随后关节误差超限，不能当成闭环跟踪成功。

复现 `stop25_root500`（已知会在约 4.314 s 失败的实验配置）：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/test_g1_2.py \
  --tracking-profile response-wbc --response-gain 0 \
  --state-limit-priority waist --state-braking-budget 25 --state-limit-weight 320 \
  --state-root-metric 500
```

新增状态反馈、制动、根位置恢复及约束恢复测试 13 项；连同公式测试 8 项和既有
完整跟踪回归 15 项，**36 项全部通过**。单元测试通过不代表物理整段跟踪通过。
各次试验的命令、版本哈希、日志和失败快照保存在结果目录。

- [结果摘要](../acc_data/g1_2_tracking/response_formula/summary.json)
- [17 组结果与原始 PKL 独立审计](../acc_data/g1_2_tracking/response_formula/results.json)
- [失败约束独立可行性检查](../acc_data/g1_2_tracking/response_formula/constraint_audit.json)
- [误差及腰部限位对比曲线](../acc_data/g1_2_tracking/response_formula/tracking_comparison.png)
- [36 项测试日志](../acc_data/g1_2_tracking/response_formula/tests.log)

## `test_g1_2.py` 统一公式对照（2026-11-07）

新增 `--tracking-profile formula-wbc`，复用已有 `PhaseTrackingLaw`，并与
`twelve-wbc` 使用相同的 direct WBC / tracking servo 执行方式。
`--formula-scope weights` 仅替换误差权重分配；默认 `phase` 同时使用
接触事件卸载、从起始时刻生效的物理关节制动约束和固定源时钟。
该版本尚未实现根据未来动力学可行性调整参考时间的预测控制。

五组物理试验均从第 0 帧开始，保留原始 PKL 误差阈值和真实腾空/承重
落地检查。独立重载原始 PKL 并检查足端 FK 后，结果为：

| 试验 | 通过的落地前缀 | 停止时间 | 原因 |
| --- | --- | --- | --- |
| `twelve-wbc` 基线 | 12/29 | 4.300 s | 原始轨迹误差超限 |
| 公式权重，上限 4 | 1/29 | 0.260 s | 关节 RMSE 超限 |
| 公式权重，上限 1.2 | 6/29 | 2.384 s | 关节 RMSE 超限 |
| 完整状态规则，保留预处理参考 | 3/29 | 1.292 s | QP 校验失败 |
| 完整状态规则，原始参考 | 1/29 | 0.334 s | QP 校验失败 |

因此这版统一公式尚不能替代十二步基线，也没有实现整段 tracking。
QP 失败时打印的加速度来自未通过校验的候选解，不能当成已执行的物理响应。
`phase` 默认仍使用历史预处理参考；`--formula-source-reference` 排除该参考、
摆腿净空修正及参考平滑。两种参考分别测试，不能混为同一个试验。

在 `humanoid_fabric` Python 环境中复现状态规则试验：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python examples/test_g1_2.py --tracking-profile formula-wbc
```

[完整结果及逐次复现命令](../acc_data/g1_2_tracking/formula_wbc_verification/results.json)，
[测试结果](../acc_data/g1_2_tracking/formula_wbc_verification/tests.json)。
新增公式测试 8 项、既有 tracking 回归 15 项通过。首次 `phase/source` 启动
因旧时钟函数拒绝无限切换时间失败，修正为起始时刻 0、恒定速度 1 后以
`*_v2` 重跑；首次启动失败记录保留，但不计入上表五组物理结果。
本轮没有修改 `test_g1.py` 或替换其默认基线。

## 当前第 13 步验证（2026-11-06）

入口仍为 `examples/test_g1.py --tracking-profile amp-full`。
原始 `amp_walk_1ms.pkl` 共 2000 帧、200 Hz、29 次有序落地；目前独立确认的
最佳前缀是 **12/29，第 13 步尚未通过**，完整 PKL tracking 也未通过。
本轮候选没有替换默认 profile 或启动脚本。所有物理试验从第 0 帧开始，
保留原始位姿误差、真实腾空/承重保持和 QP 物理约束检查。

- [第 13 步试验汇总](../acc_data/g1_full_tracking_20261106/step13_progress.json)
- [腰部限位冲击与独立可行性复核](../acc_data/g1_full_tracking_20261106/step13_waist_stop.json)
- [限位、根部误差和支撑力曲线](../acc_data/g1_full_tracking_20261106/step13_failure_trace.png)

下文保留历史试验记录；历史段落中的“当前”仅指对应日期。

> Historical experiment log: the ZMP planners, their CLI options and the legacy launcher were removed on 2026-10-13. Commands in those historical sections are no longer supported. Use `run_walk1_four_steps.sh` for the current mocap pipeline.

## 首步接入状态审计（2026-10-04）

新增 `examples/plan_tracking_first_step.py`，读取已通过的首个承重转移记录，
分别检查实际姿态与名义全身参考，并以原始落脚目标规划 114→129。
这是离线预检查，不修改机器人状态，也不执行未通过的参考。

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/plan_tracking_first_step.py \
  acc_data/walk1_staged_baseline_v2_20261004/transfer.npz \
  --profile acc_data/walk1_full_stride_profile_20261004.json \
  --output acc_data/walk1_handoff_new.json
```

`run_tracking_stages.py` 也支持 `--first-step-profile`：站立、转移通过后执行该
检查；失败时返回非零，不自动进入物理首步。通过离线检查也不等于可以无缝接入，
实际状态差异仍单独报告，后续需要连接段与闭环验证。

本次实测：保持末端左肩 pitch 与名义参考相差 0.25627 rad，根位置相差
12.13 mm，最大广义速度差 0.01091。实际姿态的静态覆盖间距为 +56.80 mm，
名义参考为 -43.27 mm，后者不满足碰撞屏障。实际姿态的静态检查不代表
其瞬时加速度可行性；覆盖球重叠也不等同于网格穿透。
从名义参考规划的完整首步仍有 751 个碰撞屏障违规样本，最小间距
-80.71 mm，即使力矩/接触/关节检查已通过，也会拒绝下发。
报告：`acc_data/walk1_staged_handoff_20261004.json` 及 `.candidate.json`。
接入/阶段/碰撞/首步相关 17 项测试通过。完整 PKL tracking 状态仍为未通过。

实测状态候选对照：保持实测腰部/上肢，使用实测 q/v，初始加速度暂设为零
（这是离线假设，不是实测加速度）。0.62 m 落地质心候选通过全部摆动碰撞检查，
最小覆盖间距 46.82 mm，但在 1.414 s 左踝 pitch 的参考加速度为 0.3741，
所需下界为 2.1084 rad/s²，未通过制动检查。提高到 0.63 m 后，最小间距
50.78 mm，仍在末端右踝 roll 违反制动下界，且少数样本的支撑力检查未通过。
两者均未下发；0.63 m 对照只规划摆动，不含停止段，不能视为完整首步。
报告为 `acc_data/walk1_staged_measured_upper_20261004.json` 和
`acc_data/walk1_staged_measured_upper_h63_20261004.json`。
上述离线入口支持 `--measured-upper-body --landing-height .63 --stop-duration 0`
复现此类候选；实验选项不改变分阶段入口或物理控制器的默认配置。
后续需联合修正终端质心/制动与上肢参考，并核验输出起点的 q/v 连续性，
而非将测量状态直接替换为零速度或放宽硬约束。

## 固定基线的分阶段验收（2026-10-04）

入口 `examples/run_tracking_stages.py` 先做 4 s 站立，再做 16 s 首次承重转移/抬脚诊断。
两个阶段采用相同的控制、碰撞和初始化配置：500 Hz、原覆盖球保护、平足初始化、
5 s warmup、无上肢偏移。站立阶段在 warmup 内结束，源帧必须保持不动。
转移阶段复用 `--unified-transfer-test`；两个阶段为独立仿真，第二阶段重新初始化，
不是从上一条记录恢复状态。任何阶段进程或验收失败都会停止，且不会宣布全文件通过。

```bash
/home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/run_tracking_stages.py \
  --output-dir acc_data/walk1_staged_new_run
```

输出目录必须不存在，避免覆盖证据；保存实际命令、日志、QP 快照和 `status.json`。
站立需满足约 4 s 完整 500 Hz 记录、源帧不变、始终双支撑、质心最大误差 ≤ 3 cm、
最后一秒两脚载荷各 ≥ 20 N、质心速度 ≤ 2 cm/s，以及控制器验证通过。
转移复用现有实测抬脚、支撑载荷与稳定保持验收，不以源帧推进代替动作完成。

本次 `acc_data/walk1_staged_baseline_v2_20261004` 的站立通过：峰值质心误差
9.908 mm，末秒双脚最低载荷 152.13 / 153.45 N，质心最高速度 0.540 mm/s。
同配置的 16 s 承重转移/抬脚阶段也通过：11.218 s 释放右脚，随后保持左脚
单支撑 4.780 s。末秒右脚最低净空 39.072 mm、接触力为 0，左脚最低载荷
321.30 N，最大质心误差 1.934 mm；8000 个 500 Hz 控制样本。源帧止于
114.036，尚未执行原始首步落地或完整文件。后续应沿用该基线验证抬脚放回与
完整首步，不能据此认定原始 PKL 保真度通过。阶段/初始化/转移相关 19 项测试通过。

独立 HiGHS 可行性复核中，当前初始 QP 存在可行解（最大约束残差 7.44e-9）；
旧的 0.3 rad 上肢偏移试验在 0.014 s 的 QP 被判不可行。两者采用不同初始化配置，
此对比不能单独确定是哪一项配置导致差异。证据见目录内 `initial_feasibility_comparison.json`。

## 凸网格核对与上肢控制参考修正（2026-10-04）

覆盖球的重叠不能解释为真实网格穿透。对原始帧 113、114、118、122、129
的最坏覆盖球对做独立凸包优化，凸包均未相交，距离为 1.9–18.2 mm。
MuJoCo 凸体距离查询与这些结果吻合；例如第 122 帧由覆盖球的
-74.2 mm 变为凸包距离 +1.916 mm。普通定向包围盒仍过于保守。
这些正间距依然小于当前 20 mm 保护余量。

新增可选 `--convex-wrist-collision`，仅用于 CPU 的手腕—身体距离计算；
Isaac 物理仿真、原有碰撞对、环境障碍物、接触和关节/力矩限制保留。
距离使用原碰撞网格的凸包；橡胶手沿用与原保护模型相同的 visual 网格。
返回最近点的距离雅可比，偏置通过原有有限差分计算。
运行时任务与离线屏障审计共享几何；未默认开启，也未减小保护余量。
可选依赖为本地 `mujoco`，这里只调用几何与运动学，不在其中做物理步进。
凸几何闭环尚未通过：首次试验在 3.656 s 停止；修复查询截断位于屏障激活
阈值的雅可比跳变后，重试在 5.128 s 的承重转移阶段停止。该选项保持实验性，
不能据此推荐为已稳定的整段控制配置。距离/最近点切换对加速度屏障的影响
仍需进一步验证。

新增 `--source-arm-roll-offset`：对控制参考的左右肩 roll 分别加正/负偏移。
原始运动数组保留，`history_reference_q` 和 PKL 误差仍使用原始轨迹。
原始根、腿部、接触序列及源落脚目标不变；全文件关节限位检查不通过时拒绝偏移。
这是一项明确的上肢适配，不是原始 PKL 的逐关节精确重放。

0.2 rad 上肢外展、0.622 m 落地质心、完整原始首步落点和 0.26 s 停止段，
分别通过 751 / 131 个 500 Hz 必要约束样本；已包含凸网格碰撞屏障。
报告见 `acc_data/walk1_first_full_stride_arm_clearance622_20261004.json` 和
对应 `.stop26.json`。相关 71 项测试通过，含穿透距离符号、平移/旋转雅可比、
真实 G1 任务行布局以及原始参考保持不变的验证。

但固定偏移不能覆盖全文件：全部 7839 个源姿态在偏移 0.2 rad 后，仍有
797 帧小于 20 mm 余量，其中 468 帧有凸包重叠，最小间距 -69.64 mm
（第 5097 帧右橡胶手—右髋 yaw）。这只是静态几何检查，没有验证全段动力学。
数据见 `acc_data/walk1_full_arm_geometry_20261004.json`。
后续需要随源动作变化的上肢避碰参考，不能把首步参数直接宣布为全片解。

原覆盖球保护下另做了 0.3 rad 上肢偏移试验，保持完整原始首步落点，
并按上肢变化补偿终端质心。751 个摆动样本及 0.27 s 制动的 136 个样本
通过离线必要约束检查（含碰撞屏障），但物理仿真在初始化 0.014 s
触发 QP 校验失败，尚未进入源摆动。此结果表明还必须解决初始化/承重
转移的闭环可行性，不能直接使用离线首步证书宣布 tracking 成功。
报告为 `acc_data/walk1_first_full_stride_sphere_arm_20261004.json`、
对应 `.stop27.json`，失败记录为 `acc_data/walk1_sphere_arm_live_20261004.failure.npz`。
该试验使用下方命令去掉 `--convex-wrist-collision`，上肢偏移改为 `.3`、
停止时长改为 `.27`、配置改为 `acc_data/walk1_sphere_arm_profile_20261004.json`。
两种几何方案均未通过闭环，均非完整文件运行的推荐参数。

首步物理诊断配置：

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction 1 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .26 --source-support-switch \
  --source-contact-events --source-step-count 1 \
  --source-step-profile acc_data/walk1_convex_arm_profile_20261004.json \
  --source-swing-pose-priority --balance-body-priority --separate-balance-priority \
  --convex-wrist-collision --source-arm-roll-offset .2 \
  --duration 16 --output-path acc_data/walk1_convex_arm_trial.npz
```

以下旧记录保留为诊断历史，均不证明完整 PKL tracking 已通过。

## 已定位参考与碰撞约束不一致（2026-10-04）

完整首步仍未通过闭环，完整 `walk1_g1.pkl` tracking 未完成。
摆腿优先级接入修复了早期位置误差，但不能消除参考自身与约束的冲突。

补查使用与运行时相同的手腕—身体覆盖几何和加速度屏障：

- 完整首步参考的覆盖几何最小间距 -0.08063 m，876 个摆动及制动样本
  均违反至少一项屏障约束；最坏加速度约束残差 6.99148 m/s²。
- 原始 PKL 的抽样帧 113、114、118、122、129 也有覆盖几何重叠；
  第 122 帧为 -0.07421 m。这里只证明当前覆盖模型与这些参考冲突，
  不等同于精确网格穿透深度，也不是整段碰撞检查结果。
- 同时优先身体与质心的闭环试验，在 10.848 s 再次提前触地并触发 QP 失败。
  单纯把优先级从脚移到身体，会转移跟踪误差，不能解决参考不相容。

已将运行时屏障提取为共用公式，源步及双支撑制动/尾段审计现在也检查
启用的碰撞屏障。真实 G1 重放验证中，该候选在下发前被拒绝；报告为
`acc_data/walk1_first_full_stride_collision_checked_20261004.json`。
原先的“必要约束通过”报告未检查此项，不能继续作为完整可执行证书。
旧的五步实测结果仍是历史记录；开启碰撞屏障后，重跑旧参考也可能被新增
预检查拒绝，需要先修正参考。没有关闭碰撞保护或放宽原始轨迹验收标准。
碰撞约束、全身任务、源步、停止及优先级相关 58 项回归通过。

下一步需校核覆盖模型是否过于保守；若模型正确，应在保留源脚步目标的
前提下重规划上肢参考，再把碰撞、动力学、落地与源轨迹偏差统一验收。
不能把减小摆腿误差、源帧推进到 129 或历史五次换脚当成整段成功。

证据：`acc_data/walk1_full_stride_collision_audit_20261004.json`、
`acc_data/walk1_original_collision_samples_20261004.json`、
`acc_data/walk1_swing_body_priority_live_20261004.failure.npz`。

## 统一源步的摆腿优先级对照（2026-10-04）

新增两个互斥的实验开关，均要求统一全身参考与 `--coordinated-source-start`：

- `--source-swing-position-priority`：质心优先，其次摆动脚 XYZ。
- `--source-swing-pose-priority`：质心优先，其次摆动脚位置与姿态。

选择作用于实际单支撑的离地脚，不依赖旧的 `com_preview` 对象，
也不会被第二步以后的腿部姿态分支覆盖。双支撑、承重转移和停止阶段保留
原有选择。身体姿态选项仍有效：`--balance-body-priority` 将身体与质心
放在第一层（同时需要 `--separate-balance-priority`）；
`--body-orientation-priority` 将身体与摆动脚放在第二层。
所有阶段保留物理接触、关节、力矩和碰撞约束。

完整原始首步 114→129 使用未缩短的落点、1.5 s 摆动、0.25 s 制动，
参考配置见 `acc_data/walk1_full_stride_profile_20261004.json`。
摆动与制动参考均通过 500 Hz 离线必要约束检查，但这不保证闭环通过。

同一 10.30–10.87 s 区间，原控制的摆动脚最大位置误差为 0.09987 m，
XYZ 优先级降为 0.001321 m，均方根误差从 0.02751 m 降为 0.000734 m。
证据见 `acc_data/walk1_swing_priority_comparison_20261004.json`。
该方案仍在 11.448 s 失稳；完整足端姿态优先级也在第 129 帧前后因
没有实测落地而停止。两次记录均不构成成功的首步或整段 tracking。
姿态记录显示骨盆已在 10.8 s 开始显著偏离，必须同时检查身体稳定性。

下面是首步诊断命令，不是完整 PKL tracking 的通过命令：

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction 1 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .25 --source-support-switch \
  --source-contact-events --source-step-count 1 \
  --source-step-profile acc_data/walk1_full_stride_profile_20261004.json \
  --source-swing-pose-priority --balance-body-priority --separate-balance-priority \
  --duration 16 --output-path acc_data/walk1_swing_body_priority_trial.npz
```

`--source-contact-events --source-step-count 1` 在第一步成功落地和制动后
保持双支撑，用于隔离首步问题；不会伪装成整段完成。
最新优先级、源步、接触调度与原始轨迹验收相关 50 项测试通过。

## 整段 PKL 检查：当前仍未完成完整 tracking（2026-10-04）

对原始 `walk1_g1.pkl` 全部 7839 帧进行了数据和接触序列审计：30 Hz、
261.267 s、622 个接触阶段、340 个单支撑阶段。原始关节角未越 URDF 限位，
四元数有效，带接触标签的副本与原始运动逐值一致；这些检查不证明动力学可行。

先前全片审计定位到以下问题（接触事件调度与尾段参考现已补上代码及单元测试，尚未完成整段闭环验证）：

- 第 433 帧首次出现同一侧连续摆动，不能继续无条件左右交替。
- 第 2498 帧首次出现单支撑直接切换，没有现有规划器要求的双支撑落地段。
- 最后 7822–7838 帧为双支撑尾段，仍需要对应的全身收尾参考。
- 从五步实测停止状态完整接入第 198 帧原始落点时，受关节限制的几何求解失败。
- 第六步追近已通过离线转移、摆动和停止检查；未进行闭环验证。
  第七步名义接续在 0.66 m 质心高度几何求解失败，改为 0.64 m 后仍违反
  左踝制动约束，没有执行这些失败候选。

已有五步记录只到 176 帧，身体位置峰值误差 0.964 m、关节 RMSE 峰值
0.313 rad，超过整段验收的 0.2 m / 0.2 rad 阈值。
因此即使继续追加动作，已有记录也不能成为通过原始 PKL 保真度验收的完整轨迹；
后续需要重规划前段参考并重新执行全程。

新增 `--source-recovery-step-count`，默认 0 保留无限追近行为；指定正偶数后，
追近窗口结束会恢复源帧选择。例如 `--source-recovery-after-step 3
--source-recovery-step-count 2` 在两次追近后尝试下一源步。
总步数上限还需大于 5 才会尝试接续。此参数仅解除永久锁帧，
不会绕过下一步可行性检查，也不是已验证的完整 tracking 开关。
新增窗口和整段审计相关 23 项测试通过；本轮没有新增闭环通过范围。

复现全片审计：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  examples/audit_full_source_clip.py \
  --output acc_data/walk1_full_source_audit_20261004.json
```

结构审计覆盖每一帧，但 `physical_feasibility_checked=false`、
`full_tracking_passed=false`。完整状态及失败候选摘要保存在
`acc_data/walk1_full_tracking_status_20261004.json`。

## 已打通追近后的换腿，并完成第五次落地（2026-10-04）

在第四步停止后加入平滑的骨盆局部侧倾 -0.1 rad（约 -5.7°），解除
此前转向左腿承重时的左踝制动约束冲突，再让右脚向原始源帧 176 追近。
Isaac 闭环已完成 **右 → 左 → 右 → 左 → 右** 五次实测落地，
`recovery_passed=true`，105 项相关回归通过。
本次是 **3 个适配源步 + 2 个固定相位追近步**，源相位仍为 176。

| 相对原始第 176 帧的水平位置误差 | 两次追近前 | 两次追近后 |
| --- | --- | --- |
| 左脚 | 0.711 m | 0.301 m |
| 右脚 | 1.041 m | 0.629 m |
| 双脚均方根 | 0.892 m | 0.493 m |
| 身体根位置 | 0.766 m | 0.455 m |

此前失败的左腿承重转移在 34.670 s 启动，36.718 s 实测门控释放右脚。
实测骨盆局部侧倾变化为 -0.1083 rad；抬脚保持的最后 0.5 s，骨盆姿态
相对命令最大误差为 0.00227 rad，左脚最小载荷 323.11 N，右脚载荷为零。
第五步在 39.718 s 启动、41.578 s 确认右脚落地，适配落点误差 7.65 mm。
停止后双支撑保持 5.878 s，最后一秒左右脚最小载荷为 80.29 / 231.46 N，
最大质心误差 1.132 mm，最大质心速度 0.000603 m/s。

侧倾使用连续姿态及一致的角速度、角加速度参考，并在足部/质心雅可比
求解中扣除骨盆旋转的贡献。保留原有关节、力矩和实测接触门控要求。
1502 个转移参考样本的关节检查通过，终端最小腿部关节余量 0.05854 rad。
另行检查的 1501 个转移/抬脚动力学样本均通过接触力分配与扭矩限制，
单支撑时离地脚接触力显式固定为零，峰值扭矩约为电机限额的 27.2%。
第五步摆动和停止参考也分别通过密集约束检查。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .5 --source-support-switch \
  --source-step-count 5 --source-recovery-after-step 3 \
  --recovery-left-transfer-roll-offset -.1 \
  --duration 48 --output-path acc_data/walk1_recovery_pair_trial.npz

PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  examples/analyze_source_recovery.py acc_data/walk1_recovery_pair_trial.npz \
  --recovery-steps 2
```

新增 `--recovery-left-transfer-roll-offset` 默认 0；示例的 -0.1 只在追近步
结束、接下来要转向左腿支撑时启用。离线复现从第四步实际停止状态接续：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/plan_source_continuation.py \
  --recording acc_data/walk1_fixed_frame_recovery_live_20261003.npz \
  --completed-steps 4 --source-end-frame 176 --fixed-source-frame 176 \
  --transfer-pelvis-roll-offset -.1 --landing-com-fraction .4 \
  --output acc_data/walk1_recovery_right_plan.npz
```

**尚未追上原始第 176 帧，也未实现连续行走或整段精确 tracking。**
当前原始左右脚仍有约 0.30 / 0.63 m 水平偏差，身体根位置偏差约 0.45 m。
源相位没有虚增，原始目标保持不变。五步之外的接续尚未闭环验证。
运行器因源覆盖和原始保真度不通过仍返回 1；独立恢复验收返回 0。
仿真控制/力矩更新为 500 Hz，QP 墙钟中位约 4.95 ms，未满足硬件 2 ms 周期。

记录为 `acc_data/walk1_recovery_pair_live_20261004.npz`，对应
`.recovery.json`、`.roll_switch.json`、`.switch_4_plan.json`、
`.step_5_plan.json`、`.step_5_plan.stop.json` 保存验收与规划证据；
`acc_data/walk1_tracking_progress_20261004.json` 保存本轮进度。

## 已加入固定源帧的追近步（2026-10-03）

已在 Isaac 闭环完成 **右 → 左 → 右 → 左** 四次实测落地。
前三步推进到源帧 176；第四步锁定该帧，将左脚向原始 PKL 的左脚位置
追近，水平计划步长上限 42 cm。`recovery_passed=true`，103 项相关回归通过。
本次是 **3 个适配源步 + 1 个追近步**，源覆盖范围没有增加。

| 相对原始第 176 帧的水平位置误差 | 追近前 | 追近后 |
| --- | --- | --- |
| 左脚 | 0.711 m | 0.300 m |
| 右脚 | 1.041 m | 1.042 m |
| 双脚均方根 | 0.892 m | 0.767 m |
| 身体根位置 | 0.766 m | 0.675 m |

误差使用原始参考目标计算：追近前取摆动启动前 0.2 s，追近后取最后 1 s。
第四步在 32.268 s 启动，34.122 s 确认左脚落地；相对适配落点的误差为
8.67 mm。停止后保持双支撑 5.328 s，最后一秒左右脚最小载荷为
230.83 / 82.07 N，最大质心误差 1.144 mm，最大质心速度 0.000492 m/s。
原始 PKL、目标世界位置及关节/扭矩/接触约束均保留；追近时保存独立的
`history_source_recovery`、原始双脚目标和原始根位置，防止将相位推进或
目标移动误认为跟踪改善。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .5 --source-support-switch \
  --source-step-count 4 --source-recovery-after-step 3 \
  --duration 40 --output-path acc_data/walk1_recovery_trial.npz

PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  examples/analyze_source_recovery.py acc_data/walk1_recovery_trial.npz
```

`--source-step-count` 包含追近步；`--source-recovery-after-step 0` 为默认，
不启用追近。`--recovery-maximum-step-distance` 默认 0.42 m，
`--recovery-landing-com-fraction` 默认 0.4。后续接续仍需通过实测接触门控
和每一步的规划约束检查，不能直接把步数调大视为已验证。

**仍未完成连续行走或整段 PKL 精确 tracking。** 原始目标尚未追上，
尤其右脚仍有约 1.04 m 偏差。当前身体姿态下，追近步之后转向左腿承重的
基于本次实测停止状态的离线候选仍触发左踝关节制动约束；此前基于名义停止状态
改变质心高度或延长转移时间的已测候选
也未通过。没有放宽限制或强行释放右脚。
运行器因整段覆盖/保真度不通过仍返回 1；独立恢复验收脚本返回 0。
控制与力矩在仿真时间内按 500 Hz 更新，QP 墙钟中位约 4.90 ms，
尚未达到 2 ms 硬件实时周期。

证据：`acc_data/walk1_fixed_frame_recovery_live_20261003.npz` 和对应
`.recovery.json`、`.switch_3_plan.json`、`.step_4_plan.json`、
`.step_4_plan.stop.json`。进度汇总为
`acc_data/walk1_tracking_progress_20261003.json`。

## 已续接第三步，到达源帧 176（2026-10-03）

从第二步停止状态继续向左腿转移承重，再执行右脚摆动、落地和减速。
本轮闭环确认了 **右 → 左 → 右** 三次落地，独立报告
`sequence_passed=true`；源时钟全程单调，98 项相关回归通过。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .5 --source-support-switch --source-step-count 3 \
  --duration 32 --output-path acc_data/walk1_three_steps_trial.npz

PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  examples/analyze_source_sequence.py acc_data/walk1_three_steps_trial.npz \
  --steps 3 --end-frame 176
```

| 指标 | 实测结果 |
| --- | --- |
| 第二次承重转移开始 | 19.770 s，转向左腿 |
| 右脚释放时刻 | 21.818 s |
| 第三步启动 | 24.818 s，参考区间 155–176 |
| 第三次右脚确认落地 | 26.674 s |
| 第三步适配落点误差 | 8.62 mm |
| 第三步右脚最大脚底净空 | 2.99 cm |
| 停止后双支撑保持 | 4.778 s |
| 最后一秒左右脚最小载荷 | 59.13 / 253.14 N |
| 最后一秒最大质心误差 | 1.138 mm |
| 最后一秒最大质心速度 | 0.000718 m/s |

后续换腿使用新的参数组合：质心降低 2 cm、抬脚 3 cm；
1502 个参考样本通过关节制动检查，终端最小腿部关节余量约 0.0607 rad。
第三步保持 2 s 摆动、0.66 m 落地质心高度和分布式上身姿态续接，
落点比例设为 0.3，停止时间为 0.4 s。摆动及减速的密集动力学检查均通过。
没有放宽关节、扭矩、摩擦或实测接触条件。

参数 `--source-step-count` 现在允许正整数；每次新转移和摆动都必须重新
通过规划检查，并等待实测接触门控。闭环目前只验证了 3 步，更多步数
尚未验证，也不保证现有参数在后续区间可行。
新增可配置项：`--later-transfer-height-offset`、
`--later-transfer-lift-height`、`--later-step-landing-fraction`、
`--later-step-stop-duration`，分别默认 -0.02、0.03、0.3、0.4。
已有 1 步和 2 步运行的默认行为保持不变。

**原始动作的精确整段跟踪仍未完成。** 第三步计划落点相对原始落点缩短
约 1.033 m，实测原始落点误差约 1.041 m，存在累计位置偏差。
当前仍在步间执行停止和承重转移，`continuous_walking=false`、
`full_pkl_tracking=false`。`motion_tracking_valid=true` 只表示控制验收通过；
整段覆盖和原始轨迹保真度未通过，因此运行器仍返回退出码 1。
仿真步长 2 ms，QP 墙钟中位约 4.90 ms，未证明硬件实时性。

证据：`acc_data/walk1_three_steps_live_20261003.npz`、同名前缀
`.sequence.json`、`.switch_2_plan.json`、`.step_3_plan.json` 和
`.step_3_plan.stop.json`。每次换腿的规划报告独立保存。

离线复现第三步及其前置转移：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/plan_source_continuation.py \
  --recording acc_data/walk1_two_steps_clock_20261003.npz \
  --output acc_data/walk1_third_step_plan.npz
```

该入口先验收停止记录，再规划转移、摆动及停止；不把离线通过当成闭环验证。
以下保留此前阶段记录。

## 已续接第二步，到达源帧 155（2026-10-03）

`walk1_g1.pkl` 的跟踪从此前第 129 帧继续推进至第 155 帧：
右脚落地、转移至右腿支撑后，执行左脚摆动、实测落地、减速及双支撑保持。
使用 `walk1_g1_with_contacts.pkl` 提供接触标签；原始动作文件未修改。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .5 --source-support-switch --source-step-count 2 \
  --duration 24 --output-path acc_data/walk1_two_steps_trial.npz

PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  examples/analyze_source_sequence.py acc_data/walk1_two_steps_trial.npz
```

复跑证据：`acc_data/walk1_two_steps_clock_20261003.npz` 和同名前缀 `.sequence.json`。
独立验收 `two_steps_passed=true`，96 项相关测试通过。

| 指标 | 实测结果 |
| --- | --- |
| 右脚首次确认落地 | 11.662 s，适配落点误差 5.39 mm |
| 第二步启动 | 17.268 s，从右腿支撑续接左脚摆动 |
| 左脚确认落地 | 19.124 s，适配落点误差 8.37 mm |
| 左脚摆动最大脚底净空 | 3.87 cm |
| 停止后的双支撑保持 | 4.228 s |
| 最后一秒左右脚最小载荷 | 282.21 / 28.28 N |
| 最后一秒最大质心误差 | 1.152 mm |
| 最后一秒最大质心速度 | 0.001203 m/s |
| 最终源帧 / 源时钟 | 155 / 全程单调不回退 |

新增 `--source-step-count 2`，默认值 1 保留单步实验。
第二步读取接触序列中的下一段右支撑区间（137–155），将 129–155
参考区间续接到已经抬起左脚的当前状态；接触时序因此有适配。
第二步使用 2 s 摆动、0.66 m 落地质心高度、0.85 摆动相位混合、
1 cm 额外抬升及 0.5 s 落地减速。落点比例默认 0.4，可用
`--following-step-landing-fraction` 显式指定；质心高度对应
`--following-step-landing-height`。运行前仍须通过全部轨迹预检查。

关键修复：原先只把源区间首帧替换成当前姿态，会把躯干/手臂变化
集中在摆动早期；新的 `body_join=distributed` 将姿态变化逐渐引入
整个区间。第一步继续使用原有方式。另修复首步时钟接管产生的
114.036→114 帧回退：承重转移从起步边界接管时钟，然后交给摆动参考。
未放宽关节、扭矩、接触或验收约束。

**这仍是两步适配参考的验证，整段原始 PKL 跟踪未完成。**
原始落点误差第一步约 25.59 cm、第二步约 68.44 cm；
第二步计划落点相对原始落点缩短约 67.60 cm。
`full_pkl_tracking=false`、`continuous_walking=false`，第三步尚未接入。
本轮聚合控制验收 `motion_tracking_valid=true`，独立两步验收也通过；
运行器因整段源动作尚未完成仍返回退出码 1，不能据此宣称整段任务完成。
该次记录末尾旧日志把整段失败也显示成控制失败；已修正这一显示，
以 NPZ 中的控制标志及独立验收报告区分两者，退出条件保持不变。
仿真更新间隔 2 ms；QP 墙钟中位约 4.11 ms，未证明硬件实时性。

离线复现第二步规划（输出包括摆动和停止参考）：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/plan_next_source_step.py \
  --recording acc_data/walk1_switch_load_priority_20261003.npz \
  --output acc_data/walk1_next_step_plan.npz
```

实现入口：`examples/test_g1.py`；相位选择与交接位于
`src/fabrics_sim/reference/source_sequence.py`，姿态续接位于
`src/fabrics_sim/reference/source_step.py`。以下保留此前阶段记录。

## 单次左右支撑切换已验证（2026-10-03）

新增 `--source-support-switch`：在首个适配半步落地并完成减速后，
从左腿支撑经双支撑转移至右腿支撑，再抬起左脚保持。
它依据实际载荷、质心速度/误差及腿部跟踪误差决定释放时刻；
条件必须连续满足 50 ms。新脚至少承重 100 N，旧脚载荷不超过总载荷的 10%。
释放后若新支撑脚失去接触或载荷低于 20 N，立即中止。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .5 --source-support-switch \
  --duration 20 --output-path acc_data/walk1_switch_trial.npz

PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  examples/analyze_support_switch.py acc_data/walk1_switch_trial.npz
```

2026-10-03 实测结果（`walk1_switch_load_priority_20261003.npz`）：

| 指标 | 结果 |
| --- | --- |
| 左脚释放时刻 | 14.268 s |
| 释放后的仿真持续时间 | 5.730 s |
| 最后一秒左脚最小脚底净空 | 3.827 cm |
| 最后一秒左脚最大载荷 | 0 N |
| 最后一秒右脚最小载荷 | 319.609 N |
| 最后一秒最大质心误差 | 1.124 mm |
| 独立单次切换验收 | `support_switch_passed=true` |
| 相关回归 | 89 项通过 |

实现：`src/fabrics_sim/reference/support_switch.py`，在
`examples/test_g1.py` 续接参考、接触模式、接触力目标及任务优先级。
新切换段使用质心优先、腿部姿态其次的顺序，并持续启用载荷分配目标。
转移目标质心比停止段高 1 cm；1502 个轨迹样本均通过关节边界预检查，
终端腿部关节最小余量约 0.0597 rad。未放宽关节或接触约束。

**范围：只验证了一次左支撑到右支撑，以及之后的抬脚保持。**
门控逻辑两侧均有单元测试，但反向切换尚未做物理仿真验证。
源帧仍为 129/7838，首次落点相对原动作缩短约 25 cm；
`full_pkl_tracking=false`、`continuous_walking=false`。
运行器仍按整段任务判定退出码 1，聚合 `motion_tracking_valid` 未通过；
这不是完整 tracking 成功。独立报告只检验本次切换、QP/伺服标志和保持指标。
仿真控制周期为 2 ms，但 QP 墙钟中位耗时约 3.94 ms，未达到硬件实时部署验收。

证据：`acc_data/walk1_switch_load_priority_20261003.switch.json`、
同名前缀 `.npz` 和 `.switch_plan.json`。下文保留此前阶段记录。

## 落地后减速续接已执行（2026-10-03，本轮）

现在可在已确认的首个半步落地后，继续执行 **0.5 s 双支撑减速参考**，
不再把仍有速度的末帧直接冻结。源相位仍停在 129；连续换脚和整段 PKL
跟踪尚未完成。首次落点仍比原始动作缩短约 25 cm。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --source-step-stop-duration .5 \
  --duration 16 --output-path acc_data/walk1_landing_stop_trial.npz
```

本轮物理仿真执行至 12.218 s：

| 指标 | 实测结果 |
| --- | --- |
| 减速段 | 11.718–12.218 s，确认触地后执行 |
| 末端三维 COM 速度 | 0.00393 m/s |
| 末端 COM 位置误差 | 1.92 mm |
| 左/右脚法向载荷 | 94.57 / 230.56 N |
| 右脚承重比例 | 约 71%，仍是双支撑，尚未抬左脚 |
| 适配落点误差 | 5.40 mm；原始落点误差仍约 25.59 cm |

程序在减速终点因下一次承重转移/抬脚参考尚未规划而停止，退出码为 1。
减速终点检查通过，不等于后续稳定保持或连续行走通过。新的错误信息为
`Landing reference stopped but the next load-transfer/lift trajectory is not planned`。
报告 `landing_stop_endpoint_passed=true`，`post_stop_hold_validated=false`，
`full_pkl_tracking=false`。本轮新增与相关回归共 83 项通过。

实现位于 `src/fabrics_sim/reference/double_support_stop.py`：分段常值 jerk
积分保留起始位置、速度、加速度，末端速度和加速度为零；压力中心规划
保留 6 mm 内部余量。之后在 2 ms 网格检查全身逆动力学、双足力矩约束、
扭矩、速度、加速度及原有关节制动条件。251 个样本全部通过，峰值扭矩
约为限值的 38.9%。进入减速段必须已经确认实测落地；支撑判断允许
已确认的新脚接管载荷，同时拒绝双脚均失去实测支撑。

接触搜索留下的足端偏置仍是控制参考的虚拟目标；执行时实际接触约束
来自仿真观测。离线必要条件检查不能单独证明实际碰撞或闭环稳定性。
本轮闭环记录单独验证了上述 0.5 s 减速段；没有新增硬件实时性结论。

复现离线规划：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/plan_double_support_stop.py \
  --recording acc_data/walk1_bridge_contact_activation_20261003.reference_failure.npz \
  --duration .5 --pressure-margin .006 \
  --output acc_data/walk1_stop_candidate_trial.npz
```

主要证据：

- `acc_data/walk1_landing_stop_audited_20261003.json`：包含速度限制的独立密集审计。
- `acc_data/walk1_landing_stop_live_20261003.reference_failure.npz`：完整起步、摆动、触地和减速记录。
- 同名前缀 `.reference_failure.step.json`：实测终点与载荷指标。

下一次左脚抬起的候选仍触发踝关节制动/余量检查，未放宽约束或直接
切回原始 PKL。相关探索保存在 `acc_data/walk1_next_transfer_*_20261003.json`；
未通过的抬脚姿态改动没有接入运行入口。

## 首个缩短过渡步已确认触地（2026-10-03，本轮）

现在已越过此前“第一步规划不通过”的阻塞，并在物理仿真中执行到源相位
129。首次接入使用半步：足端水平位移约 25 cm，比源落点缩短约 25 cm；
因此这是适配后的过渡步，不是原始 PKL 落点的精确复现。
**连续换脚与完整 7839 帧跟踪仍未完成。** 下文较早的小节保留历史失败结果。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --source-step-landing-fraction .5 --source-step-landing-search-depth .004 \
  --duration 15 --output-path acc_data/walk1_bridge_trial.npz
```

可加 `--show-isaac-window` 查看仿真。当前预期在第一步末端停止并提示
`First landing reached but the next load-transfer trajectory is not planned`。
这条错误准确表示后续参考缺失，不表示整段跟踪通过。

本轮对照结果：

| 实验 | 结果 |
| --- | --- |
| 半步、无接触搜索 | 751 个 2 ms 样本通过关节制动、扭矩、接触力检查；执行到帧 129，但足底仍悬空约 1.18 mm，无载荷 |
| 半步、末段 4 mm 接触搜索、延迟激活约束 | 实际碰地一次后反弹，未确认落地 |
| 半步、接触搜索、即时激活实测接触约束 | 11.662 s 确认右脚触地；保持至记录末尾 11.718 s，随后因后续规划缺失停止 |

最后一项末帧左/右脚载荷为 260.95/50.98 N，适配落点误差 5.40 mm，
COM 误差 2.33 mm。原始 PKL 落点误差仍为 25.59 cm，不能用适配目标
误差代替原动作精度。触地后的记录只有约 56 ms，尚未验证后续稳定承重转移。
5860 次 QP 与扭矩更新对应 500 Hz 仿真步；QP 实际耗时中位数约 3.77 ms，
不具备 2 ms 实时性验证结果。

本轮实现要点：

- `landing_fraction` 显式控制首次足端水平位移，默认 1 保留原落点；记录同时保存源落点和适配落点。
- `landing_search_depth` 默认 0；启用后在最后 0.2 s 用连续位置/速度/加速度的曲线请求向下接触，深度上限 1 cm。这是虚拟目标，不是允许实体穿过地面。
- 首次实测平脚接触就激活双支撑约束；连续三次载荷确认仍用于认定落地。一次冲击不会被认定为稳定落地。
- 关节制动、扭矩、摩擦和力矩约束保持生效。78 项相关回归通过。

主要证据：

- `acc_data/walk1_bridge_half_20261003.dynamics.json`：独立逆动力学审计。
- `acc_data/walk1_bridge_contact_activation_20261003.step_plan.json`：带接触搜索的密集预检。
- `acc_data/walk1_bridge_contact_activation_20261003.reference_failure.npz`：实际执行记录，包含停止前的已确认触地。
- `acc_data/walk1_bridge_contact_activation_20261003.reference_failure.step.json`：实测指标；`measured_touchdown_confirmed=true`，完整跟踪及连续行走标志仍为 false。

后续必须规划带非零质心速度的落地过渡，再完成右脚承重与左脚摆动。
目前尝试的双支撑减速候选尚未通过约束检查，不能直接冻结末帧或切回
原 PKL 来冒充连续跟踪。

## 协调起步入口（2026-10-03，本轮进展）

新增 `--coordinated-source-start`，把已验证的承重转移接入普通跟踪入口：
从停稳参考出发，用实际支撑脚位置协调全身参考；确认承重后释放右脚，
再规划源帧 114–129 的第一步。当前还没有完成第一步落地或整段 PKL 跟踪。

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --coordinated-source-start --transfer-balance-priority \
  --duration 15 --output-path acc_data/walk1_coordinated_trial.npz
```

该实验目前预期以规划拒绝结束，不能作为已成功行走的启动命令。
`walk1_coordinated_start_20261003` 实测在 7.218 s 释放右脚，之后保持
左脚支撑，到约 10.218 s 进行第一步规划。751 个 2 ms 采样点中仍有
关节制动约束违反，最大归一化违反量 0.064954，因此没有执行摆动候选。
峰值扭矩比 0.3395、加速度比 0.3776、接触力矩检查通过都不足以放行。
报告见 `acc_data/walk1_coordinated_start_20261003.step_plan.json`；
失败前仿真记录为同名前缀的 `.reference_failure.npz`。

独立 12 s 承重转移回归通过：离地后保持 4.78 s，末秒右脚最小净空
3.872 cm、接触力为零；左脚最小支撑力 320.07 N，最大 COM 误差
0.918 mm。证据为 `acc_data/walk1_transfer_contacts_20261003.transfer.json`。
这只验证承重转移和抬脚，不验证向前落步。

离线规划现在支持 `--terminal-capture double_support`，末端捕获点检查
使用两脚足底的真实投影凸包；双支撑内可捕获并不保证下一次换脚可行。
同时可以显式设置 `--landing-com-fraction`、`--pelvis-pitch-offset` 和
`--body-interpolation cubic`。骨盆偏置属于参考适配，会偏离原始运动；
原 PKL 不被修改。默认插值仍为 PCHIP，cubic 虽然二阶连续，但可能超调，
所有候选仍经过原有动力学、关节制动检查。

复现本轮二阶连续插值候选：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/plan_unified_source_step.py \
  --recording acc_data/walk1_transfer_contacts_20261003.npz \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --duration 1.5 --landing-height .61 --swing-phase-blend .7 \
  --body-smoothing-passes 4 --body-interpolation cubic \
  --landing-com-fraction .45 --terminal-capture double_support \
  --pelvis-pitch-offset -.4 --full-dynamics --momentum-iterations 10 \
  --output acc_data/walk1_cubic_candidate_trial.npz
```

该候选仍未通过关节制动约束。后续需要让踝关节的接近限位速度与加速度
满足制动条件，并补齐落地后的承重转移和逐步续接；仅改插值或提供
`contact_sequence` 不能完成这两项工作。

## 使用 PKL 接触序列（2026-10-03）

`--reference-manager` 现在优先使用 PKL 的 `contact_sequence[T,2]`
（左脚、右脚），不再用几何推断覆盖已提供的标签。未提供序列时沿用
原推断路径。管理器不支持腾空；提供双脚均不接触的帧会明确报错。
实测承重、平衡与落地确认仍然生效，文件标签不直接授权释放支撑脚。
`.contact_plan.npz` 的 `method=pkl_contact_sequence` 表示采用文件标签；
`geometry_method` 和 `ambiguous` 仍记录几何推断方法及其歧义。

跟踪含推断接触标签的 walk1 副本：

```bash
bash examples/run_lafan_tracking.sh \
  --motion-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --output-path acc_data/walk1_with_contacts_trial.npz
```

添加 `--show-isaac-window` 可显示仿真。副本只增加接触相关字段，原始
运动数据不变。接触标签来自足底运动推断，不代表实测接触或动力学可行性；
本入口仍是完整跟踪实验，不能将接触标签接入成功视为行走成功。

本次已逐帧核验 7839 个接触模式与文件标签完全相同，32 项接触管理回归
通过。完整跟踪尝试仍在 8.114 s 被 QP 校验拒绝，源帧停在 112.50024，
全程命令双支撑；峰值关节 RMSE 0.4649 rad、根位置误差 0.4618 m。
接触序列接入修复没有解决首次承重转移和完整步态规划问题。
结果：`acc_data/walk1_with_contacts_20261003.tracking.json`。

## 统一全身参考（2026-09-28）

`run_lafan_tracking.sh` 现在默认使用 `--reference-mode unified`。
关节、足端、手部、骨盆、躯干、质心、角动量及其导数统一由同一份
`WholeBodyReference(q, v, a)` 生成。动作时间仍由 ReferenceManager 管理。
这项修改只保证参考一致性，不等于原 PKL 动力学可行或全段跟踪成功。

旧的独立屈膝、摆脚桥接、质心预瞄、手部平移实验保留在
`run_lafan_tracking_legacy.sh`，它显式指定 `--reference-mode legacy-managed`。
下文旧实验命令若使用这些修正，需要补上该参数。
详细说明见 [统一参考说明](../docs/g1_unified_reference.md)。

## 500 Hz QP / 500 Hz 扭矩（2026-09-27，正在验证）

**当前尚未完成全段原始 PKL 跟踪。** 本节按实验顺序记录证据；局部换脚、
QP 可行或仿真 2 ms 更新均不等于全段精度或实时性通过。

### 最新回归：接触过渡与滚动预瞄

`--landing-speed-limit` 读取实际 USD 的足部碰撞几何（每脚一个盒子和五个球，
共 13 个顶点/球心），对临近地面的摆脚表面施加法向速度制动不等式。
球面高度按世界球心高度减半径计算，不能把半径当成随足部旋转的固定点。
这是局部制动条件，不保证冲击后的状态可行。0.05 m/s 对照延迟了落地，
反而导致失去旧脚支撑；没有将其设为默认开关。

连续源时钟模式下，管理器现在允许已确认的新脚接管实测支撑：必须已经
完成落地稳定、承重确认及当前预瞄检查，且实际单脚接触恰好是计划的新脚。
它只解除名义源帧的起步等待，不跳源帧，也不绕过物理承重/可行性检查。
实测接触筛选也覆盖 USD 的 x=0.14 m 脚尖球面，避免丢掉约 x=0.1407 m
的真实加载点。上述几何、关节制动、管理器与接触回归共 96 项通过。

物理对照均未完成：

| 记录前缀（均在 `acc_data/`） | 结果 |
| --- | --- |
| `lafan_centroidal_timing_500hz_20260927` | 最短可行的 0.5 s 首步在约 5.594 s 首次落地后触发踝关节限位 |
| `lafan_centroidal_timing65_500hz_20260927` | 最短步长时间 0.65 s，6.718 s / 源帧 155 QP 失败，根姿态已明显失稳 |
| `lafan_landing_speed_500hz_20260927` | 法向落地速度 0.05 m/s，5.948 s / 源帧 131.55 失去旧脚支撑 |
| `lafan_flat_contact_transition_500hz_20260927` | 平脚摆动及新管理器，6.002 s / 源帧 133.02 失去旧脚支撑；当时预瞄不可行，未绕过检查 |
| `lafan_observed_takeoff_500hz_20260927` | 源摆脚、0.15 m/s 落地速度、新管理器，约 6.49 s / 源帧 151.58 左踝俯仰限位 |

最后一项峰值关节 RMSE 0.405 rad、根位置误差 0.280 m、根姿态误差
0.549 rad，均未达标；3245 次 QP/扭矩一一对应 2 ms 仿真步。
QP 中位耗时 2.82 ms、P99 4.75 ms，未满足墙钟实时 500 Hz。
审计文件为同前缀 `.reference_failure.audit.json`。

新增可选 `--centroidal-replan-period .05`：单支撑阶段根据实测 COM、速度及
角动量重规划剩余轨迹，保留当前源时钟、原落脚点和落地截止时间。
最后 0.15 s 不再重规划。候选仍须通过原动力学、压力中心、摩擦、加速度、
角动量及求解证书；失败时保留旧计划并记录原因，不能称为新可行性证书。
日志新增 `centroidal_replan_attempted/accepted/reason`。48 项源时钟、
预瞄、管理器与落地测试通过，包括扰动后从实测状态开始、截止时间不变。
这仍是简化质心模型，不含未来全身关节和扭矩可行性保证。

滚动预瞄物理回归 `lafan_centroidal_replan_500hz_20260927` 未改善结果：
10 次重规划仅前 5 次通过，5.356 s 起剩余时域不可行，5.772 s / 源帧 129
右踝横滚限位。关节 RMSE 峰值 0.266 rad、根位置误差 0.201 m。
该开关保持实验性，不默认启用。综合接触、关节、预瞄及任务分层回归
125 项通过；这不改变物理跟踪失败的结论。

为定位控制目标冲突，`lafan_control_attribution_500hz_20260927` 在 5.2、
5.5、5.9、6.0、6.2、6.5 s 保存完整 QP。物理运行于 6.718 s / 源帧 155
失败。离线保留全部原物理约束及后续任务，比较不同首层：5.2 s 的原加权
质心 Y 加速度残差为 +1.492 m/s²；先优化 COM 与身体/骨盆朝向时约为
+0.02 m/s²，身体朝向加速度残差也约为 0.02 rad/s²，代价是摆脚和关节
加速度残差略增。5.5 s 同一纠偏则会产生约 49 m/s²/rad/s² 的足端混合
加速度残差 RMS，说明晚期强行满足平衡请求有严重代价。数据为
`acc_data/g1_pre_failure_task_tradeoffs_20260927.json`，仅证明快照级可行性。

新增实验开关 `--balance-body-priority`（要求 `--separate-balance-priority`
及正的源骨盆权重）：首层联合 COM 和身体/骨盆朝向；源角动量、脚和关节
姿态进入后层。物理约束仍先于全部任务，不能靠此开关绕过扭矩限幅。
13 项分层 QP 测试通过，包括相冲突的源角动量/姿态请求不能覆盖已取得
的身体朝向结果、且不可达朝向仍须服从电机限幅。物理结果另行审计。

该身体优先对照 `lafan_balance_body_500hz_20260927` 未通过：5.934 s / 源帧
129 停止，摆脚未完成落地，峰值关节 RMSE 0.751 rad；实测水平角动量
达到约 9.04 kg·m²/s，超出简化预瞄假设的 4 kg·m²/s。

新增实验 `--centroidal-momentum-bounds`：预瞄生效时，执行 QP 同时约束
`Hdot = A*qdd + bias` 的 XY 分量及 `H + dt*Hdot`，边界直接取当前预瞄
的 30 N·m 和 4 kg·m²/s。这是一步预测契约，不是未来全身可行性或冲击
保证；实测越界不会被剪裁。`lafan_momentum_budget_500hz_20260928` 在
5.652 s / 源帧 127.29 遇到第二层数值证书拒绝；关节 RMSE 峰值 0.304 rad，
仍未通过跟踪验收。

同一失败 QP 的独立复核确认数值问题：较大乘子将约 1e−8 的活动约束
舍入残差放大，互补性校验失败。DAQP 后端现在可按原生乘子选择活动行，
对行归一化并在分解中去掉线性依赖，再精化解。接受时仍检查全部原始
约束、乘子符号、驻点与互补性，未放宽容差。原失败第二层残差降至
5.68e−14，原最终层也通过同一证书，见
`acc_data/g1_momentum_budget_qp_polish_20260928.json`。32 项相关回归通过。

另做了参考几何一致性检查：在记录的 5.2/5.5/6.0/6.2 s，用原骨盆朝向、
实际支撑脚和规划的摆脚/COM 目标求解参考 IK，所需关节 RMSE 改变量为
0.043/0.087/0.126/0.129 rad，最大单关节改变量达 0.509 rad。
它们是局部几何解，不是可直接发送的动力学轨迹。对一段 IK 序列直接
重采样 PCHIP 甚至出现 286.5 rad/s²；不能把逐帧几何可行等同于运动可行。
记录：`g1_reference_consistency_20260928.json`、
`g1_reference_path_consistency_20260928.json`。

最终对照 `lafan_posture_secondary_500hz_20260928` 保留 COM/角动量、
身体/骨盆、足端在同一首层，将原始关节姿态放到后层，并包含角动量
边界与数值精化。仍在约 5.884 s / 源帧 130.78 的 `SETTLE` 阶段失去
旧脚支撑；新脚尚未满足稳定计时，下一步预瞄因实测角动量超预算而拒绝。
峰值关节 RMSE 0.514 rad、根位置误差 0.270 m、根姿态误差 0.116 rad。
保住身体朝向没有同时保住腿部姿态和落地，不能视为更好的完整跟踪配置。
2942 次 QP/扭矩按 2 ms 一一更新，QP P99 5.34 ms，仍非实时 500 Hz。

本次最终相关回归共 **145 项通过**，语法检查通过；所有上述物理实验
均未通过完整跟踪验收。实验开关没有写入默认启动脚本。
下一项结构性工作是让脚位、COM、全身关节参考和承重过渡共同满足
跨时间段的运动/动力学可行性，不能再把局部 IK 或质心预瞄证书当成
全身轨迹可行证书。角动量执行边界目前只在预瞄生效阶段应用，承重
过渡及接触冲击后的预算恢复也仍须处理。

简要结果及文件索引见 [`../docs/g1_tracking_status_20260928.md`](../docs/g1_tracking_status_20260928.md)。

早期对照 `lafan_velocity_duration_500hz_20260927` 使用 0.2 s 源摆脚过渡、
脚与姿态共用源时钟（落地提前量为 0）、COM XY 分层，以及比较全部可行
时长的双支撑落地预瞄。运行到 7.162 s / 源帧 150.16 后，左踝俯仰下一步
位置约束不可行。记录显示源脚底仍有 2.11 cm 间隙，实测脚尖已触地、
左脚法向力达 124 N，左踝速度反向并逼近限位；见
`acc_data/g1_second_landing_clearance_20260927.json`。
该次审计峰值关节 RMSE 0.358 rad、根姿态误差 0.533 rad，均未达标。
3581 次 QP/扭矩一一对应 2 ms 仿真步；QP P99 为 5.86 ms，不满足实时要求。
完整审计：`acc_data/lafan_velocity_duration_500hz_20260927.reference_failure.audit.json`。

新增实验开关 `--swing-sole-barrier`：在实测脚已抬起后，对四个脚底角点
加入 5 mm 间隙的加速度不等式，包含旋转向心加速度；距计划落地小于等于
0.1 源秒时释放。与已有碰撞约束合并，保留原接触、扭矩及关节限制。
这只是局部加速度保护，不是全段可行性或抗碰撞冲击保证。角点有限差分、
激活时序及相关源时钟/管理器/加速度约束共 44 项测试通过。物理回归：

```bash
bash examples/run_lafan_tracking.sh \
  --separate-balance-priority --horizontal-balance-priority \
  --source-bridge-duration .2 --swing-touchdown-lead 0 \
  --double-support-preview --swing-sole-barrier \
  --output-path acc_data/lafan_sole_barrier_test.npz
```

以下为本轮之前的实验记录；均不能视为完整跟踪通过。

脚底保护物理对照停在 7.274 s / 源帧 152.54，未重现原左踝限位失败，
但姿态继续恶化（峰值关节 RMSE 0.502 rad、根姿态 0.915 rad）。失败快照
的独立 LP 确认原物理约束、前两层约束均可行；最终层不可行源于第二层
失败结果生成的任务锁，不能把它解释为原物理约束不可行。
DAQP 返回成功标志但未通过独立证书时，现也触发冷启动近端重试。
不改变任何约束或接受容差。同一第二层快照的最大约束误差降至
3.02e−11，保留第一层锁并重建第二层锁后最终 QP 也通过（1.24e−10）。
见 `g1_native_success_retry_20260927.json` 与
`g1_native_success_retry_tracking_20260927.json`。20 项相关测试通过。
此数值修复尚不解决身体下沉或完整跟踪误差；完整 COM 分层物理对照
`lafan_sole_full_com_500hz_20260927` 在该修复前启动，不含新的重试逻辑。
该完整 COM 对照随后因失去旧脚支撑停止，6.0 s 关节 RMSE 为 0.397 rad。

新增实验 `--source-com-path`（要求双支撑预瞄、落地提前量 0）：
在每个候选时长中，用与全身相同的单调源时钟，加入 9 个中间 COM XY
位置的软代价，并以该时钟的实际终端速率计算源落地速度。支撑压力区域、
终端捕获区域、动力学方程和终端位置约束保持原值。COM 源位置由逐帧
正运动学后进行 PCHIP 插值；中间代价是拟合偏好，不是连续误差保证。
11 项预瞄/源时钟测试通过，包括独立动力学与边界校验、非平凡路径拟合
误差下降。此开关的物理结果仍需验证，不自动启用。

路径拟合物理对照 `lafan_source_com_path_v2_500hz_20260927` 的第二步时长
从 1.50 s 降至 1.19 s，但约 7.036 s / 源帧 153.83 仍因左踝触地后
逼近限位而停止；峰值关节 RMSE 0.317 rad、根姿态误差 0.646 rad，未通过。
首轮曾触发批量积分结果不支持多时刻查询的接口错误，现已修复，测试
覆盖整组候选时长及多时刻查询。

随后确认规划/跟踪的落地姿态不一致：源摆脚保留原足部 roll/pitch，
而旧高度 IK 使用完全放平的落地脚。源摆脚模式现在统一使用
`source_contact_step`：保留世界 XY 和四元数，仅在原 4 cm 范围内将
最低脚底角点对齐地面；非源摆脚模式保留其平脚目标。倾斜脚底的压力
约束也改用投影矩阵的逆及真实脚底偏移，代替仅适用于水平脚的转置。
同一首步几何对照：COM 高度 0.64465 → 0.66075 m（源 0.66415 m），
关节 RMSE 0.1316 → 0.0725 rad。见 `g1_source_contact_height_20260927.json`。
第二步的源高度仍未被当前局部 IK 求出，不能声称几何全段可行。
24 项脚步/倾斜足底/预瞄/高度 IK 测试通过。物理回归
`lafan_matched_contact_500hz_20260927` 使用完整 COM 分层，结果待验证。

该对照仍在第一次承重转换失去旧脚支撑：记录中新的右脚一直是角点接触，
管理器持续处于 `SETTLE`，并非已有证据证明下一步物理约束不可行。
现将 `--observed-contact-patches` 模式下的承重转换压力目标限定在实际
加载接触点的凸包内，支持单点、线段和多边形，不再把脚跟角点扩大为
整个脚底。8 项平衡回归通过。新增 `--landing-orientation-priority`，
在承重转换的任务层中加入落地脚实际自由的转动坐标，接触硬约束不变。
`lafan_landing_priority_500hz_20260927` 跨过第二次落地，进入第三次摆脚，
最终在 7.922 s / 源帧 170.96 因腰部俯仰限位停止。峰值关节 RMSE
0.369 rad、根姿态误差 1.189 rad，仍未达标。完整 COM 高度优先的对照
`lafan_landing_full_com_500hz_20260927` 又停在第一次承重转换；保持较高
COM 目标时，新脚仍未完成从脚跟接触到平脚承重。

平脚落地对照 `lafan_flat_arrival_500hz_20260927` 保留原始落脚世界 XY、
yaw 和步幅，使用现有 `LandingTransition`，落地目标为平脚、零速度。
它通过第一次承重转换，但在约 7.0 s / 源帧 155.38 触发左踝滚转限位；
峰值关节 RMSE 0.483 rad、根姿态误差 1.007 rad。该实验改变摆脚路径，
不能称作原始足端轨迹重放；误差审计仍对照原 PKL。

补充 `--whole-body-posture-priority`（与 `--upper-body-priority` 互斥），
可在平衡层之后同时考虑全部关节参考，避免先锁定上身/足端结果后腿部
仅有很少剩余自由度。同一 6.5 s 快照的离线比较中，腿部加速度请求残差
RMS 17.01 → 9.71 rad/s²，但摆脚位置加速度残差变大；这是明确的权衡，
物理对照 `lafan_whole_posture_500hz_20260927` 在第二步中段降低了误差，
但在 7.008 s / 源帧 155 因 QP 不可行停止；失败前峰值关节 RMSE
0.517 rad、根姿态误差 0.914 rad。独立 LP 也判定第一层的原始物理约束
不可行，不能归因于后续任务锁或单一求解器误报。数据：
`acc_data/g1_posture_priority_tradeoff_20260927.json`。
本轮接触规划、源时钟、平衡、几何约束和 QP 分层的组合回归共 103 项通过；
测试通过不代表原始动作的物理跟踪通过。

新增可选 `--centroidal-balance-priority`，将角动量变化率加入 COM 所在任务层。
搭配 `--separate-balance-priority` 时，首层共同优化 COM 与角动量变化率，
第二层再处理姿态/足端，保留各层结果；否则与选中的姿态/足端任务共同优化。
既有物理约束、任务权重和接受容差均不变；在约束限制下两种平衡目标
仍是软目标。平脚落地运行的 5.8 s、6.5 s 快照离线对照中，两者可同时
满足（角动量任务残差约 1e−5，COM 残差约 1e−4）；到 7.0 s 已无法同时
满足，不能把较早的局部可行性视作全程保证。
数据 `acc_data/g1_centroidal_priority_20260927.json`；21 项 QP 测试通过，
包括受扭矩限幅时保留质心/角动量联合层的回归。
分层运行 `lafan_centroidal_balance_500hz_20260927` 在 5.886 s / 源帧 129
因求解器未通过证书停止。根姿态峰值误差 0.150 rad，但摆脚未充分抬起，
最高间隙仅约 4 mm，低于管理器 1 cm 的已抬脚判据，后续脚底保护未激活。
摆脚在非目标位置重新承重，足端跟踪未通过；该实验不是成功方案。

共同优化运行 `lafan_joint_tasks_500hz_20260927` 覆盖到源帧 134.76。
已运行部分峰值关节 RMSE 0.175 rad、根位置 0.110 m、根姿态 0.304 rad，
均满足当前误差阈值，但下一步预瞄无可行解，旧支撑脚的踝滚转触及限位，
整段验收仍失败。审计见同名 `.reference_failure.audit.json`。

新增可选 `--swing-liftoff-barrier`（要求 `--swing-sole-barrier`）：从已释放
摆脚的起始时刻，以 C2 曲线在 0.2 s 内将最低间隙从 0 升至 2 cm。
角点加速度约束包含移动间隙下界的速度/加速度，仍在落地前 0.1 源秒释放。
它解决“先达到 1 cm 才启用保护”的启动空隙，不改变关节/扭矩/接触限制，
也不保证摆脚必然可行。有限差分、管理器、约束组合测试 32 项通过。
物理运行 `lafan_liftoff_floor_500hz_20260927` 仍在第一次换脚触发旧脚踝
滚转限位，停止于 5.966 s / 源帧 134.62，峰值关节 RMSE 0.202 rad。
该下界约束没有解决主要的承重转换问题，默认不启用。

骨盆/躯干显式优先对照 `lafan_pelvis_balance_500hz_20260927` 将根姿态峰值
误差降至 0.116 rad，但关节 RMSE 达 0.509 rad，在 6.05 s 失去旧脚支撑。
新增可选 `--com-height-margin` 用移动参考的导数约束 COM 下沉；例如 `.02`
允许比规划高度低 2 cm。它是加速度级下界，不是高度不变的硬保证。
使用 `.02` 的 `lafan_height_floor_500hz_20260927` 仍在 5.986 s / 源帧 134.89
QP 验证失败；关节/根姿态峰值 0.331 / 0.774 rad。均不是成功方案。

新增实验 `--centroidal-preview`：在原始落脚位置、源时钟和高度曲线下，
联合规划水平 COM 与角动量。分段动力学满足
`c_xy'' = (g+h'')/h * (c_xy-cop) + [-Hdot_y,Hdot_x]/(mass*h)`；
压力中心保持在支撑脚内，XY 角动量/变化率预算为 4 kg·m²/s / 30 Nm，
终端 COM、捕获区域和角动量共同约束。高度系数在每段中点冻结，这是
简化预瞄，不包含未来全身关节/扭矩的可行性证明。QP 系数缩放后独立
检查原物理单位约束残差；质心/速度分段连续，加速度允许在控制段边界变化。
下发时使用同一预瞄的 COM 和 XY 角动量目标，yaw 保留源参考。

同一 5.9 s 状态的离线对照中，零角动量变化在所测 0.65–1.5 s 时长下
不可行；允许有界角动量时 0.65 s 方案通过独立 LP 和 QP 可行性检查，
最大原单位约束残差 6.1e−7。它用到了上述预算边界，不能据此保证全身执行。
证据：`acc_data/g1_centroidal_feasibility_20260927.json`。
物理对照 `lafan_centroidal_preview_500hz_20260927` 停在 5.954 s / 源帧
134.95：已执行片段峰值关节 RMSE 0.190 rad、根位置 0.127 m、根姿态
0.321 rad，但下一步预瞄不可行，旧脚失去支撑。将 COM/角动量置于独立
首层的 `lafan_centroidal_staged_500hz_20260927` 更早在 5.714 s 停止，
摆脚在计划落地附近仍高约 24 cm；严格平衡分层并未解决任务冲突。

随后统一了落地转换的质心模型：捕获反馈将角动量变化率及竖直加速度
换算为实际压力中心，再投影到实测接触区域；力矩符号有独立回归测试。
`lafan_centroidal_capture_500hz_20260927` 仍停在约 5.958 s / 源帧 135，
未改善完整性。执行片段峰值关节/根位置/根姿态误差为 0.191 rad /
0.128 m / 0.322 rad。两种通过片段精度的运行均未通过整段验收。

预瞄现还约束每段两端的水平加速度，遵守控制器 4 m/s² 请求上限和
同一摩擦系数 0.6，并在 QP 目标中加入终端速度；接受解检查物理残差、
驻点和互补条件。13 项质心预瞄、源时钟及平衡测试通过。加入这些约束
后，同一 5.9 s 状态的 0.65 s 预瞄也不可行。原始摆脚轨迹物理复测
`lafan_centroidal_source_500hz_20260927` 停在 5.94 s / 源帧 134.99，
旧脚踝滚转的下一步位置/制动约束冲突。峰值关节/根位置/根姿态误差为
0.157 rad / 0.096 m / 0.206 rad，片段精度改善，覆盖未改善。

离线逐项约束对照确认：保留实际压力区域、摩擦、4 m/s² 加速度和动量
预算时，去掉“终端动量必须距原 PKL 小于 0.25”的人为跟踪约束即可求解；
仅去掉终端捕获约束仍不可行。证据 `g1_centroidal_constraint_audit_20260927.json`
中的解仅用于诊断。当前实现将终端源动量改为二次跟踪代价（权重 10），
仍保留全程动量/变化率 4 / 30 预算和全部物理约束。同一状态的实现回放
原单位最大残差 5.8e−9，见 `g1_centroidal_feasibility_20260927.json` 中的
`implemented_preview`。14 项相关测试通过，包括不可达源动量应让步于
预算的回归。物理验证另存 `lafan_centroidal_soft_terminal_500hz_20260927`。
此前组合回归 112 项通过，尚未取得全段跟踪或墙钟 500 Hz 证明。

终端代价版本在 5.89 s 进入第二次摆脚，但 5.926 s / 源帧 134.71 仍因
左踝滚转限位停止。实测接触冲击使关节离开原制动包络，原固定恢复预算
和二阶距离约束不足以保证刹住。新增可选 `--joint-limit-impact-recovery`：
仅在实测向外速度超出原停止包络时，以剩余行程计算 `v²/(2*d)` 制动下界，
最多要求单步停住；不截断成虚假的可执行加速度，扭矩/加速度仍由 QP
检查。22 项关节/加速度约束测试通过。该版本跨过原滚转失败点，运行到
6.632 s / 源帧 144.79，随后触发左踝俯仰限位，根位置峰值误差 0.216 m。

接触观测审计还确认：离地脚的实际承重点从脚尖外角移到前端尖点后，
旧模型仍保留相距 3.64 cm 的角点。现 `--observed-contact-patches` 对
离地脚也使用 `ObservedSoleContact`，根据实测载荷更新点、任意斜边和
多边形；是否放平只影响软姿态任务，不能删掉真实接触。观测回放的
锚点到承重点中心误差由最大 36.4 mm 降至 0.36 mm。证据：
`g1_outgoing_contact_anchor_audit_20260927.json`、`g1_outgoing_contact_replay_20260927.json`。
47 项接触/制动组合测试通过。物理运行 `lafan_observed_toe_500hz_20260927`
中，[5.7,5.9) s 踝滚转加速度峰值约 54 rad/s²；仍在 6.614 s / 源帧
144.62 停止，关节/根位置/根姿态峰值为 0.162 rad / 0.214 m / 0.228 rad。

严格水平 COM/角动量分层对照 `lafan_centroidal_xy_500hz_20260927` 在
5.332 s / 源帧 116.08 更早失败。6.5 s 状态的独立任务层比较也表明，
此时单独保住 COM 会产生很大的角动量和关节请求误差，不能仅调优先级。
见 `g1_linear_balance_tradeoff_20260927.json`。

第二步的时长审计发现，0.65 s 已满足预瞄约束，但仅比较不同重定时动作
的拟合代价会偏好约 1.5 s，源时钟在摆脚中段明显减慢。当前质心预瞄改为
优先选择从请求时长起的最早认证可行候选，仅在不可行时延长；每个候选
仍优化完整 COM/角动量代价并检查原物理约束。15 项预瞄/时钟/平衡测试
通过，包含不可行短时长必须延长的回归。物理验证为
`lafan_centroidal_timing_500hz_20260927`，尚未取得完整跟踪通过结果。

本轮增加可选 `--motor-armature`，使用 Unitree 官方 G1 MIMIC 的关节端
电机折算惯量（`examples/g1_armature.py` 中注明来源）。离地对照确认
PhysX 质量矩阵 API 不包含该项，但积分器包含，因此 QP 读取动力学时
将实测 armature 加入关节对角线；URDF 一致性检查也显式计入该项。
无接触移动状态的最大关节加速度 RMS 误差从 96.01 降至 0.00893 rad/s²。
数据：`acc_data/g1_armature_dynamics_audit_20260927.json`。此结果验证
动力学读取，尚不能证明落地或全段跟踪通过。74 项相关回归测试通过。

可复现的物理回归入口（实验配置，尚未全段通过）：

```bash
bash examples/run_lafan_tracking.sh --output-path acc_data/lafan_armature_test.npz
```

追加 `--show-isaac-window` 可显示窗口。控制和扭矩频率均为 500 Hz，
保留原始步幅、扭矩/加速度/接触约束。上一轮的低载荷脚跟释放仍在
5.680 s 失败：右踝横滚速度约 −9.89 rad/s，制动与加速度约束冲突。
该失败是本轮电机惯量对照的出发点。

物理对照 `lafan_armature_500hz_20260927` 已跨过原首次触地失败点、完成
第二次换脚，但约 6.54 s 遇到原生段错误；Python 栈停在预瞄积分中，
尚未定位底层根因。调试器重跑跨过该点，到 6.6 s 关节 RMSE 已达
0.342 rad，主动停止该未通过配置。6.5 s 快照的完整历史审计显示：
源帧仅到 147.84 / 7838，峰值关节误差 0.259 rad、根位置 0.164 m，
第一次触地附近右踝横滚速度峰值降至 4.24 rad/s。3250 次 QP 与扭矩
按 2 ms 一一对应，但 QP P99 3.62 ms，仍非实时通过。

`MovingZmpStep` 的竖直制动候选时长现可批量积分，与独立物理时间
积分对照的 8 项预瞄测试通过（相关测试累计 82 项）。80 个候选的
离线对照耗时 0.473 s / 原逐个 0.896 s；这仍不适合每 2 ms 重算。
当前继续验证 `--separate-balance-priority`，优先保障质心轨迹，尚未
证明全段可行；不能将上述局部改进视为完整跟踪成功。

质心分层第一次物理对照在 5.106 s 被矩阵凸性检查拒绝。独立 LP 确认
原物理约束可行；第二层单精度 `AᵀWA` 的最小特征值为 −2.85e−5。
现让 DAQP/OSQP 的任务目标在双精度中组装，控制输入/输出精度及所有
物理约束、正则项保持原值，不放宽非凸拒绝条件。新增低秩任务回归，
44 项求解器/任务测试通过；物理重跑为 `lafan_double_balance_500hz_20260927`。

该分层对照在 5.986 s 因下一步预瞄不可行、随后失去旧脚支撑而停止；
单独改用 0.2 s 摆脚过渡也未通过。新增实验 `--double-support-preview`
将落地终端捕获区域建在旧脚趾边与新脚脚底组成的双支撑区域内，并
优先匹配源动作 COM 速度；真正抬旧脚仍需实测承重确认和下一步预瞄。
离线同状态比较的前向落地速度从 0.909 降至 0.719 m/s，源速度为
0.523 m/s，见 `g1_landing_options_20260927.json`。这不是全身可行性保证。
另修复显式接触位姿 IK 对已可行源姿态仍先屈膝的问题；现在保持源种子，
但该修正在本次首步几何中没有显著改变最终高度。新增/更新的 17 项
预瞄、接触几何和摆脚测试通过。物理对照 `lafan_double_support_500hz_20260927`
使用分层、0.2 s 过渡和双支撑预瞄，尚未通过全段。

**全段 `walk1_g1.pkl` 跟踪尚未通过。** 最新代码继续保持 2 ms 物理步长、
每步重解 QP 并下发一次扭矩；实际计算耗时尚未满足稳定的实时 2 ms。

本轮补充：

- 连续换脚确认：`SETTLE` 中连续满足的下一支撑脚载荷/预瞄确认可延续
  到 `PREPARE_STEP`，不重新等待完整确认周期。仍需完成落地稳定、源
  抬脚时序门和预瞄；载荷或预瞄失效即清零。新增两个回归测试通过。
- 源速度界使用全局时间的解析链式求导，修复起点贴近整数源帧时，
  零长度浮点分段导致除零的问题；新增临界源帧测试通过。
- 第二步的高度规划诊断见 `lafan_height_braking_comparison_20260923.json`：
  从同一个实测状态出发，3 m/s² 竖直制动将规划继续下沉从 3.70 cm
  降至 1.11 cm，原落脚位置不变。这是离线规划比较，尚非物理通过。
- 已保存状态的离地动力学对照 `lafan_moving_free_response_20260923.npz`
  确认关节摩擦、armature、内部驱动刚度/阻尼均为零；移动状态下自由
  响应基本吻合，不能把接触阶段偏差归因于未经证实的隐藏关节驱动。
- 新增 `--separate-balance-priority`：先优化 COM，再优化其他已选优先
  任务，最后优化完整跟踪目标。每一层保留所有前层输出锁，并独立验证
  原物理约束和求解状态；各层原始 QP 均可保存回放。
  `--posture-last-priority` 还可将 29 关节姿态放在之后。两种分层物理
  试验均未通过；后者在首次摆脚的关节误差更大，不能作为推荐配置。
  当前改回原任务组，单独验证连续确认及竖直制动。
- 原问题数值重试：自动分解可在近相关输出锁上误报不可行，现以有限
  次近端分解重试；所有接受结果仍需通过原始可行性、驻点和互补性
  验证。三层输出锁、前层失败传播和真不可行拒绝的测试通过。
  `lafan_posture_last_conditioned_replay_20260923.json` 的四阶段回放均
  通过；其中最终阶段 6.0 ms，仍不满足实时 2 ms。
- 完整修正后的原速对照 `lafan_source_bound_speed1_500hz_20260923`
  在 6.072 s、源帧 140.35 失败。第二次摆脚发生过早触地，关节 RMSE
  峰值 0.340 rad；根位置/朝向峰值 0.0885 m / 0.2447 rad。
  3036 次 QP 与扭矩按 2 ms 一一对应；QP P99 3.70 ms，未实时达标。
  半速对照在 10.068 s、源帧 135.26 右踝限位不可行，也未通过。
- `ObservedSoleContact` 按实际承重点区分单点、任意方向边缘和多点区域。
  三点支撑使用其真实三角形力矩约束；脚侧边不再误作自由单点。
- 单地面下 5.8 s 和 5.9 s 的响应快照确认，实际下发扭矩与 QP 输出
  完全一致；余下动力学偏差需要结合接触切换和负载转移分析。
- DAQP 数值失败可用近端迭代重试，结果仍必须通过原问题的独立验证。
  `lafan_speed07_qp_retry.json` 记录此前失败的优先级问题通过，残差 9.7e-12；
  此次重试耗时超过 2 ms，不代表实时达标。
- 源摆脚时长现包含原 PCHIP 路径、五次时间映射、起步过渡和落地高度
  修正的连续速度界。按每个多项式区间的 Bernstein 控制点做保守界检查，
  不只按起终点直线距离估算；运行时速度保护继续保留。
- 76 项相关测试通过。`lafan_observed_sole_500hz_20260923` 已完成第二次
  换脚，但在随后摆脚速度保护处停止。0.7 倍速度对照仍发生后续预瞄
  不可行和旧脚失去支撑。`lafan_source_bound_speed05_500hz_20260923`
  是当前半速、原始步幅的物理对照，不能作为已通过配置。

### 当前回归：右脚接触区域更新

`lafan_limit_diagnostic_500hz_20260923.reference_failure.npz` 在 5.990 s
记录右踝俯仰限位不可行。实测右脚两端脚跟已同时承重，但旧
`ToeContact.update_support` 因原角点仍接地而提前返回，始终保留单点
模型，遗漏第二接触点的运动约束和可用接触力矩。

现在优先识别新承重边缘；新增边缘的旋转参考也在形成时重新锚定，
避免恢复单点阶段的自由倾斜姿态。机器人状态和原 PKL 均不重置。
28 项接触/求解器测试通过。第一次物理对照 `lafan_line_promotion_500hz_20260923`
确认边缘升级，但在 5.736 s 发生数值精度停止；原始问题经精化可解，
物理残差约 1.9e-8。后端现允许近可行候选尝试精化，接受仍要求
完整 KKT 与 1e-6 原约束可行性检查。包含旋转锚点修正的后续试验为
`lafan_line_anchor_500hz_20260923`，尚未确认全段通过。

### 当前修正：重复地面碰撞体

原始 PhysX 回调确认，旧场景同时启用了 `/World/GroundPlane/geom` 与
`/World/GroundPlane/collisionPlane` 两个重合地面，而足部传感器只过滤前者。
在 5.662 s，右脚对后者产生 80.965 N、对前者为零，因此详细接触点中
没有右脚；左脚的 246.364 N 也被分为 181.005 N 和 65.359 N。
证据保存在 `acc_data/lafan_duplicate_ground_diagnosis_20260923.json`。

场景初始化现保留原生 `collisionPlane`，关闭可视网格的重复碰撞，
将地面摩擦材质和接触过滤器都指向同一物理平面，并断言有效地面碰撞体
恰好为一个。不推测缺失接触点，不放宽 QP 验证。物理回归为
`lafan_single_ground_500hz_20260923`，仍需验证完整动作与实际计算耗时。
下面的历史结果都来自修正前场景，不能据此宣称本版本已通过。

单地面首次回归确认：全部 2838 个控制周期的净竖直接触力与过滤地面力
完全一致，右脚跟落地已进入双脚支撑/点接触约束。运行在 5.676 s 因
DAQP 的互补性误差停止（原约束可行），源帧仅到 129.12 / 7838；
关节/根位置/根朝向峰值为 0.157 rad / 0.046 m / 0.116 rad。
QP 中位数 2.006 ms、P99 3.533 ms，仍未实时达标。
原矩阵回放显示，新接触改变的活动约束超过四个；将罕见数值精化的
迭代上限改为 16，保留全部独立 KKT 和物理阈值后，此问题通过。
该次冷精化约 8.3 ms，不能算作实时成功。37 项相关单元测试通过；
后续物理回归为 `lafan_single_ground_polish_500hz_20260923`。

最新状态：仍未完成全段物理跟踪。`lafan_source_transfer_500hz_20260923`
在 6.132 s、源帧 135.396 失去旧脚支撑，根位置/朝向峰值误差分别为
0.1083 m / 0.2867 rad，关节 RMSE 0.2791 rad。3066 次 QP 与扭矩
一一对应，但 QP 中位耗时 3.398 ms、P99 约 0.507 s，不满足实时预算。

时序对照 `lafan_timed_takeoff_500hz_20260923` 检查两个修正：

- 摆腿时长根据实际五次平移与抬脚曲线的全区间速度极值求解。旧三角
  上界在上一轮 6.00 s 状态给出 0.779 s，实际曲线仅需 0.702 s
  （均含 0.1 s 落地提前量），后者落在独立预瞄的 0.60–0.76 s 可行区间。
- 可选 `--takeoff-lead .1` 允许在源抬脚帧前最多 0.1 源秒释放；仍要求
  实测新脚承重、60 ms 确认和预瞄通过，保持连续源时钟。默认提前量为零。

该轮第二次换脚成功触发，但在 6.176 s 出现左右脚碰撞，6.194 s
QP 失败；源帧 145.453 / 7838，关节 RMSE 峰值 0.4018 rad，根位置
0.0974 m、根朝向 0.2806 rad。3097 次 QP/扭矩一一对应，QP 中位数
3.40 ms、P99 685 ms。`lafan_timed_takeoff_500hz_20260923.failure.audit.json`
明确记录全段验收失败，不能作为成功运行配置。

另有待物理验证的 `--source-swing-reference`：摆脚使用原 PKL 的 FK
位置、姿态与解析导数，加入实测起点位置/旋转过渡，避免水平脚姿态与
原关节姿态冲突。它不保证源动作本身动力学可行，也不作为通过结论。
相关曲线、旋转导数、接触时钟共 43 项测试通过。
物理对照日志为 `/tmp/lafan_source_swing_500hz_20260923.log`。
此轮在 5.890 s 失败；源摆脚改善了第一步空中关节误差，但实际落地滞后。
随后独立摆脚时钟提前 0.1 s 的对照 `lafan_source_swing_lead_500hz_20260923`
仍在 5.864 s 失败。当前摆脚路径已补回有界的触地搜索，沿原轨迹到达
落地帧后，最多下探 4 cm，0.2 s 内平滑完成；仍以实测接触结束摆动。
该补丁的物理对照为 `lafan_source_swing_search_500hz_20260923`。

任务冲突诊断 `lafan_source_swing_lead_500hz_20260923.tradeoffs_2750.json`
在 5.5 s 原始约束下确认：COM + 摆脚 + 躯干 + 骨盆四组请求可同时实现，
第二阶段任务加速度残差约 0.02（已有优先级锁容差）。原配置同时优先
17 个上肢/腰部关节请求，却产生 1.786 m/s² 的 COM 竖直残差。
因此需在此源摆脚版本上重测 `--body-orientation-priority`，
不能沿用旧水平摆脚对照的结论，也不能将一个时刻的可行性视为全程保证。

触地搜索对照在 6.004 s 第二次抬脚后失败：关节 RMSE 峰值 0.2179 rad、
根位置 0.1468 m、根朝向 0.1521 rad。记录为
`lafan_source_swing_search_500hz_20260923.failure.audit.json`，未通过全段验收。

### 可选 DAQP 后端

新增 `--qp-solver daqp`，使用 [DAQP](https://github.com/darnstrom/daqp)
0.9.1。按优先级阶段隔离原始解热启动，保留所有原始等式、不等式和上下界，
独立验证物理残差、驻点条件与互补性；失败候选不作为扭矩命令。优先级锁
仍使用现有 0.02 加速度容差。已通过后端、力矩约束、伺服耦合和频率审计测试。

本机隔离安装及启动环境：

```bash
.runtime/motion_generation/bin/python -m pip install --target .runtime/qp_solvers --no-deps daqp==0.9.1
export PYTHONPATH=src:.runtime/qp_solvers
```

运行时选 `--qp-solver daqp`，不使用 OSQP 专属的 `--qp-reuse-workspace`
和 `--qp-active-set-predictor`。`lafan_source_body_daqp_500hz_20260923`
同时验证源摆脚、触地搜索和 `--body-orientation-priority`。
离线原始 QP 测得单阶段约 0.4–2.7 ms，见 `lafan_daqp_benchmark.json`；
尚不证明两阶段 QP 加完整控制循环满足实时 2 ms，也不证明全段跟踪成功。

DAQP 验证补充：

- 强制近端迭代在第一步优先级 QP 上达到迭代上限，尽管该问题可行。
  改为奇异矩阵自动启用近端法；数值停止仅在独立原始约束与 KKT 检查
  全部通过时接受，显式不可行/非凸状态仍拒绝。大的乘子放大微小边界
  残差时，使用原矩阵做活动集精化，不放宽物理阈值。
- `examples/benchmark_qp_backend.py` 可无仿真回放保存的两阶段问题。
  `lafan_daqp_snapshot_suite.json` 的 32 个阶段全部通过；CPU 冷启动
  中位数 0.734 ms、最大 1.933 ms，仅代表各单阶段静态回放。
- `lafan_body_daqp_cert_500hz_20260923` 在 5.696 s 停止。根位置/朝向
  峰值误差为 0.0845 m / 0.1098 rad，但上肢和膝关节明显偏离，整体
  关节 RMSE 0.5371 rad，故不采用该优先级配置。实际 QP 调用中位数
  3.04 ms、P99 11.85 ms，仍不满足实时 500 Hz。
- 下一轮 `lafan_joint_barrier_daqp_500hz_20260923` 恢复上肢优先，启用
  已有的 `--joint-limit-barrier-bandwidth 8` 做提前制动；使用 `--device cpu`
  对单机器人控制计算减少小矩阵 GPU 往返，物理与扭矩步长仍为 2 ms。

CPU 制动对照在 5.662 s 停止：新右脚已有约 81 N 地面载荷，但因脚掌未平，
旧调度仍保持左脚单支撑。QP 中位耗时 2.032 ms、P99 3.539 ms，尚未实时达标。
该现象需要接触模型修正，而不是把未放平的脚直接当成完整平脚。

新增可选 `--observed-contact-patches`（需 managed early touchdown + adaptive
toe contact）：`ContactPatchManager` 为两脚分别维护实测脚尖/脚跟线或单点。
载荷、底面位置与地面法线确认后，脚跟可立即进入接触模式；其 Jacobian、
锚点、摩擦/力矩锥独立映射。落地脚使用软放平任务，稳定平脚再恢复平脚约束，
起步承重确认仍保留。接触管理、脚尖模型、调度及优先级 50 项测试通过。
物理对照为 `lafan_dual_patch_500hz_20260923`，尚无全段通过结论。

离线另评估了 [ProxSuite](https://github.com/Simple-Robotics/proxsuite)
0.7.3，安装在 `.runtime/qp_solvers`，未接入控制器。对保存的同一 QP，
原始形式有数值不可行误判；消去等式的形式能解可行快照，但仍需约
5–74 ms，未达到 2 ms。证据为 `acc_data/lafan_proxqp_reduced_benchmark.json`。

按最新要求，`--control-frequency` 默认改为 500，物理与扭矩周期仍为 2 ms。
基线完整源帧试验日志为 `/tmp/lafan_full_500hz_20260923.log`，开启
`--until-motion-end`，600 s 仅为仿真时间上限。该轮在 7.656 s 失败，记录为
`acc_data/lafan_full_500hz_20260923.failure.npz`；源帧仅到 156.50 / 7838，
仍须通过全段源进度、最终稳定支撑及原 PKL 保真度检查。

该轮 3828 次 QP 与 3828 次扭矩更新一一对应，物理步长为 0.002 s。
第二步 QP 预测与实测 COM 加速度的 RMS 差约 0.226 m/s²，但参考到 QP
的偏差明显；第二次落地后 COM 下沉，左踝接近下限，物理预测要求至少
242.37 rad/s² 制动，与 150 rad/s² 指令上界冲突。HiGHS 独立确认同一
硬约束不可行。峰值关节 RMSE 0.449 rad、根位置误差 0.249 m、根朝向误差
0.558 rad，均未通过现有保真度阈值。

另发现旧脚卸载后，两脚尖仍在实测接触流形中贴地，但因载荷低于 0.5 N，
旧的脚尖释放检查始终不通过。`--toe-off` 现分别检查新脚承重和旧脚尖贴地，
保留 60 ms 确认、实际脚尖几何、非负法向力和脚尖力矩约束。此修改有
12 项接触测试通过；贴地几何不作为单独承重证明。物理对照在 6.24 s、
7.32 s 两次实际释放旧脚脚跟，但随后旧右脚退化为单点接触，7.484 s
右踝 roll 速度达到 5.37 rad/s 并接近上限而失败。记录为
`lafan_toe_release_500hz_20260923.failure.npz`，完整跟踪仍未通过。

新增 `examples/analyze_control_rate.py` 核对 QP 时间、扭矩时间、真实物理
时间和每条扭矩使用的 QP 更新编号。500/500 应为一一对应；仅设置频率参数
不作为证据。新增 `history_qp_wall_seconds` 分别报告实际求解耗时，仿真
2 ms 周期不等于实现了真实机器人的实时 2 ms 计算预算。

`--motion-interpolation pchip-analytic` 是解析导数对照选项：保留 PCHIP
姿态样本，解析计算重定时后的关节/根部速度与加速度，再经 Jacobian 和
偏置生成脚、手、躯干、COM、角动量参考。已有 `pchip` 路径保持原行为。
同段 float32 位姿二阶差分的根加速度数值误差在 2 ms 间隔可达约 0.21 m/s²；
证据为 `acc_data/lafan_reference_difference_precision.json`。解析参考相关
33 项测试、调度/优先级 33 项测试及频率审计 3 项测试通过，尚不代表全段
物理跟踪通过。

解析导数物理对照 `lafan_analytic_500hz_20260923.failure.npz` 在 7.280 s
第二次落地后失败，源帧到 155.03。峰值关节 RMSE 0.260 rad、根位置误差
0.196 m、根朝向误差 0.278 rad；根部两项通过阈值，关节和全段验收未通过。
3640 次 QP/扭矩一一对应，但 QP 实测耗时中位数 4.51 ms、P99 615.85 ms、
最大 1.54 s，**不满足真实机器人 500 Hz 的实时计算预算**。频率与保真度
审计见同名 `.failure.audit.json`。后续脚跟释放修复对照日志为
`/tmp/lafan_toe_release_500hz_20260923.log`，中段完整 QP 可用
`examples/diagnose_task_tradeoffs.py` 在不改变物理约束的条件下比较任务冲突。

后续可选实现与验证范围：

- `--qp-reuse-workspace` 为 OSQP 的平衡/跟踪阶段分别保留求解工作区和
  原始/对偶解；稀疏结构变化时重建，复用未收敛或约束残差超标时冷启动
  重试，保留原有独立细化和全部验收。13 项后端/优先级测试通过。在同一
  6.6 s 实际 QP 的重复离线求解中，第二阶段从约 229 ms 降为 1.78 ms，
  解最大差约 2e-14；此结果不代表连续变化 QP 或完整控制环已达到实时。
- `--height-braking-acceleration` 可先以指定加速度预算制动实测竖直速度，
  再连接 C2 连续的落地高度过渡。10 项高度/预瞄测试通过，尚未在物理
  对照中启用。
- 对照 `lafan_aligned_warm_500hz_20260923` 启用已有
  `--align-takeoff-frame`，避免第二步在源帧 129 便摆动，而原动作直到
  第 137 帧才抬脚；随后仍在第一次落地后失去旧脚接触。单点脚尖姿态
  阻尼对照也在 6.718 s 停止，不能视为连续行走成功。
- 工作区复用现保留数值为零的稀疏系数位置，避碰不激活的固定行保持
  `0 <= 0`；激活时原不等式不变。37 项求解器/任务测试通过。
- `--moving-zmp-preview` 允许压力中心沿五次曲线在真实支撑多边形内移动，
  约束原始落地 COM 和终端捕获点。5 项轨迹/动力学/边界测试通过。
  物理对照 `lafan_moving_zmp_500hz_20260923.reference_failure.npz`
  仍在 6.718 s 失去旧脚支撑。3359 次 QP/扭矩更新均一一对应；工作区
  复用率 86.5%，QP 耗时中位数 4.53 ms、P99 814.51 ms、最大 2.85 s，
  **尚未达到实时 500 Hz，也未完成全段跟踪**。
- 参考失败记录现保留完整扭矩历史。连续时钟在制动距离和单调性允许时
  使用名义落地速度，避免把延长摆腿的平均速度继续施加到下一段双支撑。
  `lafan_continuous_landing_500hz_20260923` 使用
  `--motion-ramp-duration .25 --swing-touchdown-lead .1`，在 6.436 s、
  源帧 136.95 因临近换脚时承重比例从约 98% 降至 74% 而撤销授权，
  时钟已无法在第 137 帧前制动。3218 次 QP/扭矩一一对应；峰值关节
  RMSE 0.270 rad、根位置误差 0.141 m、朝向误差 0.365 rad，仍未通过。
- 换脚承重门槛现有滞回：首次持续满足 80% 后，降至 65% 以下才撤销；
  实测支撑、预瞄和所有 QP 物理约束仍检查。连续模式在承重确认期间按
  下一事件的制动余量控制速度。`lafan_transfer_hysteresis_500hz_20260923`
  在 6.426 s 换左脚、7.210 s 左脚落地，7.282 s 右踝 pitch 接近下限
  后 QP 失败；HiGHS 确认同一硬约束不可行。源帧到 155.73，峰值关节
  RMSE 0.318 rad，仍未通过。3641 次 QP/扭矩一一对应；预测分支通过率
  52.0%，QP 耗时中位数 3.68 ms，P99 2.27 s，仍不满足实时 500 Hz。
- 另修复抬脚时钟在事件帧夹紧一个周期的问题：旧记录在 5.304 s 出现
  约 306/s 的虚假源时钟加速度。现在在跨过已授权事件帧的周期中切换
  模式，保留未授权越界检查。31 项管理器/时钟测试通过；此项修改晚于
  上述承重滞回物理对照启动，需下一轮验证。
- `--qp-active-set-predictor` 是可选的相邻 QP 加速分支：利用前一个解
  预测激活约束，求 KKT 线性系统，并检查完整物理约束、驻点残差和
  不等式乘子符号；任一检查失败就回到 OSQP。没有把仅可行的解当成最优解。
  39 项求解器/参考时钟测试通过。两个实际 QP 的完整离线预测中位耗时
  约 1.0–1.2 ms，与保存解最大差小于 1.5e-5；更困难的 7.4 s 样本
  未通过预测检查，正确要求回退。尚未取得连续物理运行的实时性能证据。
- `--minimum-step-duration` 默认保持 0.8 s，设为 0 可用源动作时长和
  摆脚速度上限决定最短步时；后续预瞄仍必须通过支撑/落地捕获约束。
- `--source-toe-reference` 使用源脚部旋转、角速度和角加速度，60 ms 内
  平滑引入离地旧脚的软角度任务。旧的固定 1 s/0.25 rad 脚跟抬起目标
  与原动作快速滚动离地不一致。线接触仍只释放 pitch，点接触仍不支持
  任何力矩，实际接触点与锥约束保持原定义。45 项接触/管理器测试通过，
  尚待物理验证；需要 `--toe-off --reference-footsteps source
  --dynamic-reference --motion-interpolation pchip-analytic`。
- `lafan_source_toe_500hz_20260923` 在 6.494 s、源帧 139.16 处因左踝
  roll 接近上限失败；源偏航软任务在单点接触下请求了过快旋转。
  此后只跟踪源 pitch，点接触 roll/yaw 保留原阻尼，并添加可选的
  `--toe-pitch-priority`，避免抬脚跟任务被上身任务的二阶段锁定压制。
  该轮还使用最多 4 次激活约束修正（每次仍须完整 KKT/物理检查）及
  批量步时积分。80 个候选步时离线从 0.804 s 降至 0.043 s；与逐个
  积分的终端转移矩阵最大绝对差 2.6e-5，独立数值/动力学对照通过。
- `--source-stance-rollover` 为后续可选对照：源动作要求脚跟抬起时，
  允许单支撑旧脚在已确认的真实承重脚尖线上滚动，不必等待另一脚落地。
  已有实测脚尖边/点接触则保留其支撑几何。需同时开启源脚尖参考和
  自适应接触。没有给点接触增加任何力矩能力。
  `lafan_stance_rollover_500hz_20260923` 同时改为身体朝向优先后，第一步
  关节 RMSE 在 5.70 s 达到 0.428 rad，已主动中止；不能作为完整跟踪候选。
  72 项相关回归测试通过不能替代这个失败的物理结果。
- `--roll-aware-landing` 将旧支撑脚按源 pitch 绕固定脚尖规划，再据同一
  足部几何求落地 COM。340 个落点的离线未收敛数从 54 降到 47；仍有
  未解决的几何问题。第二个落点的参考关节 RMSE 从 0.142 降至 0.082 rad。
  这项检查使用理想落点顺序，不代表实际落点误差下的动态可行性。
- `--source-transfer-reference` 在双支撑保留源 COM 的前进位置/速度/加速度，
  横向仍向承重脚转移。原反馈相当于每步都向脚踝中心停下。新增运动参考
  反馈仍投影至支撑多边形，并受原 QP 约束检查。17 项参考/几何测试通过。
  当前运行 `lafan_source_transfer_500hz_20260923`：恢复 `--upper-body-priority`，
  加入上述滚动几何、连续前进、脚尖 pitch 优先级，并设
  `--minimum-step-duration 0`。第一步找到 0.6 s 可行点质量预瞄；全段物理
  跟踪与实时性能仍待验证，尚无通过全段验证的推荐命令。

## 50 Hz QP 跟踪进展（2026-09-22，完整跟踪仍未通过）

最新的 `lafan_consistent_landing_50hz_20s.reference_failure.npz` 在 6.08 s、
7.46 s 两次落地，原始落点的 XY 误差分别为 1.08 cm、1.34 cm。随后
动作时钟越过尚未确认的第 158 帧换脚事件而停止，仍不是完整跟踪。
连续时钟现根据下一次换脚的距离限制当前步的结束速度，保留减速余量；
短双支撑间隔回归及相关管理测试共 26 项通过，物理复跑进行中。

这次实际修复包括：

- `ToeContact` 原先只识别脚尖横向接触边，把两个承重脚跟点误归为单点。
  现保留真实脚跟接触边及其力矩能力，支撑作用点仍取实测位置；没有承重
  的第二个点不计入支撑。11 项接触测试通过。
- `--nominal-implicit-servo` 在既有隐式反馈中保留标称电机增益，避免
  脚边接触时质量缩放将踝阻尼降得过低；扭矩仍限制在原电机上限内。
  控制器、接触和调度相关的 73 项测试通过。此选项尚未通过完整步态。
- 同时使用 `--support-aware-height --source-com-endpoint` 时，用真实
  旧支撑点、新落点和源动作前后 COM 偏移共同求参考几何，再将同一
  COM 终点/高度交给动态预瞄。第 155 帧对应几何的关节 RMSE 约
  0.144 rad，COM 高度约 0.606 m。强制 COM 在双脚正中的试验未能
  满足同一固定身体朝向下的几何条件，不能靠调整高度单独补救。
  全段离线 340 个落点的此项几何检查仍有 54 个未收敛，不代表整段可行。

上述均保留原始世界落点、物理碰撞及全部控制器验收阈值。局部接触/几何
改进不等于全段跟踪通过；当前没有经过全段验证的推荐运行参数。

最新复核继续采用 QP 50 Hz、扭矩和物理步进 500 Hz：

- `lafan_body_barrier_50hz_20s` 未再报告手腕/手掌碰撞躯干，但在
  第二次摆腿 6.820 s 膝限位失败。新增 `--wrist-body-avoidance` 将躯干及
  对侧肩部加入已有避碰约束，包含 URDF 圆柱碰撞体；26 项相关测试通过。
- `--maximum-swing-speed` 可按起点速度、完整位移和抬脚高度增加摆脚时间；
  起步放行与实际规划使用同一时长下界。它只约束参考脚的平移速度，
  不保证关节可达或动态可行。18 项摆脚、预瞄和动作时钟测试通过。
- 完整 COM/摆脚优先、摆脚与 COM 同时到达的试验
  `lafan_com_pose_sync_50hz_20s` 在第二步 6.920 s 因求解精度失败。
  同一 QP 离线消去动力学等式后重新优化，全部原约束残差从 0.01315
  降到 2.44e-12，转换回 float32 后为 1.38e-5；见
  `acc_data/lafan_nullspace_refinement_audit.json`。仅优化器成功且全部
  原约束重新通过才接受，没有提高验收容差。12 项求解器测试通过。
- 数值修复后的 `lafan_com_pose_sync_refined_50hz_20s` 越过原故障，
  但 7.300 s 再次失败，姿态误差已显著增大。不能将数值修复视为步态修复。
- `--upper-body-priority` 是正在验证的实验选项：活动 COM 优先阶段
  同时权衡躯干朝向及腰部/手臂姿态，避免只保留质心与脚部加速度。
  仍未找到可推荐用于完整原步幅跟踪的参数组合。

后续对照记录（均为 QP 50 Hz、扭矩/物理 500 Hz）：

| 记录前缀 | 结果 |
| --- | --- |
| `lafan_reference_height_50hz_20s` | 解决初始化检查误用于规划后，完成第一次落地及换脚；第二次摆动 7.480 s 失败，质心前进超出预瞄约 20 cm |
| `lafan_full_com_no_momentum_50hz_20s` | 全 COM 优先、停用角动量任务，6.200 s 失败；不作为推荐方案 |
| `lafan_synchronized_com_50hz_20s` | 单支撑全 COM 优先、摆脚与 COM 同时到达，6.300 s 失败 |
| `lafan_support_height_50hz_20s` | 用真实支撑锚点拟合高度，首次落地未确认，6.240 s 失败 |
| `lafan_clearance_com_50hz_20s` | 全 COM + 摆脚高度优先，第一次落地后在 7.120 s 支撑踝限位失败 |
| `lafan_clearance_lateral_50hz_20s` | 横向/高度 COM + 摆脚高度优先，两次落地；落点分别短约 13/16 cm，8.140 s 丢失旧支撑，未满足原步幅 |
| `lafan_position_lateral_50hz_20s` | 摆脚 XYZ 优先但朝向仍为次级，脚边触地后 5.800 s 踝速度突增失败 |
| `lafan_pose_lateral_50hz_20s` | 完整摆脚姿态优先，上肢误差增大，6.700 s 肘关节限位失败 |
| `lafan_source_momentum_50hz_20s` | 源角动量使第一步部分时段误差改善；约 6.2 s 后发生原生崩溃，无完整物理结果，栈保存为同前缀 `crash.txt` |
| `lafan_aligned_momentum_50hz_20s` | 源角动量 + 起步帧对齐，6.860 s 第一次换重失败 |

新增实验参数均默认关闭：`--swing-com-priority`、
`--swing-clearance-priority`、`--swing-position-priority`、
`--swing-pose-priority`、`--support-aware-height`、`--source-momentum-reference`。
`--swing-touchdown-lead` 默认保留 0.15 s。这些参数不是已验证的全程跟踪配置。
记录新增 `history_landing_goal/foot`，用于对照真实落点，避免把提前触地
误报为完成原步幅。原始 PKL 关节/根位姿保真度审计仍然保留。

`--align-takeoff-frame` 等待源准备帧完成后才释放摆脚，避免实际已经摆腿、
源姿态还在双支撑准备阶段；不跳过源帧。`--source-com-endpoint` 保留源
动作的前后质心终点偏移，横向使用双脚中点，并补偿真实支撑锚点与源
支撑脚的差异。动态起步改由同一下一步预瞄判定可行性，不额外要求能
刹停在当前单脚内；真实承重门槛继续生效。25 项调度/预瞄测试通过。
`--finish-com-preview` 可让 COM 轨迹持续到规划终点，避免提前落地时突然
切换为刹停参考；换重 COM 优先级现覆盖双支撑 LANDING/SETTLE/PREPARE_STEP。
上述动态组合尚在物理验证，不能视为完整跟踪实现。

`--adaptive-toe-contact` 现在也观察单支撑脚的点/边接触。此时软角度任务
要求脚放平，区别于双支撑换重中的抬跟；只有全脚平整、承重且低角速度
持续 60 ms 才恢复整脚约束。恢复时只更新控制器的接触锚点，不写入机器人
姿态。34 项接触/调度/优先级测试通过，物理组合验证进行中。

整段 340 次落地的离线几何检查也尚未通过：固定源骨盆 XY 的平脚 IK
有 81 次未收敛，放开骨盆 XY 的对照仍有 47 次，按已放置支撑点和 COM
中点拟合则有 112 次。它们说明当前几何规划仍有限制，不能直接当作
“源动作物理不可行”的证明。见 `acc_data/lafan_*geometry_audit.json`。

当前补充诊断与修复：

- `--contact-aware-servo`（依赖 `--tracking-servo`）按接触约束下的有效
  惯量计算伺服增益，原物理增益上限和耦合离散步长限制仍保留。每个 QP
  更新计算无支撑/左/右/双支撑四组增益；500 Hz 读取实际足底载荷，卸载
  时立即选择相应的较少接触模式。自由轴和线/点接触释放的角度不假定固定。
  初版显式反馈在落地时发生踝速度交替反转，未采用。现使用自由系统的
  耦合惯量计算隐式 PD 扭矩，避免把“有足底载荷”当作“所有足底转动均被
  固定”。17 项关节控制测试通过，包括高增益、低惯量关节的离散能量下降；
  尚未通过整段物理跟踪。
- `servo_time`、`servo_physics_time`、`servo_qp_update_index`、`servo_kp/kd`
  记录每次扭矩更新。`acc_data/lafan_50_500_rate_audit.json` 已核验每次 QP
  对应 10 次扭矩更新，物理/扭矩间隔为 2 ms、QP 间隔为 20 ms。这是仿真
  时间调度验证，不是实际硬件上的求解延迟保证。
- 高度优先与接触伺服对照在 5.760 秒因第一阶段 QP 残差 0.100 停止；
  最终解的物理约束通过。离线同约束 SLSQP 精化将残差降至约 1.6e-14。
  原生 OSQP 后端现对残差较大的候选解执行独立精化，只有优化器成功且
  全部原约束残差不超过 1e-6 才接受；线性可行性检查的点不会直接作为
  控制输出。10 项求解器/优先级测试通过。
- `lafan_refined_contact_servo_50hz_20s` 已在 6.020 秒触地后失败：显式
  高增益导致右踝约 ±10 rad/s 交替反转，同一 QP 被 HiGHS 确认不可行。
  `lafan_implicit_contact_servo_50hz_20s` 越过该故障，6.460 秒在首次落地
  换重时横向失稳；不能据此认为步态已稳定。
- `--swing-lateral-priority`（依赖 `--swing-height-priority`）在支撑脚
  朝向坐标系下优先保持横向质心与高度，前进方向仍参与完整加权任务。
  投影只用于第一阶段，不重复增加第二阶段权重。28 项关节控制/求解器
  测试通过。`lafan_lateral_implicit_50hz_20s.reference_failure.npz` 已完成
  第一次落地和向右脚换重，但第二步规划持续拒绝，9.040 秒丢失左脚支撑。
- 发现落地高度规划误用 `flat_foot_spawn` 的初始位移检查：源第 155 帧
  平脚参考需改变腿部关节最大 0.383 rad，超过初始化的 0.35 rad 门槛，
  导致下一步一直无法放行。现以 `flat_foot_reference` 计算参考几何，
  保留脚 XY/朝向、骨盆朝向、上肢和物理关节限制；不写入仿真实际姿态。
  初始站立的原位移检查保留。几何失败不再被吞成等待中的平衡条件；
  当前预瞄失败原因也记录进历史。7 项几何/预瞄测试通过，物理复核中。
- 实际保存的第一阶段 QP 经后端精化并转换回 float32 后最大残差为
  3.05e-5，见 `acc_data/lafan_qp_refinement_replay.json`。上述 1.6e-14
  是离线 float64 解的残差，两者不可混为一谈。完整跟踪尚未完成。

按用户要求，后续物理试验固定 `--control-frequency 50`，物理及关节反馈为
500 Hz。已保存的控制历史间隔为 0.020 s，详见 `acc_data/lafan_50hz_followup.json`。

- 仅直接力矩/阻尼反馈时，起步手臂误差显著放大；启用已有
  `--tracking-servo` 跟踪每个 QP 周期的预测关节轨迹后，起步部分时段关节
  RMSE 从约 0.4–0.5 rad 降至约 0.03 rad；落地换重仍失败，不能称为跟踪通过。
- `--adaptive-toe-contact` 补充整脚直接变为实测边缘/单点的识别路径。
  只有足底几何已抬起且有真实载荷才启用。对照中旧脚在低载荷时仍平贴地面，
  新路径没有触发，状态与未修改前相同；不能把力集中在一个点等同于其余
  足底几何已离地。
- 平脚几何高度预瞄试验在 6.00 s 已出现新脚放平、载荷约 2.1 N，却低于
  旧即时接触门槛 5 N，随后新脚再次离地。即时接触现要求曾离地、全脚平整、
  正载荷至少 1 N；持续接触确认及后续承重稳定门槛分别保留。26 项接触/管理测试通过。
- `lafan_source_grounded_servo_50hz_20s.failure.npz` 两个阶段的硬约束均通过
  HiGHS 可行性检查，OSQP 却报告近似不可行。将不可行证书容差从库默认
  1e-4 收紧到 1e-8，离线 100000 次迭代候选最大残差约 0.00435；仍需要
  控制器逐项验证，也不代表最优性已收敛。8 项原生 QP/优先级测试通过。

轻载落地识别与更严格证书设置的组合正在物理验证；未放宽动力学、摩擦、
力矩、关节或避碰约束，原始步幅及原 PKL 保真度审计继续保留。

## 最新诊断：手掌/手腕与髋部自碰撞（2026-09-21，完整跟踪仍未通过）

`--joint-limit-avoidance-weight 20` 增加软提前制动；右膝在第二步的制动改善，
但 `lafan_source_soft_joint_brake_20s.failure.npz` 仍在 7.128 秒因左腕限位失败。
15 项关节控制测试通过，不能据此声称全段跟踪成功。

仅增加接触与加速度记录的对照，状态历史与原试验逐项相同，PhysX 报告：

- 5.552 秒，右橡胶掌与右髋 yaw 部件接触。
- 5.800 秒，左腕 yaw 部件与左髋 pitch 部件接触。
- 7.012 秒，左橡胶掌与左髋 yaw 部件接触；随后左腕发生速度反转，
  7.024 秒达到约 24.97 rad/s。该冲击未包含在仅建模脚底接触的 QP 中。

诊断见 `acc_data/lafan_soft_brake_self_collision_diagnosis.json`，原始记录为
`acc_data/lafan_soft_brake_contact_diagnosis_10s.failure.npz`。

新增可选 `--wrist-hip-avoidance`：根据实际 STL 包围盒划分球形覆盖，增加
手腕、手掌与左右髋部的软避碰任务。特别包含 USD 已启用碰撞、URDF 只列为
视觉几何的橡胶手掌。24 项几何/任务测试通过，包括对网格顶点的覆盖检查；
`lafan_source_mesh_avoidance_20s` 消除了第一步原有的两次接触，但第二步在
6.916 秒再次手掌撞髋，7.008 秒失败，仍未通过。
原物理碰撞、关节限位、摩擦和力矩检查继续生效。

新增可选 `--wrist-hip-barrier`（需同时启用 `--wrist-hip-avoidance`），对
新增球形覆盖加入距离与接近速度约束：`h'' + 16 h' + 64 h >= 0`，其中
`h` 为表面间距减去 2 cm 参考余量。约束在距离 40 cm 内参与 QP，作用于
包含伺服响应的物理预测加速度，并独立验证残差。它不保证冲击下或模型误差
下的安全性；若无解仍然失败。44 项 QP/任务测试及 22 项几何任务复核通过。
`lafan_source_collision_barrier_20s` 在 6.802 秒因新增约束残差 0.0618 超过
0.05 停止；同一 QP 离线 OSQP 成功求解，未放宽阈值。改用
`--qp-solver osqp` 的 `lafan_source_collision_native_20s` 无自碰撞地推进到
8.328 秒，但右踝限位约束随后无解，仍未完成。提高姿态权重至 10 的
对照在 7.296 秒失败，未作为推荐配置。

接触记录显示：旧脚两个脚尖点逐渐只剩一个承重点，而旧线接触模型仍
允许滚转反力矩。新增可选 `--adaptive-toe-contact`（依赖 `--toe-off`），
在实测支撑线消失时保留实际有载荷的材料点，释放三个角度约束，同时
将该点的三个反力矩限制为零；物理验证进行中。接触/调度器 22 项测试通过。
另修复了落地确认满 60 ms 后绕过整脚平整检查的分支；有完整脚底高度
观测时，持续边缘触地也不能被确认成六自由度刚性平脚支撑。

新增可选 `--motion-interpolation pchip`：原始采样帧不变，关节插值不越过
相邻两帧范围，速度连续；保真度仍对原始线性插值参考计算。前 180 帧的
500 Hz 离线差分中，加速度峰值从约 9449 降至 833 rad/s²，但超过任务
加速度上限的采样比例没有下降，因此不能凭此声称可执行性改善。
4 项插值测试通过；详见 `acc_data/lafan_interpolation_audit.json`。

整段源数据 7839 帧均在关节位置范围内，原始差分关节速度峰值约
18.89 rad/s；位置范围正确不代表动力学可行。见 `acc_data/lafan_full_source_joint_audit.json`。

## 原始步幅：骨盆控制与承重转换（2026-09-21，尚未完整跟踪）

在修正物理时钟后继续进行原步幅 500 Hz 仿真：

| 组合 | 实测结果 |
| --- | --- |
| 平脚初始化、即时接触、骨盆朝向权重 1、原滚脚门槛 80% | 第一次落地后未触发腰部限位；12.152 秒第二步准备超时，源进度 135.50 / 7838 帧 |
| 同上，滚脚门槛提前到 65% | 6.15 秒进入旧脚脚尖支撑；仍在第二步准备阶段超时，源进度未继续 |

根朝向误差峰值降至 0.1683 rad；提前滚脚后峰值关节 RMSE 为 0.2474 rad，
仍高于当前 0.2 rad 的保真度检查阈值。实测质心停在新脚脚踝后约 4.1 cm，
无法通过平衡/下一步预瞄检查。不能把无 QP 报错但停滞的运行当作完整跟踪。

新增 `--toe-transfer-load-fraction .65` 只改变开始滚脚时的新脚承重比例，
仍需实测两个脚尖有载荷并连续确认 60 ms；完全抬脚继续要求 80% 承重与
平衡/预瞄可行。默认仍为 0.8，短步模式不受影响。

新增 `--transfer-balance-priority`：仅在动态预瞄已开始、落地稳定之后的
`PREPARE_STEP` 承重转换阶段，先优化质心，再保留其已达到的加速度求解其余
动作目标。摆动阶段和第一次起步保留原任务分配。两阶段均使用原物理约束。
`history_balance_priority_active` 标记其启用时段，对应误差字段在未启用时为 0。

**后续实测仍未通过完整跟踪**：原 ADMM 求解器在 6.534 秒成功切入右脚单支撑、
开始第二步，但 7.060 秒右膝高速伸直并触发关节加速度边界失败。
以下为诊断配置，不是已通过的完整跟踪命令：

```bash
cd /home/sky/Humanoid_fabric
conda activate humanoid_fabric
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python examples/test_g1.py \
  --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --direct-wbc --reference-footsteps source \
  --dynamic-reference --swing-metric-scale 10 --toe-off \
  --toe-transfer-load-fraction .65 --early-touchdown --flat-foot-spawn \
  --control-frequency 500 --source-pelvis-weight 1 --transfer-balance-priority \
  --duration 20 --show-isaac-window \
  --output-path acc_data/lafan_source_transfer_priority_20s.npz
```

记录：`acc_data/lafan_source_pelvis_clock_fixed_15s.reference_failure.npz`、
`acc_data/lafan_source_early_roll_20s.reference_failure.npz`；汇总于
`acc_data/lafan_full_tracking_followup.json`。19 项参考管理/脚尖接触测试与
4 项 QP 优先级测试通过；这些测试不替代待完成的物理验证。

### 后续控制器对照（2026-09-21）

- `lafan_source_transfer_priority_20s.failure.npz`：第二步已启动，源进度约
  149.03 / 7838；右膝在 7.060 秒失效。第二次摆动期间质心高于其目标，
  膝关节制动不足；完整跟踪未通过。
- 关节距离屏障 `--joint-limit-barrier-bandwidth 8`：ADMM 对照在 6.092 秒
  第一脚落地时失败，右踝速度突变；离线 OSQP 也判定同一 QP 不可行。
  该屏障仅是可选实验，不是已验证的修复。
- 可选 `--qp-solver osqp`：CPU 原生 OSQP 后端，用于单机器人对照，需要
  环境安装 `osqp` 和 `scipy`。保留两阶段与物理残差检查；迭代达到上限的
  有限候选解仍须通过全部物理检查，显式不可行结果始终失败。当前物理
  实验没有取得比原 ADMM 更好的完整跟踪结果。
- `lafan_source_balance_500hz_20s.failure.npz`：全程三维质心优先时，
  摆动脚未及时落地，6.204 秒 QP 校验失败。
- 新增 `--swing-height-priority`：动态单支撑时仅将质心 z 坐标放在第一
  阶段；水平质心和摆脚仍共同优化。可与 `--transfer-balance-priority`
  配合，默认关闭。固定高度对照在 6.244 秒失败；改为源动作落地高度后
  在 6.248 秒失败，摆动脚仍未落地。停用角动量任务则在 6.056 秒更早失去
  稳定支撑，未采纳为修复。

这三个高度/质心优先试验中的前两个，以及全程质心优先试验，失败时的原始
硬约束均通过 HiGHS 线性可行性检查，说明还存在数值收敛问题。变量高度
失败 QP 用 OSQP 100000 次迭代得到最大残差约 0.00657，但仍未获得最优性
收敛状态，不能称为已收敛最优解。可选原生后端提高迭代预算，并继续逐项
验证物理残差；不增加约束容差。复核见 `acc_data/lafan_variable_height_qp_replay.txt`。

参考管理已跳过随后会被覆盖的源角动量计算；21 项任务测试通过，比较生成的
任务矩阵、偏置、目标、权重及激活掩码，优化前后数值完全一致。
实际仿真中，高度优先启用之前的 2666 个控制周期（0–5.330 秒），状态、速度、
源进度及 QP 加速度也与原基线逐项相同，记录于
`acc_data/lafan_reference_momentum_optimization_regression.json`。
20 项 QP/优先级测试通过。这些单元测试不等于实际行走通过。

## 已修复：开启窗口后 0.02 秒失控（2026-09-21）

用户的 `lafan_alternating_preview.failure.npz` 与通过的无窗口短步回归具有完全
相同的初始状态、首个 QP 输出和首个力矩。关键运行参数差异是可视化开关。
旧循环每 2 ms 更新力矩，却调用 `world.step(render=True)`；本机 Isaac Sim
对此调用执行 `app.update()`，按 20 ms 渲染周期推进多步物理，导致力矩保持时间
与反馈计算假设不一致。记录中的 0.02 秒来自控制循环计数，不是可靠的物理时间。

修复为每次发送力矩后调用 `world.step(render=False)`，每 10 个物理步单独调用
`world.render()`。物理 500 Hz、画面 50 Hz；渲染不推进物理。每步检查实际物理
时间增量是否为 2 ms，结果文件增加 `physics_elapsed_seconds`、
`physics_step_seconds`、`rendering_step_seconds`。

**带窗口的 35 秒实测通过**：7 次左右交替稳定落地，前进 0.83948 m，最大支撑脚
滑移 9.78061 mm，物理时间 35.000002 s。状态、速度、源帧进度、QP 加速度及实际
力矩历史与无窗口基线逐项完全一致。这验证了可视化时钟修复和短步行走；原始
步幅的整段 PKL 跟踪仍未通过。

```bash
cd /home/sky/Humanoid_fabric
conda activate humanoid_fabric
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python examples/test_g1.py \
  --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --direct-wbc --reference-footsteps compact \
  --duration 35 --show-isaac-window \
  --output-path acc_data/lafan_alternating_preview.npz
```

证据：`acc_data/lafan_visual_clock_diagnosis.json`、
`acc_data/lafan_visual_clock_fixed_35s.npz`、
`acc_data/lafan_visual_clock_fixed_35s.audit.json`。

## Isaac Sim 可视化实测（2026-09-21）

已实际运行 `--early-touchdown --flat-foot-spawn --control-frequency 500
--show-isaac-window`，输出
`acc_data/lafan_flat_early_500hz_visual_10s.failure.npz`，启动和运行日志为
`/tmp/lafan_flat_early_500hz_visual_10s.log`。此前等待执行的组合现在已有结果：

- 5.330 秒左脚单支撑，6.078 秒识别右脚落地并进入双支撑。
- 6.118 秒 QP 校验失败并关闭窗口，未运行完所设 10 秒，也未完成第二步。
- 腰部 pitch 为 0.50820 rad、速度 1.44091 rad/s；要求加速度不高于
  -270.50 rad/s²，QP 输出 -210.25 rad/s²，边界违约 60.25 rad/s²。
- 源进度只到约 129 / 7838 帧，峰值关节 RMSE 0.2621 rad，根朝向误差
  0.5358 rad；完整跟踪和姿态保真度检查均未通过。
- 新脚载荷峰值仍约 1596 N。接触激活延迟已缩短，但没有消除冲击或腰部限位问题。

随后申请增加 `--source-pelvis-weight 1` 的可视化对照，自动审批服务因连接中断
拒绝执行，进程未启动，不能报告该组合的物理结果。

## 独立求解器复核（2026-09-21）

对 `lafan_source_early_touchdown_20s.failure.npz`（6.100 秒）和
`lafan_flat_spawn_500hz_8s.failure.npz`（6.112 秒）保存的实际 QP 进行离线复核：

- OSQP 均报告 primal infeasible；HiGHS 将相同硬约束作为线性可行性问题，
  也报告无解。增加在线求解迭代次数不能消除这些时刻的约束冲突。
- 仅在离线诊断中移除腰部 pitch 的预测加速度上下界，两个问题均恢复可行。
  实际控制器仍保留该约束。这定位了当前冲突，但不证明导致腰部高速接近
  限位的全部原因。
- 50 Hz 即时接触试验中，6.04 秒落地切换时腰部 pitch 已为 0.2865 rad、
  速度 4.2078 rad/s；同一时刻质心目标高度向下跳变 25.57 mm。
  高速运动在参考高度切换前已经出现，不能把这一跳变当作唯一根因。

证据：`acc_data/lafan_touchdown_constraint_feasibility.json`、
`acc_data/lafan_touchdown_reference_diagnosis.json`。

用户已明确授权 GPU 执行；新的两次外部执行申请仍因自动审批服务超时未启动。
待验证组合仍为 `--early-touchdown --flat-foot-spawn --control-frequency 500`，
没有新增物理仿真结果。不要把下面的诊断命令当作已验证的完整跟踪命令：

```bash
cd /home/sky/Humanoid_fabric
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/test_g1.py \
  --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --direct-wbc --reference-footsteps source \
  --dynamic-reference --swing-metric-scale 10 --toe-off \
  --early-touchdown --flat-foot-spawn --control-frequency 500 \
  --duration 10 --output-path acc_data/lafan_flat_early_500hz_10s.npz \
  > /tmp/lafan_flat_early_500hz_10s.log 2>&1
```

## 落地时序与腰部限位排查（2026-09-20，完整跟踪仍未通过）

**尚未实现整段 7839 帧 `walk1_g1.pkl` 的物理跟踪。** 本轮增加的开关均为实验选项，
没有将失败的参数组合设为默认值。现有短步路径仍能通过 35 秒回归，但改变了步幅。

发现并区分了两个问题：

- 500 Hz、平脚初始化试验中，新脚在 6.078 秒已平脚落地受力，管理器仍在等待
  60 ms 确认，控制模型继续按旧脚单支撑求解。6.080 秒新脚载荷约 1467 N。
  该记录表明接触激活滞后；它不能单独证明后续失稳的全部原因。
- 新增 `--early-touchdown`，把“实测平脚接触进入控制模型”与“承重稳定后放行
  下一步”分开。50 Hz 对照在 6.04 秒进入双支撑，但 6.10 秒仍失败：腰部 pitch
  已到 0.5088 rad，正向速度 3.2781 rad/s，接近 0.52 rad 上限。QP 刹车约束未通过。

其他可选实验包括初始平脚姿态、骨盆朝向任务、COM 优先两阶段 QP、变化高度预瞄、
关节速度阻尼边界。已运行的原步幅试验均未通过，详见
`acc_data/lafan_full_tracking_followup.json`。这些选项不构成可用的完整跟踪方案。

新增失败诊断会打印违反实测加速度边界的关节名、位置、速度与约束范围，并保存
`physical_acc_lower/upper`，与命令积分器的加速度约束分别记录。

验证：53 项参考管理测试、54 项控制器/任务/QP 测试通过。短步回归
`acc_data/lafan_source_changes_compact_regression_35s.npz` 完成 7 次交替稳定落地、
前进 0.8395 m，最大支撑脚滑动 9.78 mm；状态与 QP 加速度历史和此前基线逐项相同。
该物理回归运行在随后新增的几个默认关闭选项之前，不代表这些新选项已验证。

待验证组合：`--early-touchdown --flat-foot-spawn --control-frequency 500`。
两次外部执行审批均超时；普通沙箱无法访问 CUDA / 创建 Isaac 图形设备，已停止
启动进程，因此本次没有获得该组合的物理运行结果。

## 脚尖支撑与第二步试验（2026-09-20，完整跟踪仍未通过）

新增可选 `--toe-off`：双脚承重转换时，将旧脚从整脚刚性约束切换为脚尖线接触，
释放俯仰转动，同时将该方向接触力矩限制为零。切换需要新脚连续 60 ms 承担至少
80% 总载荷、总载荷至少 100 N，并实测到两侧脚尖接触。脚尖接触点的力臂、
离心加速度偏置与摩擦锥使用一致坐标；没有放宽 QP、摩擦、力矩或关节限位校验。

同时修复了 `SETTLE → TRACK` 的瞬时状态切换造成卸载确认中断的问题，
并把下一步 COM 预瞄可行性检查提前到抬脚之前。

实测 `acc_data/lafan_toe_preview_gate_20s.failure.npz`：

- 6.08 秒第一次落地，6.16 秒旧脚进入脚尖支撑。
- 脚尖支撑期间，接触点相对其固定锚点的最大位置误差 **3.04 mm**。
- 7.36 秒切到右脚单支撑，开始第二步；此前旧方案在双支撑转换中失去旧脚支撑。
- 7.64 秒右膝接近限位时，QP 加速度边界校验失败。仅开始第二步，不代表第二步
  落地或完整源动作跟踪通过。放行时旧脚仍约 44 N，因此又做了卸载力上限试验；
  该试验在 7.46 秒更早失败，已撤回，没有纳入当前 `--toe-off` 行为。

以下是**会触发失败停止的诊断命令**，不是通过验证的完整播放命令：

```bash
python examples/test_g1.py --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --direct-wbc --reference-footsteps source --dynamic-reference \
  --swing-metric-scale 10 --toe-off --duration 20 \
  --output-path acc_data/lafan_toe_trial.npz
```

复核接触：

```bash
python examples/analyze_reference_contacts.py acc_data/lafan_toe_preview_gate_20s.failure.npz \
  --output acc_data/lafan_toe_preview_gate_20s.contacts.json
```

`acc_data/lafan_toe_second_step_response.json/.png` 对照了右膝的瞬时 QP 预测和随后
20 ms 的平均实测加速度。后段已请求减速，实际却继续加快伸直；该差异还包含
周期内力矩与接触变化，不能仅凭它认定某一个模型参数错误。

进一步在 7.54 秒记录 2 ms 响应（`acc_data/lafan_toe_response_7s.response.json`）：
右膝预测加速度 +5.00 rad/s²、最初 2 ms 实测 +9.28 rad/s²，实际发送力矩与 QP
输出一致；随后一个 20 ms 周期内仍再次加快伸直。该记录只运行到 7.56 秒，
用于诊断，未覆盖已知 7.64 秒失败点。100 Hz 对照
`acc_data/lafan_toe_100hz_20s.failure.npz` 在 6.10 秒更早触发加速度边界校验，
没有改善稳定性，默认控制频率仍为 50 Hz。

接触约束接口修改后的短步 35 秒回归再次通过：7 次交替稳定落地、前进 0.8395 m、
最大支撑脚滑动 9.78 mm，记录为 `acc_data/lafan_toe_api_compact_regression_35s.npz`
及同名 `.audit.json`。`--toe-off` 默认关闭。完整原步幅跟踪仍未通过。
70 项相关测试通过，包括脚尖雅可比/偏置/力矩变换、自由俯仰坐标及零俯仰接触力矩、
单角点不得误判为接触线、状态转换中的旧脚归属、QP 和参考管理回归。

## 原步幅完整跟踪进展（2026-09-20，尚未通过）

**整段 `walk1_g1.pkl` 尚未完成物理跟踪。** 35 秒短步回归仍通过，但它只到源动作
第 263.52 帧，并改变了步幅；不能替代原文件的 7839 帧保真度检查。

本次修复与验证：

- `--reference-footsteps source` 使用同一重定时时钟下的源 FK 足部与 COM 参考；
  脚底角点接触候选推断把歧义帧从旧脚踝速度法的 514 帧减至 28 帧。这仍不是接触真值。
- 修正源 FK 模式起步时 COM 目标跳变。新增 `ReferenceBridge`，让参考偏移的
  位置、速度、加速度在过渡两端匹配。实测 5.66 秒起步处目标位置跳变量从
  **0.1100 m 降至 0**；摆动脚平移从实测姿态/速度开始，逐渐回到源落点。
- 修正动态预瞄双支撑转换：新脚是承重目标，两脚仍是共同支撑区。
  之前提前把可行 ZMP 区域缩到新脚，会在 COM 尚处两脚之间时产生反向加速度。
- 源模式摆动时长现在计入 `--motion-speed`。这不代表任意慢速都动力学可行。
- `--diagnose-contacts` 记录 PhysX 实际脚底接触点；
  `examples/analyze_reference_contacts.py` 汇总接触范围与起步目标跳变。
- `--until-motion-end` 与独立保真度检查区分完整源覆盖、最终稳定支撑和误差；
  仅仿真运行结束或 QP 数值可行不会被报告为完整跟踪成功。

| 试验 | 实测结果 |
| --- | --- |
| 当前短步基线回归 | 35 秒通过，7 次交替稳定落脚，前进 0.8395 m，最大支撑脚滑动 9.78 mm |
| 源 FK 起步连续性修复 | 跳变消除，但 6.64 秒落地时 QP 校验失败 |
| 源 FK 摆动权重 2 → 10 对照 | 6.44 秒 QP 校验失败，不作为默认参数 |
| 动态预瞄、摆动权重 10 | 6.08 秒进入双支撑，7.26 秒 QP 校验失败 |
| 再修复双支撑区 | 第一次落地后转移承重，7.58 秒旧脚在放行前失去支撑，管理器停止 |
| 半速动态对照 | 1.62 秒起步阶段关节加速度边界校验失败，未能验证落地效果 |

修复双支撑区的记录只覆盖源动作 `0 → 135.51` 帧，峰值关节 RMSE
`0.5331 rad`、根位置误差 `0.1429 m`、根朝向误差 `0.6164 rad`，未通过完整跟踪
或姿态保真度检查。实际接触记录还显示了脚跟/脚尖局部支撑，而当前 QP 使用整脚
刚性约束。剩余工作包括局部接触模型、承重与抬脚时序、关节边界附近的控制响应；
不能仅靠增加 epoch、降低播放速度或放宽校验解决。

证据文件：

- `acc_data/lafan_compact_regression_35s.audit.json`
- `acc_data/lafan_source_contact_patch_8s.contacts.json`
- `acc_data/lafan_source_bridge_12s.contacts.json`
- `acc_data/lafan_dynamic_support_polygon_20s.reference_failure.npz`
- `acc_data/lafan_dynamic_support_polygon_20s.tracking.json`

47 项参考管理/轨迹过渡/保真度检查测试通过；Python 编译与 diff 空白检查通过。
这些测试与短步回归不构成完整 PKL 跟踪成功的证明。以下保留可复现的短步命令及历史记录。

## 连续行走验证（2026-09-20）

`ReferenceManager + --direct-wbc` 已完成 35 秒平地仿真，并通过独立行走检查：

| 指标 | 实测 |
| --- | --- |
| 完成落地与承重稳定的交替步数 | 7 |
| 沿初始根朝向的净前进距离 | 0.8395 m |
| 预热后最大支撑脚 XY 滑移 | 0.00978 m |
| QP / 控制器校验 | 通过，1750 次 50 Hz 更新 |
| 源动作进度 | 0 → 263.52 帧 |

在仓库根目录、`humanoid_fabric` 环境中运行可视化：

```bash
python examples/test_g1.py --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --direct-wbc --duration 35 --show-isaac-window \
  --output-path acc_data/lafan_walk_preview.npz
```

关键修复是将软载荷分配目标限定在 `PREPARE_STEP`，使新支撑脚真正达到 80% 承重放行条件；保留足底力、捕获点、落地稳定时间门控。之前无载荷目标的运行完成第一步后，新支撑脚承重约 77%，第二次换脚超时。全程启用载荷目标的试验在起步出现腕部失控，因此本次没有将其设为全程默认；该模型响应偏差的根因尚未完全定位。

配套调整包括相对实测支撑脚的短步落点、独立于源动作时钟的摆腿与触地搜索、屈膝和 COM 高度参考，以及耦合脚底偏航摩擦约束。`--direct-wbc` 使用 QP 力矩加 500 Hz 速度阻尼，保留物理关节限位和力矩约束。详细机制见 `src/fabrics_sim/reference/README.md`。

结果文件：

- `acc_data/lafan_walk_phase_load_35s.npz`：完整状态、力、目标、阶段记录。
- `acc_data/lafan_walk_phase_load_35s.audit.json`：独立行走检查。
- `acc_data/lafan_walk_phase_load_35s.audit.png`：移动轨迹、承重和脚踝高度图。

重新生成检查和图：

```bash
python examples/analyze_reference_walk.py acc_data/lafan_walk_phase_load_35s.npz \
  --output acc_data/lafan_walk_phase_load_35s.audit.json \
  --plot acc_data/lafan_walk_phase_load_35s.audit.png
```

这是慢速、小步、平地的有限时长结果。由于步幅缩短、屈膝及源时钟重定时，原 PKL 保真度尚未通过：峰值关节 RMSE 0.5891 rad、峰值根位置误差 2.8188 m。不能据此声称精确跟踪整段 261 秒 PKL。参考/落地/步态检查 25 项、接触力矩锥 2 项、QP 14 项、物理任务 19 项测试通过。

以下为历史记录，保留用于对照。

## ReferenceManager 实验（2026-09-19）

已接入可选参考管理器：候选接触计划、承重确认、动作时钟减速、起步过渡，以及不依赖 PKL 时钟的落地轨迹。完整说明见 `src/fabrics_sim/reference/README.md`。

```bash
python examples/test_g1.py --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --duration 10 --show-isaac-window \
  --output-path acc_data/lafan_manager_preview.npz
```

当前代码完成 10 秒无窗口仿真，结果保存为 `acc_data/lafan_manager_landing_current_10s.npz`。**控制器校验通过，但连续行走未成功**：

- 5.22 秒进入左脚单支撑，5.72 秒开始落地过渡。
- 最后停在源动作第 130.51 帧，状态 `waiting_for_contact`，已等待 4.28 秒；未切换到右脚支撑。
- 右踝目标高度 0.035 m、实际高度 0.1123 m，右脚受力 0 N，右脚位置误差 0.1659 m。目标已下降，实际脚未跟上。
- 相对当前重定时、地面对齐的 PKL，最终关节 RMSE 0.1775 rad、根位置误差 0.2644 m。不能用较小的内部伺服误差替代这些指标。
- 10 项参考管理/落地轨迹测试通过，覆盖接触门控、事件前刹停、超时、时钟连续性及落地轨迹导数。

旧基线在 5.22 秒出现 QP 失败；本次运行更久，但源动作停滞，因此不代表跟踪质量已改善。对比数据见 `acc_data/lafan_reference_manager_comparison.json`。

当前脚部任务属于加权软目标，终态仍存在较大残差。下一步应验证落脚点与 COM 目标能否同时满足，并调整落地阶段的任务策略；本次没有证明单纯增加脚部权重就能解决问题。管理器仍为实验选项，默认关闭；完整连续步态尚未验证。

## 启动与参考时序修复（2026-09-18）

- 默认 `--ground-motion` 在生成仿真初始状态前，将整个参考轨迹沿 Z 平移，让初始姿态的最低脚底角点接触平地。使用与 QP 一致的脚底矩形和 0.035 m 深度；对 `walk1_g1.pkl` 平移约 -0.021464 m。保留所有关节角、朝向及相对轨迹，不修改 PKL，也不重置运行中的仿真状态。`--no-ground-motion` 可保留原绝对高度；该对齐用于平地起步，不表示支持跳跃或非平地初始化。
- 默认 `--motion-ramp-duration 1` 在预热后用 1 秒平滑提高播放速度，随后达到 `--motion-speed`。参考位置、速度、加速度使用同一个时钟；`0` 恢复即时起步。不会删除第一帧或跳过动作。
- 实测关节速度受扰后超出保守刹停包络时，按刹车预算逐步恢复，而不是强制 20 ms 内返回包络。下一步预测关节位置仍限制在真实限位的 margin 内，动力学/摩擦/力矩约束不变。这不是跨整个控制周期的物理安全保证，仍由仿真和跟踪校验检查。
- QP 错误现在同时报告 `bound_violation`，避免物理加速度边界失败被误认为只有力矩问题。

独立测试包括：全轨迹平移不改关节和原始数据、起步时钟连续性、刹车恢复，以及临近限位时硬位置约束优先。`walk1_g1.pkl` 的 3 秒起步测试通过，零持续振荡、零力矩饱和；20 秒验证在 5.22 秒出现后续 QP 失败，尚未通过连续步态验证，不能推断整段 261 秒动作都能跟踪。

## 当前默认任务（2026-09-18）

`examples/test_g1.py` 默认使用 `--task-profile dynamics-aware`，以物理任务请求直接构造 QP 跟踪目标：

- COM 控制平移和高度，身体任务控制朝向，不再同时要求骨盆 XYZ 严格跟随原 PKL。
- 支撑脚不参加摆动脚位姿跟踪，由已有接触约束处理；摆动脚跟踪参考位姿。
- COM 参考使用已有的速度/加速度受限平滑更新。
- 手部默认 `--hand-tracking position`：保留手的位置轨迹，关闭手部朝向任务。`pose` 恢复完整位姿，`off` 关闭手部任务；关节姿态软目标仍然存在。
- 动量、碰撞软任务及低权重关节姿态任务保留。任务权重是软偏好，不是严格分层优先级。
- 动力学、接触、摩擦和力矩校验不变。

可显式运行：

```bash
python examples/test_g1.py --motion-pkl retarget/lafan/walk1_g1.pkl \
  --task-profile dynamics-aware --hand-tracking position --duration 20
```

旧行为可用 `--task-profile legacy` 对照（`--hand-tracking` 只作用于 dynamics-aware）。以下历史诊断命令如需复现旧结果，应显式指定 legacy。

任务结构调整不代表 LAFAN 启动不可行问题已解决；必须以短仿真及完整跟踪的实际结果为准。

# G1 tracking experiments

Frequency is unchanged: Fabric, contact processing, QP and command integration
run at 50 Hz; explicit PD and PhysX stepping run at 500 Hz. PhysX drive gains
remain zero. The reference PKL does not overwrite integrated joint commands.

The joint posture task is configured in `src/fabrics_sim/fabric_params/g1_params.yaml`.
It is a soft Fabric task with zero reference-acceleration feedforward. A target
must be supplied to enable it; the example supplies PKL joint positions and
finite-differenced reference velocities (zero during warmup).

Explicit servo gains are bounded using initial effective inertia and a local
discrete-time coupling budget. This is not a proof of whole-body stability.
QP acceleration bounds constrain the integrated position and speed (4 rad/s
command ceiling), including braking room. There are no threshold-triggered
position/velocity resets. A slow, bounded acceleration-level back-calculation
term coordinates the command integrator with measured state BEFORE QP. The
solver's constraints take precedence. If no admissible command acceleration
exists, the run fails with a snapshot rather than silently clipping or resetting.

QP separates nominal inverse-dynamics feedforward from persistent explicit PD.
The next integrated target gives an affine feedback law
`tau_PD = servo_bias + servo_acc_gain * a_nominal_joint`. The QP computes its
local contact-consistent acceleration/wrench response and constrains TOTAL
effort and contact wrench. Feedforward does not subtract the current PD.
`acc`/`history_qdd_qp` is nominal acceleration for integration;
`predicted_acc`/`history_qdd_predicted` includes feedback at the solve snapshot.
These should not be confused when comparing against measured accelerations.

The prediction assumes the same rigid contacts and dynamics as the QP. It is
not a nonlinear rollout over the following 20 ms. The QP reserves 20% effort
margin; final total effort is still clipped at 500 Hz. The optional solver
`predicted_acc_limit` is tested but is not enabled by the example: adding the
same limits used for nominal acceleration made the static simulation worse.

Run the servo diagnostic first:

```bash
python examples/test_g1.py --motion-pkl retarget/output/B2A_131_02_stageii.pkl --fixed-joint-test --duration 1 --output-path /tmp/g1_servo.npz
```

This diagnostic holds initial joint targets and initial QP feedforward. It
intentionally bypasses command integration and is not motion tracking.

Next test the full controller with a static reference:

```bash
python examples/test_g1.py --motion-pkl retarget/output/B2A_131_02_stageii.pkl --motion-warmup-duration 5 --duration 2 --output-path /tmp/g1_hold.npz
```

Inspect `servo_*` arrays at 500 Hz, not only endpoint position errors. Their
order is `servo_joint_order`; `history_*` joint arrays use `final_joint_order`.
QP/command-bound failures save a `.failure.npz` snapshot and return failure.
Full-run validation includes peak sampled COM/servo errors, minimum COM height,
500 Hz sustained high-speed reversals, and per-joint saturation fractions.
Five consecutive speed sign reversals above 5 rad/s, or clipping a joint for
more than 5% of samples, fails servo validation. These are experiment acceptance
criteria, not robot specifications. Endpoint errors alone cannot pass the run.

Current limitation: the corrected controller still fails the two-second static
reference test. The latest slow back-calculation run had peak COM error 0.276 m
(limit 0.25), peak servo RMSE 0.218 rad, no sustained speed reversal, and no
torque clipping. Without back-calculation, COM error was smaller but servo error
grew beyond its limit. Thus integral coordination and balance still require
closed-loop validation/tuning. Do not interpret structural fixes or unit tests
as proof of stable motion tracking. Frequencies remain unchanged.

Contact-wrench follow-up: `build_single_support_wrench_constraints` now rotates
the friction/CoP inequalities from each foot frame in double support as well as
single support. Each environment is handled independently; a mixed batch no
longer drops double-support inequalities. Double support permits zero minimum
normal load on either foot, while single support retains the minimum load.
Rotation invariance and mixed-batch equivalence have regression tests (the full
20-test suite passes). With this correction, the same static-reference test
failed QP validation at 1.64 s after losing balance, saved as
`/tmp/B2A_wrench_frame_hold_2p0.failure.npz`. It did not improve stability.
The unresolved nominal-versus-predicted acceleration objective and the physical
execution response still need investigation before claiming PKL tracking.

The optional `--qp-track-predicted-base` experiment changes only the objective:
base rows track servo-inclusive predicted acceleration, while joint rows still
track nominal acceleration for integration. This avoids applying an all-joint
physical-acceleration objective that would cancel joint PD. Synthetic coupled
dynamics and zero-feedback equivalence tests pass, but the static simulation
failed earlier (0.50 s, contact residual 0.058 versus the 0.05 tolerance), with
larger motion errors. Therefore this option is OFF by default; it is not a fix
for tracking. The snapshot is `/tmp/B2A_predicted_base_hold_2p0.failure.npz`.
In the first 0.48 s of the preceding default run, the root-angular acceleration
RMSE against the reference was 0.665 for nominal acceleration versus 11.845 for
servo-inclusive prediction. Prediction versus measured 20 ms average was
16.425 rad/s^2. These compare an instantaneous model to an interval average,
so they indicate a mismatch to investigate, not proof of a dynamics-model bug.

### Free-flight isolation of the startup impulse

`examples/diagnose_g1_response.py` lifts the initial pose by 2 m and compares
single-physics-step acceleration to `solve(M, tau - h)`, including a small
torque excitation. Contact reporting identified an initial self-contact between
`left_wrist_yaw_link/collisions/left_rubber_hand/mesh` and
`left_hip_pitch_link/collisions/left_hip_pitch_link/mesh`, with separation
-0.001553 m. Initial joint limits are satisfied. With self-collisions enabled,
the first step has an approximately 1228 rad/s^2 acceleration mismatch. Disabling
self-collisions BEFORE physics initialization reduces the first-step maximum
error to 0.0000453. Changing the USD flag after physics initialization did not
remove the impulse; neither did stepping and reapplying initial state (reverted).

```bash
python examples/diagnose_g1_response.py --output-path /tmp/g1_contacts.npz
python examples/diagnose_g1_response.py --disable-self-collisions --output-path /tmp/g1_no_self.npz
```

This is evidence for an unmodeled self-contact, NOT a reason to disable collision
physics for tracking. The main example's `--disable-self-collisions` option is
diagnostic-only and cannot certify tracking. Its two-second static test still
failed at 0.98 s. Increasing `--servo-bandwidth` from the unchanged default 30 to
120 rad/s also failed (0.96 s with self-collisions enabled). No gain increase or
collision disabling was adopted as a default fix. Stable tracking remains
unachieved; both collision-feasible references and support-control execution
need further work. All frequencies and original PKL data remain unchanged.

### Matched model and explicit clearance-reference experiment

The main example now selects its Fabric URDF from `G1_USD_PATH.stem`, currently
`g1_29dof_rev_1_0`, instead of silently using `g1_29dof`. The old and new URDFs
differ in torso inertias and geometry (including a 9 mm waist-roll joint-origin
change). This is a model-consistency fix, not a gain adjustment.

`--left-shoulder-roll-offset 0.05` creates an explicitly modified reference in
memory, adding approximately 2.9 degrees of left shoulder abduction. The source
PKL remains unchanged; position commands still come from acceleration integration.
The offset defaults to zero, is checked against simulator shoulder limits over
the reference, and is saved in output metadata. It is NOT a general collision
avoidance algorithm or a certificate for every motion frame. In the lifted
initial-pose check, with self-collisions enabled, this offset eliminated the
reported hand/hip contact and reduced the first-step maximum acceleration error
to 0.0000341.

```bash
python examples/test_g1.py --motion-pkl retarget/output/B2A_131_02_stageii.pkl --motion-warmup-duration 5 --duration 2 --left-shoulder-roll-offset 0.05 --output-path /tmp/g1_matched_hold.npz
```

The matched-model/clearance run passed two seconds of static reference: peak COM
error 0.008779 m, peak servo RMSE 0.128893 rad, no sustained oscillation or torque
clipping. Extending static reference to five seconds failed QP validation at
4.02 s. Enabling the experimental predicted-base objective failed at 1.16 s;
it remains off. These outcomes do not establish stable tracking. Output now
labels validation scope (`static_reference`, `fixed_joint_servo`, or
`motion_segment`) so a static test is not presented as motion playback success.
The playback check (one-second warmup, requested three seconds total, same
0.05 rad offset and default base objective) failed at control time 2.12 s,
about 1.12 s into motion, after substantial lateral drift and loss of height.
The failure snapshot is `/tmp/B2A_rev_clearance_play3.failure.npz`.

### Command synchronization inside the QP

The command integrator and inverse-dynamics acceleration now have explicit
semantics. For the bounded synchronization term `c` (zero base rows), the QP
decision is `a_cmd`, and `a_nom = a_cmd - c`. Dynamics and rigid contact closure
constrain `a_nom`; joint command position/speed bounds constrain `a_cmd`. The
servo prediction uses the next integrated command, and total effort/wrench
constraints still include that feedback. The objective tracks `a_fabric + c`.
Previously, adding `c` only to the objective could project synchronization out
of constrained directions, allowing joint-command drift during static stance.
No command state is reset, and no synchronization acceleration is misrepresented
as a physical inverse-dynamics feedforward acceleration.

Saved `history_qdd_qp` is `a_cmd`; `history_qdd_nominal` is `a_nom`;
`history_command_acc_correction` is `c`; `history_qdd_predicted` is the physical
prediction including PD. The five-second matched-model/0.05-rad-clearance static
run passes: peak COM error 0.009392 m, peak servo RMSE 0.039728 rad, final servo
RMSE 0.013169 rad, no sustained oscillation or saturation. Every saved position
and velocity command exactly matches semi-implicit integration at 0.02 s; the
`a_cmd = a_nom + c` identity holds within 2.4e-7. This establishes a five-second
static result only, not full motion tracking.

Motion follow-up: the same 0.05 rad offset failed at control time 1.78 s
(0.78 s after playback started). `--report-self-contacts` records first
penetrating self-contact per collider pair, including time and separation, in
the log and `self_contact_events` NPZ field. It found renewed left hand/hip
contact at 1.476 s. A separate 0.15 rad offset experiment avoided that pair but
failed at 2.20 s, with other self-contacts appearing after lateral instability.
Thus a constant shoulder offset is not sufficient collision avoidance.

The 0.15 rad run released the left foot at 1.32 s while measured loads were
172 N left / 183 N right. An optional `--require-support-load` experiment now
requires the requested support foot to carry at least 80% of positive vertical
load for the configured confirmation count. Negative vertical forces are no
longer interpreted as support via absolute value. The gate has regression
coverage, but the three-second motion trial remained in double support and
failed at 1.58 s: the reference clock continued advancing toward swing poses.
The gate is therefore OFF by default pending phase-clock coordination, not
advertised as a completed tracking fix. No experimental shoulder offset is
enabled by default. Failure snapshots now detach synchronization tensors before
NumPy conversion so the physical failure is not hidden by a logging exception.

### Contact-aware phase-clock experiment

`--require-support-load` now couples its load gate to the reference clock.
Before entering an unprepared single-support phase, the reference stops before
that phase; the pending contact mode is still sent to the state machine so load
transfer can be prepared. While held, posture reference velocity/acceleration
are zero and COM preparation targets the pending support, not a later opposite
phase. Resuming advances by the normal per-update increment rather than catching
up to wall time. Intermediate phase labels are inspected even when one update
would skip several reference frames. Double-support frames remain reachable to
allow swing-foot touchdown. Neither controller nor physics frequency changes.

The five-second requested trial with 0.15 rad shoulder offset held at frame 32,
but failed QP validation at 1.56 s during load transfer, before resuming. Its
snapshot is `/tmp/B2A_phase_clock5.failure.npz`. Thus clock coordination removes
the advancing-reference conflict but does not establish a stable weight shift.
The option remains OFF by default. `history_phase_held` and
`phase_hold_duration` expose pauses; a static or paused segment must not be
interpreted as completion of the PKL. The 27-test suite covers hold/resume,
non-skipping, fractional-frame monotonicity and touchdown progression.

### Isolated load transfer: current passing configuration

The current experimental balance path combines:

- `--align-balance-pelvis`: pelvis and COM share horizontal displacement,
  velocity and acceleration adjustments; pelvis height/orientation stay at the
  reference values.
- `--track-support-wrench`: replace zero-wrench regularization by a soft target
  for TOTAL contact wrench, including servo feedback. A level-floor COM/ZMP
  approximation distributes the desired force between the feet. This is not a
  hard force command, and full dynamics/friction/effort constraints still apply.
  A bounded second-order COM reference limits each component to 0.2 m/s and
  0.5 m/s^2, with position/velocity/acceleration kept integrally consistent.
  Current support uses planted contact anchors instead of sliding PKL support
  positions; unplanted future support still uses its motion reference.
- `--weight-transfer-test`: hold the selected starting pose, wait through warmup,
  then smoothly request a three-second rightward load transfer without playing
  the PKL or deliberately releasing either foot.

```bash
python examples/test_g1.py --motion-pkl retarget/output/B2A_131_02_stageii.pkl --weight-transfer-test --duration 8 --left-shoulder-roll-offset 0.15 --align-balance-pelvis --track-support-wrench --report-self-contacts --output-path /tmp/g1_transfer8.npz
```

The eight-second run `/tmp/B2A_bounded_transfer8.npz` passes: final right load
fraction 0.881, peak COM error 0.027374 m, peak servo RMSE 0.039064 rad, no
sustained oscillation or torque clipping. Transfer acceptance additionally
requires >=80% right load averaged over the last ten control updates, final
COM error <=0.03 m, a completed transfer ramp, and double support throughout.
It cannot pass just by remaining in the initial stance. The original PKL is
unchanged, and the explicit 0.15-rad clearance variant is not enabled by default.

This does NOT establish full PKL tracking. With phase gating enabled, the
unanchored trial released left support at 1.82 s with measured loads
42.7 N / 248.5 N and later changed support, but failed at 3.32 s. The anchored
variant failed at 2.52 s while held at frame 32. Raising servo bandwidth to
120 rad/s failed at 2.70 s; adding `--fast-command-sync` (20/8/30 instead of
1/2/5 synchronization gains/cap) failed at 2.78 s. Neither is adopted as a
default. Removing Jdot filtering (`--jdot-filter-alpha 0`) and adding rotational
contact stabilization also failed independent transfer tests and remain off
or at their old defaults.

### Independent QP failure replay

Failure snapshots now contain the exact `problem_*` matrices and primal
solution. The offline helper uses the installed OSQP/scipy packages:

```bash
python examples/compare_g1_qp.py /tmp/B2A_qp_replay_source.failure.npz --diagnose-constraints
```

OSQP reported primal infeasibility for the 2.70 s snapshot, rather than finding
a solution that the original solver merely missed. Removing command-acceleration
bounds, contact-acceleration equalities, or effort inequalities separately made
relaxed problems solvable. These relaxations are DIAGNOSTIC ONLY and are never
applied to live control. The unconstrained command acceleration request conflicts
with bounds on left ankle roll and both waist roll/pitch coordinates. Thus
single-support reference feasibility and recovery remain unresolved.

`controller_options` stores the full CLI configuration in new snapshots;
additional histories include total/target contact wrench and pelvis target.
The 32-test suite covers reference consistency/bounds, anchored support targets,
and total-wrench objective semantics. Unit tests do not replace closed-loop
motion validation.

### Consistent planted-foot Fabric references

The experimental `--track-support-wrench` path now refreshes touchdown anchors
before evaluating Fabric. Active feet use the same world-frame position and
orientation anchors as the contact constraints, with zero reference twist and
acceleration. Swing-foot pose and derivatives remain unchanged. Previously,
Fabric continued attracting planted feet toward moving PKL targets while the
QP constrained them to remain planted. Tests cover mixed left/right support
batches, optional derivatives, and nonmutation of input references.

The comparable 50/500 Hz, bandwidth-30 run with `--require-support-load`,
`--align-balance-pelvis`, and a 0.15-rad left shoulder clearance offset saved
`/tmp/B2A_stance_targets_motion5.failure.npz`. Right support began at 1.80 s
(left/right loads 22/282 N), double support at 2.26 s, and left support at
2.48 s (202/0 N). This improves on the former frame-32 stall but is NOT stable
walking: COM error already reached 0.154 m at 2.40 s, before the first recorded
self-collision at 2.922 s. The robot subsequently fell and QP validation failed
at 3.98 s. Thus self-collision is not the initial cause of this trial's loss of
tracking; alternating-support balance/reference feasibility remains unresolved.
The change is experimental, not enabled on the default path. No frequencies,
PhysX gains, integration semantics, or source PKL values were changed.

The eight-second isolated-transfer regression after this change also passes
(`/tmp/B2A_stance_targets_transfer8.npz`): right load fraction 0.881, peak COM
error 0.027374 m, peak servo RMSE 0.039064 rad, no sustained oscillation or
torque clipping. This preserves the previous transfer result, not full tracking.
In the motion trial, actual COM y moved from -0.044 m at 2.00 s to +0.058 m
at 2.40 s while target y remained near -0.09 m. At left-support entry (2.48 s)
target y was still -0.089 m although the left foot was at +0.205 m. The next
investigation should address this delayed balance reference and single-support
COM response; force-based support readiness alone does not establish balance.

### Isolated single-support preparation

`--single-support-test left|right` freezes the selected PKL frame. A three-second
quintic shifts the raw XY target toward 95% support ankle / 5% initial COM,
followed by the bounded reference generator. Requires `--align-balance-pelvis`
and `--track-support-wrench`; excludes the other isolated tests.

Release requires ten consecutive 50 Hz updates with confirmed support contact,
>=90% load and >=20 N, COM error <=3 cm, COM speed <=5 cm/s, and support-foot
linear speed <=2 cm/s. The simplified capture point must lie within an
axis-aligned +/-6 cm X, +/-4 cm Y window about the initial support ankle.
This diagnostic window is NOT a rotated sole polygon or a dynamic stability
certificate. On release a one-second quintic lifts the swing foot 4 cm.
Passing requires >=3 s after release and a final second with the requested
mode, <=3 cm COM error, >=2.5 cm actual foot rise and <10 N swing load, plus
existing trajectory/servo checks. Never releasing cannot pass; test duration
bounds the wait, without forcing a switch. Frequencies and integration remain
unchanged. The gate regression brings the suite to 33 tests.

```bash
python examples/test_g1.py --motion-pkl retarget/output/B2A_131_02_stageii.pkl --single-support-test right --duration 10 --left-shoulder-roll-offset 0.15 --align-balance-pelvis --track-support-wrench --report-self-contacts --output-path /tmp/B2A_single_right10.npz
```

Right trial failed QP validation at 6.48 s without releasing. At 4 s COM error
was about 5.2 cm and loads 37/286 N; actual COM subsequently moved away from
the held target. First self-collision was at 5.846 s, after tracking degraded.
Snapshot: `/tmp/B2A_single_right10.failure.npz`. This more demanding transfer
is distinct from the passing 75%-biased transfer test; it exposes preparation-
stage tracking limitations and does not establish single-support stability.

The identical left-side trial (`/tmp/B2A_single_left10.npz`) completed ten
seconds but correctly failed single-support acceptance: no release occurred.
Late COM error plateaued near 3.05 cm despite left/right loads around 302/22 N.
The gate thresholds were not relaxed to manufacture a pass. Both experiments
therefore test preparation, not successful single-support holds. Next isolate
COM/pelvis versus frozen posture/hand task competition and execution response.

### Layer-by-layer COM acceleration diagnostics

Add `--diagnose-com` to a PKL run. It does not change control objectives or
gains. `history_com_acc_*` stores `requested`, `fabric`, `projected`,
`qp_nominal` (physical acceleration before explicit PD), `qp_predicted`
(after predicted PD), and `observed`. All are world-frame COM accelerations.
Generalized accelerations are mapped using `J*a + Jdot*v`, with the same
1e-5 directional difference as Fabric. The task request uses the same clipped
PD computation as the live attractor. Metric/toggle and curvature are logged.

Observed acceleration is the forward-interval difference of URDF COM velocity
computed from measured q/v, not a direct PhysX COM sensor. It is attached to
the prediction at the START of the interval. Unobserved final intervals have
NaNs and `history_com_observed_valid=False`. Observations are interval averages;
predictions are instantaneous, so contact impulses and changing feedback must
be considered before assigning fault. Failed QP snapshots additionally include
the failing tick as `diagnostic_*`, without treating it as an executed command.

```bash
python examples/analyze_com_diagnostics.py /tmp/B2A_com_layers_right7.failure.npz --start 3 --end 5
```

The report prints per-stage mean XYZ, RMS acceleration norms, and adjacent-
stage differences. Differences locate disagreement, but do not by themselves
identify a conflicting task or certify the physical model. The 36 tests cover
curvature mapping, unchanged control parameters, matching clipped requests,
and temporal alignment of observations.

Right-side diagnostic replay saved `/tmp/B2A_com_layers_right7.failure.npz`
and failed at 6.48 s, matching the previous run. All 324 saved q, qd, and QP
command-acceleration samples are exactly equal to the diagnostic-disabled
baseline; the instrumentation did not alter this trajectory. The failing
state completes the preceding executed interval, giving 324 valid observations.

Over [3,5) s the mean Y accelerations (m/s^2) were: requested -1.65864,
Fabric -0.73136, projected -0.81531, QP nominal +0.03627, QP predicted
-0.26378, observed +0.04940. Adjacent-stage vector RMS differences were
1.09062, 0.85953, 1.06496, 0.34882, 0.37373 respectively. Thus disagreement
exists at task combination, constraint processing, and predicted execution;
this is not evidence that one gain or one task alone is responsible. These
are interval statistics, not an assertion that all six accelerations must be
identical. In particular, nominal-to-predicted differences include explicit PD
by design. Next use controlled task ablations and inspect active QP constraints,
while checking the feedback response model against short physics intervals.

### Highest-weight COM motion task

At user request, the default G1 COM metric changes from [25,25,8] to
[250,250,80]. Every axis now exceeds the other motion leaf-task weights.
Stiffness, damping, acceleration limits, control rates and QP constraints are
unchanged. This affects all consumers of `g1_params.yaml`, not just this test.
It is weighted Fabric composition, NOT strict hierarchical priority: Jacobian
pullback and subsequent constrained QP can still compromise COM tracking.
The previous experiment results above used the old metric. The suite now has
37 tests, including a check of leaf-weight ordering and unchanged COM stiffness.

The nine-second right single-support regression passes with this metric:
`/tmp/B2A_com_high_weight_right9.npz` (same isolated-test command as above,
with `--duration 9 --diagnose-com`). Gate release occurred at 5.10 s; the
post-release duration exceeds three seconds and final-second COM/clearance/
load checks pass. Peak COM error 0.037776 m, peak servo RMSE 0.039060 rad,
no sustained oscillation or torque saturation. The test still uses an explicit
0.15-rad left shoulder offset and a frozen motion frame. This establishes only
this right-support diagnostic, NOT left support, alternating steps or full PKL
tracking. An earlier seven-second run exposed a numpy-bool to torch-bool
acceptance conversion error; the result is now explicitly converted to Python
bool before updating the tensor validation flag.

Follow-up validation with unchanged high COM metric:

- Left isolated support (`/tmp/B2A_com_high_weight_left10.failure.npz`):
  no gated release; QP failure at 6.46 s. In [3,4.5) s, before first recorded
  self-collision at 4.65 s, mean Y acceleration was requested +2.07948,
  Fabric +1.95393, projected +2.14904, QP nominal -0.05487, QP predicted
  +0.16500, observed -0.07952 m/s^2. Thus Fabric largely preserves the request
  here, but subsequent constrained/physical response does not. This does not
  identify which particular constraint or response-model assumption dominates.
- Continuous PKL with `--require-support-load --duration 5 --diagnose-com`
  and the same balance/clearance flags
  (`/tmp/B2A_com_high_weight_motion5.failure.npz`): right support at 1.76 s,
  double support at 2.26 s, no commanded left single support, held at frame 90,
  QP failure at 3.02 s. First self-collision at 1.894 s. It does not improve
  full tracking over the earlier lower-weight stance-target trial (3.98 s
  failure), though failure time alone is not a complete performance metric.

The user-requested high soft weight remains configured; it is not certified
for continuous tracking. The right isolated pass must not be generalized to
both legs. No additional gain increases, relaxed gates, disabled collisions or
frequency changes were used for these follow-up runs. All 37 unit tests pass.

### QP attribution at pre-collision states (diagnosis, no controller fix)

`--diagnose-qp-time 3.5 --diagnose-qp-time 4 --diagnose-com` saves complete
QP/input snapshots at those ticks and one 2 ms physical response per snapshot.
No state reset, gain modification or live constraint relaxation occurs.
The left-support preparation replay uses the same high-weight configuration,
with duration 4.1 s and output `/tmp/B2A_qp_response_left4.npz`. As expected,
the short preparation-only run fails the single-support acceptance criterion.

```bash
python examples/attribute_com_qp.py /tmp/B2A_qp_response_left4.qp_0175.npz --history /tmp/B2A_qp_response_left4.npz --response /tmp/B2A_qp_response_left4.response_0175.npz
```

At 3.5 s, independent OSQP replay predicts COM Y acceleration +0.16001,
close to controller +0.16274. Removing only command-acceleration bounds in
offline optimization changes it to +2.07279. Removing effort or tangential
friction constraints makes no material difference; removing CoP constraints
changes it only to +0.16228. The dominant active acceleration bound is
coordinate 17, right_ankle_roll_joint: upper acceleration approximately
-0.000476 rad/s^2, with integrated position already at +0.26179972 rad.
Both feet's negative roll-moment bounds and the right positive pitch-moment
bound are also active, but their removal has little effect in this state.
Dual magnitudes across differently scaled rows alone are not causal evidence;
the ablations provide the sensitivity comparison.

Importantly, exact requested COM acceleration remains feasible in the SAME
linear QP with ALL original constraints: an additional three-row COM equality
solves with constraint violation <4e-13 at 3.5 s. It raises the original
objective from about -2.25 to +572.54. Thus the current generalized-acceleration
objective chooses a different compromise when the ankle is limited. Fabric's
high COM metric is not carried into the final QP as COM task priority.
This is model feasibility only, not a claim that the physical robot could
execute that solution; it was NOT sent to physics. The 4 s snapshot confirms
the same qualitative result.

Response checks exclude a simple effort-command mismatch: first applied torque
matches QP total torque to <=2e-6 Nm. Over the first 20 ms the applied torque
changes by only about 0.014 Nm joint RMS. Recomputing the held model using the
average actual torque still predicts COM Y +0.157 at 3.5 s, versus observed
interval -0.0804. Already over the first 2 ms the full COM prediction differs:
Y +0.1627 predicted, +0.0204 observed. At 4 s these are +0.3338 and -0.1204.

The measured feet have nonzero angular accelerations (roll approximately
+3.67/-5.10 rad/s^2 at 3.5 s and +7.88/+17.53 at 4 s), unlike a perfectly
fixed six-axis contact. A least-squares inverse-dynamics fit with two foot
wrenches has generalized residual RMS 0.297 / 0.368 at the two snapshots.
Adding a reaction along the nearly upper-limited right ankle-roll coordinate
reduces these to 0.0336 / 0.0394, with fitted reactions -4.35 / -5.40 Nm.
This strongly supports missing joint-stop reaction plus non-rigid contact
response as contributors. The stop reaction is inferred, NOT directly sensed;
finite-step numerical/contact effects remain. Both snapshots precede the
first reported self-collision (4.65 s).

Next fixes require separate authorization: preserve COM priority explicitly in
the final QP, and address physical joint-limit margin/response-model validity
instead of deleting acceleration, effort, or friction constraints.

## AMP walking progress (2026-11-02)

`examples/run_amp_four_steps.sh` runs the current AMP profile at 500 Hz with
DAQP. The original input is `retarget/amp_reference/amp_walk_1ms.pkl`; its
metadata specifies **200 Hz**, despite the filename. The prepared companion is
`acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl`. Preparation and FK
checks are recorded in `acc_data/amp_walk_20261102/input_report.json` and
`flow_fixes/native_contacts.json`; contact labels are inferred, not ground truth.

```bash
conda run -n humanoid_fabric bash examples/run_amp_four_steps.sh
```

The reproducible result is **four measured alternating landings, but strict
PKL tracking still fails**. The initial landing completes a partial swing
already present at frame zero. Confirmation times are 0.188, 0.416, 0.876 and
1.362 s; landing errors are 1.51, 2.65, 2.02 and 2.70 cm. Every confirmation
requires real ground load, a preceding flight observation with at least 2 cm
native-collider clearance, and 50 ms within the 4 cm landing tolerance.

Original-PKL joint RMSE peaks at **0.21519 rad**, above the unchanged 0.2 rad
criterion. Root position/orientation peaks are 0.07181 m / 0.22318 rad. The
largest joint error occurs after the third landing, around 1.014 s, involving
the support knee and swing ankle. Two runs produced identical recorded poses.
The process intentionally returns failure while strict fidelity fails. This is
neither full-clip tracking nor a stationary final-support demonstration.

The profile uses strong-impact contact acquisition and a native-geometry swing
clearance correction. Leg IK preserves the pelvis, stance foot, upper body and
original landing endpoints, then one interpolator generates unified q/v/a.
Only control references are adapted; the original ground-aligned PKL remains
the fidelity target. The original file is unchanged. Physical QP effort,
friction and dynamics constraints remain active. Failed independent COM,
clock-retiming and smoothing experiments were archived rather than retained
in the active controller path.

Records: `acc_data/amp_walk_20261102/four_steps.npz`, `four_steps.steps.json`,
`four_steps.png`, and `tracking_progress.json`. To regenerate the prepared
companion in this workspace, run `acc_data/amp_walk_20261102/prepare.py`, then
`acc_data/amp_walk_20261102/flow_fixes/prepare_native_contacts.py` with
`PYTHONPATH=src:examples` in the same environment.

## AMP full-clip attempt (2026-11-02)

Run the entire original AMP clip with:

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_full_motion.sh
```

The runner requests all 2000 frames (9.995 s at the recorded 200 Hz), using
`--track-full-motion` instead of the four-landing stop. Add
`--show-isaac-window --keep-isaac-window` for a visible run; a failed physical
QP still aborts the run and writes diagnostics. This mode is implemented and
exercised, but **full-length physical tracking has not succeeded**.

All 29 completed source swing events are now planned and checked. The last
single-support segment, frames 1986–1999, has no recorded landing and is not
counted as a completed step. The original terminal velocity is preserved;
reaching the last sample does not claim stationary support. The full-motion
pass requires source coverage, physical controller checks, unchanged PKL
fidelity thresholds, and measured flight/load/landing evidence for every
completed source event.

The control-reference clearance correction now covers all completed arcs.
Seven arcs needed a smaller lift target to obtain converged, joint-bounded IK.
A bisection search keeps one smooth bump amplitude per arc and accepts only
solutions satisfying the original IK tolerances. This is a local numerical
feasibility check, not proof of global IK infeasibility or dynamic feasibility.
Original source poses remain the audit target, and the source file is unchanged.

The default full-length attempt reproduced the four previous landings, then
failed physical QP validation at **2.410 s / source frame 482**. The last valid
recorded sample is frame 481.6, about 24.1% through the source timeline. Step 5
failed its landing test: after lateral COM drift had accumulated during the
preceding support transition, the left foot landed late and roughly 9.5 cm
from its original landing goal. Leg self-contact followed at about 2.12 s.
This is a failed tracking attempt; continuing the source clock is not success.

| Diagnostic configuration | QP failure time | Verified landings |
| --- | ---: | ---: |
| Existing controller, full-clip reference | 2.410 s | 4 / 29 |
| Horizontal COM metric ×4 | 1.232 s | 1 / 29 |
| Error-dependent horizontal COM metric | 2.656 s | 4 / 29 |
| Landing-orientation priority | 0.920 s | 1 / 29 |

The unsuccessful controller changes are archived in
`acc_data/amp_walk_20261102/full_clip/`; they are not enabled in the runner or
retained as new controller tuning options. The default record is
`full_clip/full_motion.failure.npz`, with `.steps.json`, `.tracking.json`,
`full_motion_diagnostics.png`, and `progress.json` alongside it.

A separate **60-frame endpoint smoke test** reached its last sample with one
verified landing, valid physical checks, and PKL fidelity within tolerance;
its `final_support_settled` remains false. It validates moving-endpoint control
flow only and is not evidence of tracking the entire 2000-frame AMP motion.

Validation: 484 unit tests passed (`full_clip/regression.log`), plus the Isaac
Sim endpoint smoke test. These checks do not certify the failed full clip.


## Continued contact-switch diagnosis (2026-11-02)

Dynamic-start tracking now passes the measured **ground** normal forces into
`update_contact_mode_state_machine` even without `--require-support-load`.
That older option also gates the source clock and is incompatible with the
dynamic-start profile. The missing force input had disabled the existing early
loaded-contact override and measured-load transfer gate. In the first 1.7 s,
samples where a substantially loaded foot was omitted from the commanded
contact mode dropped from 5 to 0. The source clock and reference remain unchanged.

The current default full run still verifies only **4 / 29 landings**, at
0.188, 0.404, 0.908, and 1.356 s. QP validation fails at **2.418 s**, source frame
483.6. Joint RMSE peaks at 0.21030 rad through the fourth verified landing
(previously 0.21519), still above the unchanged 0.2 rad fidelity threshold.
Neither the first four steps nor the full clip is a strict tracking pass.

Further physical trials tested lateral COM priority, posture/joint-limit
priority, unloading timing, sole approach constraints, additional PhysX
iterations, smaller physics timesteps, and measured-height swing correction.
None verified a fifth landing. They remain diagnostic scripts under
`acc_data/amp_walk_20261102/continuation/` and are not active controller changes.
More PhysX velocity iterations reduced lateral drift but exposed early swing
scuffing after root-height loss; smaller timesteps alone worsened tracking.
These comparisons demonstrate sensitivity to contact dynamics, not a verified
single cause or a successful fix for the fifth step.

The retained run is `continuation/measured_load.failure.npz`; its
`.steps.json` and `.tracking.json` include the failure and original-source
fidelity checks. `continuation/progress.json` records all trials and validation;
`audit_runs.py` regenerates it. `tracking_comparison.png` plots root lateral,
root height, and original-source joint errors. The original PKL SHA-256 remains
`a870f6642380a5d7e23ae34fbc8a9614bf4164d6a9ce0a26432ae35f3401b8c4`.

Post-fix validation: 484 unit tests passed in 94.936 s
(`continuation/regression.log`). Isaac Sim full-motion attempts remain failed.


## Five landing events reproduced; stability still fails

The continued experiments are recorded in
`acc_data/amp_walk_20261102/impact_followup/progress.json`. A reference-only
pelvis lowering of 2.5 cm, with bounded leg IK preserving both foot positions,
produced **five measured landing events**. The fifth was confirmed at
**1.864 s**, with **3.1235 cm** landing-position error and **243.4 N** measured
normal load. An independent 1.87 s rerun reproduced all 935 recorded states and
all five landing events exactly (`five_landings_reproduction.json`).

This is **not stable tracking of five steps**: at that endpoint, the original
PKL joint RMSE reaches 0.3435 rad, root position error 0.3057 m, and root
orientation error 1.3883 rad. The full attempt loses balance and fails WBC
validation at 2.226 s, before the sixth source landing. The original fidelity
thresholds and landing checks remain unchanged. The geometric adaptation
starts at source frame 184, ramps over 60 frames, and returns to the source
between frames 600 and 700. Its maximum allowed foot orientation adjustment
is 0.01 rad; the accepted trajectory used only 0.001661 rad. Original landing
positions and the original PKL remain the audit targets.

Reproduce the experimental full attempt in this workspace:

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash acc_data/amp_walk_20261102/impact_followup/run_crouch.sh \
  --output-path acc_data/amp_walk_20261102/impact_followup/crouch_replay.npz
```

It is expected to fail the full tracking gate. The experimental reference and
controller profiles are archived rather than enabled in the default runner.
`five_landings_diagnostics.png` shows the growing body errors, including at the
fifth landing; a larger landing count alone is not a stability improvement.

One diagnostic bug was fixed in the shared code: `com_acc_requested` now reads
from the actual WBC task request. Previously it recomputed the legacy clipped
PD request, which disagreed with preserved feedforward or predictive feedback.
`com_request_source` identifies the source in new records. Two captured QP
states confirm exact equality with the actual WBC request, and a full default
rerun reproduces the prior control trajectory exactly. Failed QP runs now also
print the verified landing count and PKL-fidelity status alongside the error.

Validation after the diagnostic fix: 485 unit tests passed in 94.588 s
(`impact_followup/regression.log`), plus the unchanged default Isaac trajectory
and exact reproduction of the five-landing experimental prefix.

## AMP 连续跟踪：落地参考审计与六次落地实验（2026-11-02）

`amp_walk_1ms.pkl` 的完整物理跟踪仍未通过。新实验按同一源帧的原始足部轨迹
确认到第六次落地，时间为 2.482 s；该时刻关节 RMSE 约 0.219 rad、根位置误差
0.085 m、根姿态误差 0.172 rad。到该时刻的全过程峰值分别为 0.279 rad、
0.131 m、0.440 rad，仍超过原有 0.2 rad / 0.2 m / 0.35 rad 门槛。
不能将六次落地事件写成“六步严格跟踪通过”。

修正了连续 tracking 的落地验收参考：原先把接触开始时的脚踝 link 位置固定为
后续 50 ms 的目标。但第 352 帧开始触地后，源脚踝在足部放平期间仍前移约
5.6 cm；原始参考自身在旧目标的 4 cm 范围内最多连续保持 22 ms，无法满足
50 ms 条件。现在连续模式用原始、地面对齐 PKL 在同一源帧的足部 FK 位置作为
落地目标，记录 `history_source_feet_reference`，并在计划中声明
`landing_goal_policy: source_pose`。该参考来自 `audit_reference_q`，不使用
经过抬脚修正的控制参考或实测位置作为目标。原接触开始位置仍随每次落地记录保存，
报告中的 `fixed_contact_onset_audit` 同时保留旧判据结果；旧记录未被改写。

抬脚净空 2 cm、支撑载荷 100 N、摆动脚卸载条件、落地载荷 20 N、位置误差
4 cm、连续保持 50 ms、源相位窗口及全身保真度门槛均不变。缺少原始足部参考
记录时，声明使用新判据的离线审计会拒绝，不能悄悄用控制目标或固定点替代。
`next_step_diagnostics` 现在明确报告下一步缺少飞行、承重、落点或连续保持证据。

候选复现入口（实验配置，未替换默认控制参数）：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash acc_data/amp_walk_20261102/priority_followup/run_landing_tail.sh \
  --duration 10.1 \
  --output-path acc_data/amp_walk_20261102/priority_followup/new_run.npz
```

实验将卸载力目标以 0.05 权重参与 0.92 s 之后的前一级 QP，并在第五次摆腿中
加入经过关节限位 IK 检查的 C2 净空修正，结束于第 362 帧。IK 将请求的 8.5 cm
中点净空限制到可行值；相对卸载对照组，额外关节参考变化最大仅 0.00478 rad，
尚不能证明这一候选对微小变化稳健。没有改机器人状态、原 PKL、碰撞或硬力矩约束。
对应修改保留在实验目录内。

完整尝试在第七次落地失败：已有足部离地证据，但承重时距同帧源足部位置最小
仍为 0.2003 m。约 3.2 s 后失稳，4.982 s 时 QP 验证失败；后一个时间不能当作
稳定跟踪时长。旧固定点判据在这条记录上计 4 次，新源轨迹判据计 6 次。

验证：2.55 s 的独立重复运行中 1275 个 q/v 样本逐位相同；接入新审计后的
整段尝试与接入前记录的前 4 s、2000 个 q/v 样本也逐位相同，表明验收修正
未改变控制轨迹。全部 490 项回归测试通过。原 PKL 的 SHA-256 保持
`a870f6642380a5d7e23ae34fbc8a9614bf4164d6a9ce0a26432ae35f3401b8c4`。

证据：`acc_data/amp_walk_20261102/priority_followup/progress.json`、
`verified_reproduction.json`、`six_landing_fidelity.json`、
`landing_tail_verified.steps.json`、`landing_tail_verified.tracking.json`、
`source_landing_path_audit.json` 和 `regression.log`。
可视化为同目录 `tracking_comparison.png` 与 `landing_goal_audit.png`。

后续对第七次及之后的落地过渡，按原足部速度降至 0.2 m/s 以下的源帧选择
净空修正终点，且不跨过下一次原计划抬脚。`later_tails` 实跑仍只确认 6 次落地；
第七次承重落点误差最小降到 0.1701 m，仍未通过 4 cm 条件。其 5.606 s 的
QP 失败发生在倒地之后，不能作为稳定时长。这一候选也仅保留在实验目录中。

## AMP 连续跟踪：第七次落地与关节制动接口修复（2026-11-02，后续）

`contact_clock_followup/vertical_feedback_verified.failure.npz` 连续确认 7 次落地，
第七次在 2.906 s、源帧 581.2 确认，足部位置误差 0.037776 m、落地载荷
195.09 N，之前合格离地段的最大原生碰撞几何净空为 0.085852 m。
两次从源帧 0 开始的独立实跑，1764 个 q/v 样本及落地报告逐位一致；与之前
`later_tails` 六次落地候选相比，前 2.44 s 的 1221 个 q/v 样本也逐位一致。

这仍不是七步严格跟踪成功，更不是完整 PKL 跟踪成功。第七次落地时关节 RMSE
为 0.2450 rad、根位置误差 0.07355 m、根姿态误差 0.27796 rad；截至该时刻的
峰值分别为 0.27931 rad、0.13137 m、0.44014 rad，关节和根姿态仍超出原阈值。
完整 10.1 s 尝试在第八步失稳，3.528 s 时 QP 验证失败；失败时间不表示稳定
跟踪时长。第八步尚缺合格离地证据，最大卸载净空仅 0.018997 m，承重时足部
位置误差最小仍有 0.66751 m。

新的实验控制在 2.44 s 起，用 20 ms 的 C2 平滑过渡，将摆腿竖直反馈独立限制在
±40 m/s²，并把竖直位置任务权重增至 4 倍。原有 XYZ 向量限幅会让较大的水平
追踪误差占用竖直反馈额度；QP 快照还显示旧方案在 2.45 s 请求向上 14.46 m/s²，
实际预测仅 5.82 m/s²。实验保留统一 q/v/a 的前馈，只调整反馈与软任务权重。
时间门控用于定位问题，尚未做成通用行走策略，实验入口未替换默认控制配置。

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash acc_data/amp_walk_20261102/contact_clock_followup/run_vertical_feedback.sh \
  --duration 10.1 \
  --output-path acc_data/amp_walk_20261102/contact_clock_followup/new_run.npz
```

同时排查了连续参考时钟和卸载策略。仅在支撑不匹配时平滑减速的试验也曾计到
7 次落地，但第七次确认时根位置误差已达 0.474 m，不作为改进候选。双支撑触地
调时没有改善稳定性。第七步确认前加强卸载会把有效落地保持时间降至 28 ms，
不满足原 50 ms 门槛；等待第七次落地后再加强卸载能保留 7 次确认，但第八步
仍失败。所有失败试验保留在 `progress.json`，没有降低飞行、落地或保真度门槛。

另修复 `AccelerationQP.py` 的接口冲突：`--acceleration-and-braking-limits`
生成有限关节边界和无界浮动基座，但此前 QP 拒绝无穷边界。现在允许下界为
`-inf`、上界为 `+inf`，按 batch 将缺失的一侧编码为有限的 `0 <= 0` 行；仍拒绝
NaN、反向无穷和上下界倒置，有限边界及含伺服/同步补偿的物理加速度约束不变。
启用关节制动、未启用全身预测加速度限幅的 0.2 s Isaac 检查已通过控制器验证，
只用于验证接口可运行，不算完整轨迹通过。

证据位于 `acc_data/amp_walk_20261102/contact_clock_followup/`：
`seven_landing_reproduction.json`、`unchanged_prefix.json`、
`seventh_step_diagnostics.json`、`swing_task_audit.json`、`progress.json`，
以及 `seventh_step_comparison.png`。原始 PKL 的 SHA-256 保持不变。

验证：全部 492 项回归测试通过（94.582 s，`regression.log`），另有 4 项
平滑参考时钟检查通过（`clock_tests.log`）；关节边界相关 20 项检查通过
（`bounds_tests.log`）。七次落地实验的复跑状态
与接口修复前记录逐位相同。


## AMP 全段尝试：五次严格落地验证（2026-11-04）

当前实验在 `acc_data/amp_walk_20261103/feedback_followup/`。`leg150_vertical155`
从源帧 0 连续验证五次落地，第五次在 1.898 s、源帧 379.6，原始同帧足部位置
误差 0.026118 m、承重 153.15 N；保持原 2 cm 离地、4 cm 落地误差和 50 ms
持续承重条件。截至第五次确认，原始 PKL 关节 RMSE、根位置和根姿态误差峰值
分别为 0.199303 rad、0.148771 m、0.327794 rad，均未超出原阈值。

这还不是完整 tracking：第六次尚未验证，1.934 s 时关节 RMSE 达到 0.200968 rad。
完整文件包含 2000 帧、29 次落地；没有将失败后的模拟运行时间算作合格跟踪。
独立运行 `late_right_leg150` 在第五次确认前的 950 个 q/v 样本逐位相同，
五次落地报告也相同。证据为 `five_strict_prefix.json` 和
`five_strict_reproduction.json`，全段状态保持 `full_tracking_passed=false`。

本候选保留统一 q/v/a 前馈，在实测卸载后分别增加左腿与右踝姿态反馈额度，
并从 1.55 s 起独立限制脚部垂直反馈。时间参数仍用于实验定位，尚未形成通用
控制策略；实验没有替换默认入口。继续尝试的记录、失败原因和原始源文件哈希
汇总于 `acc_data/amp_walk_20261103/progress.json`。

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash acc_data/amp_walk_20261103/feedback_followup/run_five_strict.sh \
  --duration 10.1 --stop-at-fidelity-loss \
  --output-path acc_data/amp_walk_20261103/feedback_followup/five_replay.npz
```


后续 `five_tube10` 将严格验证前缀推进到七次落地，第七次在 2.798 s 确认、
同帧源足部位置误差 0.027906 m。运行到 3.290 s 的原始关节 RMSE、根位置与
根姿态误差峰值为 0.192928 rad、0.076270 m、0.326202 rad，均通过保真度阈值。
独立运行 `tube10_global100_290` 在第七次确认前的 1400 个 q/v 样本和七次落地
报告逐位一致（`seven_strict_reproduction.json`）。这仍未完成全段 tracking。

该实验把原始参考附近的关节、根位置、根姿态误差边界转换成物理加速度 QP
约束；原始轨迹验收保持独立。3.292 s 的 QP 无解由新加的关节误差约束引起，
离线线性可行性检查见 `tube10_failure_feasibility.json`。第八次已获得离地证据，
但承重落点误差最小仍为 0.074768 m。后续重点检查净空修正后的 q/v/a：
第八次下降过渡中关节参考加速度峰值达 3222.82 rad/s²；同帧原始 PCHIP
参考也有 3616.02 rad/s² 的尖峰，并非全由净空修正产生。逐帧对照见
`eighth_reference_acceleration_audit.json`。
数学约束检查见 `tracking_tube_algebra_audit.json`，不是物理跟踪通过的替代品。


当前最好严格前缀进一步推进到九次落地：`eight_smooth6_tube2` 第九次在
3.902 s、源帧 780.4 确认，同帧原始足部误差 0.008353 m、承重 174.59 N。
截至最后有效样本 4.002 s，峰值关节 RMSE 为 0.192928 rad、根位置误差
0.081976 m、根姿态误差 0.326202 rad，均保持原门槛。独立运行
`nine_swing_balance_first` 在第九次确认前的 1952 个 q/v 样本和九次落地报告
逐位相同，见 `nine_strict_reproduction.json`。完整 2000 帧、29 次落地尚未完成。

改进保留源文件和独立验收，将第八步之后每个摆步的净空修正关节 RMSE 限制为
0.08 rad，并从源时间 2.82 s 起将控制参考平滑为统一 q/v/a 曲线；误差边界所用
源参考也做小幅一致平滑，验收仍比较未平滑的原始姿态。复跑入口为
`acc_data/amp_walk_20261103/feedback_followup/run_nine_strict.sh`。
第十步原计划摆动仅 0.11 s，基线在 4.004 s 停于 QP 验证失败，尚缺合格离地。
缩短卸载过渡的实验已得到 2 cm 离地证据，但落脚仍未达到 4 cm 条件；这些
后续试验没有被计为十步或全段成功。


第十步的后续诊断保存在同一实验目录：
`nine_anticipate008_xy80.steps.json` 已记录合格离地（最大净空 0.025645 m），
但承重时最小同帧落点误差仍为 0.126634 m，因此严格前缀仍为九次。
单纯暂停源时间或延长左脚支撑尚未完成第十次落地，未提升全段状态。

局部参考试验新增 `joint_correction_overlay.py`：在既有全身曲线上叠加有限区间
的 C2 关节修正，解析计算统一 q/v/a，区间外保持原参考。
`prepare_stance_overlay.py` 和 `prepare_tenth_foot_overlay.py` 用纯 NumPy URDF
运动学检查支撑保持、脚掌回正和净空修正，保留原始验收参考。
`stance_left_y004.curve_audit.json` 等记录限位与导数检查；它们只证明参考的
局部运动学性质，不证明物理跟踪成功。脚掌完全回正的候选超出所设运动学残差
门槛，已拒绝；部分回正候选继续在原生碰撞和原 QP 约束下验证。


接触切换实验进一步发现：只修改卸载力目标不会更新接触状态机使用的计划。
`--gate-contact-transitions` 让二者使用同一套原始同帧落点/实测承重判断；
物理触地和支撑丢失仍可覆盖控制意图，见 `contact_gate_audit.json`。
配合有限区间支撑保持和延后下降，`nine_contact_tail5` 的第十步承重落点
误差最小为 0.043168 m，仍未满足原 0.04 m / 0.05 s 要求，不能算十步通过。
该运行最终在 4.210 s 超出原关节误差门槛。

实验 `--relax-joint-tube` 仅在辅助关节误差预测约束不可行时改用软目标，
不修改原物理约束或实际轨迹验收门槛。各失败快照的独立线性可行性检查
记录在 `tenth_failure_feasibility.json`；更多运行对照见 `tenth_comparison.json`。
支持脚速度稳定的实验复用原投影器的有界稳定项，只有真实活动的接触坐标
受到修正；自由转动方向继续由观测接触点/线/面决定。


## 十次严格落地验证（2026-11-04，尚未完成全段）

`nine_plant20` 从源帧 0 连续通过十次落地，第十次在 4.072 s、源帧 814.4
确认，同帧原始足部误差 0.012789 m、承重 610.34 N，满足原有离地和 50 ms
连续承重条件。第十一次已经取得合格离地，但未验证落地。4.252 s 的关节
RMSE 达到 0.202541 rad，触发原始 0.2 rad 门槛停止；全段状态仍为失败。

该候选在既有平滑全身参考上，用有限区间 C2 关节修正保持旧支撑、延后第十步
下降，并将第九步控制落点沿世界 x/y 各移 -0.02 m，以减少随后支撑踝关节的
力矩饱和。所有落点仍对照未修改的原始 PKL 同帧足部位置验收。参考的根位置
小幅修正、角度限位、解析 q/v/a 和父级文件哈希检查均保留；没有修改模拟状态
或施加额外外力。该轨迹专用实验尚未替换默认控制器。

证据：`feedback_followup/ten_strict_prefix.json`、`nine_plant20.steps.json`、
`nine_plant20.tracking.json` 和 `ninth_plant20.curve_audit.json`。复跑入口：
`feedback_followup/run_ten_strict.sh`（相对于 `acc_data/amp_walk_20261103/`）。


### 接触约束拆分修复与重新验证

后续检查发现 `examples/test_g1.py` 原先固定按每脚 20 行拆分摩擦/力矩锥，
但非耦合 yaw 模式实际每脚只有 12 行。右脚替换为观测点接触时可能残留该脚
错误坐标系的旧约束；左脚替换时则可能丢失另一脚的摩擦和法向力上限。
现通过 `split_foot_wrench_constraints` 按实际行数拆分，主入口与当前 tracking
实验入口同步修复。新增回归覆盖两种锥、倾斜足底替换为水平点接触，以及另一
脚摩擦和载荷约束的完整保留。30 项相关测试和完整 496 项回归均已通过。
修复后重跑 `wrench_fixed_ten` 只验证了三次落地，0.944 s 在第 4 步起脚阶段
关节 RMSE 达到 0.201106 rad；当前版本尚未恢复历史十步结果，更未完成全段。
`wrench_split_regression.log` 和 `wrench_fixed_ten.tracking.json` 保存对应证据。

此前十次落地的历史记录已独立复现：`ten_arms4` 在 4.072 s 之前的 2037 个
q/v 样本和落地报告与 `nine_plant20` 相同（`ten_strict_reproduction.json`）。
这些记录来自拆分修复之前；**不能据此声称修复后的控制器已通过十次或全段**。
`run_ten_strict.sh` 目前保留该候选参数，用于修复后的重新验证。


修复后的实验现已重新验证五次落地：`fixed_lower15_gate_xy80_pose` 第五次在
1.822 s、源帧 364.4 确认，原始同帧足部误差 0.021113 m、承重 397.15 N。
原始验收标准保持不变，见 `corrected_five_strict_prefix.json`。
1.854 s 停于辅助姿态误差预测约束冲突；独立线性可行性检查见
`corrected_fifth_feasibility.json`，不代表物理约束可以放宽。
当前复跑入口为 `feedback_followup/run_corrected_five.sh`，仍默认尝试全段，
没有把五步作为完整 2000 帧、29 次落地通过。后续试验继续检查参考净空、
平滑后的预测约束和接触转换。


## 修正接触约束后的七次连续落脚（2026-11-04）

`feedback_followup/fixed_clock120_seventh10` 已从原始源帧 0 连续验证前七次落脚。
第七次确认在仿真 2.556 s、源帧 570.44，原始同帧足部位置误差 9.897 mm。
截至此时，原始 PKL 关节 RMSE、根位置和根姿态误差峰值分别为
0.185129 rad、0.063820 m、0.251149 rad，原有净空、载荷和 50 ms 持续落脚标准未变。
记录与逐步审计见 `feedback_followup/corrected_seven_strict_prefix.json`。

参考时钟在 0.90 s 开始，用 0.35 s 的连续过渡从 1 倍变为 1.2 倍；
统一参考速度和加速度都使用同一时钟的链式求导，原始审计也读取相同源帧。
第七次摆腿末段增加 10 mm 平滑净空，以避免脚在到达原落脚位置前碰地。
这些改动均作用于控制参考，原始 PKL 文件、物理模型、力矩和接触限制保持原值。
复跑入口 `feedback_followup/run_corrected_seven.sh` 仍尝试整段，并在原始保真度或控制验证失败时停止。

完整跟踪仍未通过：该次运行在 2.936 s 前后、第八次落脚尚未通过时，
原始关节误差达到停止阈值。第八步期间根部侧向误差再次增长，右脚落地位置偏离原目标。
整段验收仍要求全部 2000 帧和 29 次原始落脚；七步记录不能替代整段通过。
后续新增的变速 MPC 使用相同源帧时钟预测未来 q/v/a；其支撑多边形仍属于规划近似，
真实 WBC 继续依据实测接触约束执行并独立验收。

### 修正接触约束后的八次连续落脚（2026-11-04）

当前 `fixed_clock120_plant7roll25` 在原始 PKL 同帧验收下通过前 8 次落脚。第 8 次发生于仿真 2.924 s、源帧 658.76，落脚误差 0.036473 m；截至该落脚，关节 RMSE 峰值 0.185129 rad、根位置误差峰值 0.161051 m、根姿态误差峰值 0.251149 rad。阈值保持不变。

参考候选 `seventh_plant_roll_y25.npz` 在第 7 步调整控制落点和足底姿态，仍以未修改的原始参考验收。复现入口为 `bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_eight.sh`，证据为同目录 `corrected_eight_strict_prefix.json`。

**完整 tracking 尚未通过**：该试验在 3.056 s、源帧 690.44 因关节 RMSE 超过 0.2 rad 停止，第 9 步尚未通过离地验收。完整要求仍为覆盖 2000 帧、验证全部 29 次落脚。

### 修正接触约束后的九次连续落脚（2026-11-04）

`fixed_clock120_plant6roll30` 已验证前 9 次连续落脚。第 9 次发生于 3.440 s、源帧 782.60，落脚误差 0.009003 m；截至该落脚，关节 RMSE 峰值 0.185129 rad、根位置误差峰值 0.123437 m、根姿态误差峰值 0.296204 rad。候选在八步版本的基础上，修改第 6 步右脚控制落点及足底滚转；原始 PKL 验收参考和阈值保持不变。

复现：`bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_nine.sh`。证据：`acc_data/amp_walk_20261103/feedback_followup/corrected_nine_strict_prefix.json`。

完整 tracking 仍未通过：在 3.526 s、源帧 803.24 触及关节误差上限，第 10 步尚未验证离地。第 10 步的原始摆动区间只有源帧 780–802，正在检查转向阶段的支撑转移和时间分配。

### 修正接触约束后的十次连续落脚（2026-11-04）

`fixed_clock120_tenth_correction` 已通过前 10 次原始源帧落脚验收，使用修正后的双脚接触力约束。第 10 次发生于 3.634 s、源帧 829.16，落脚误差 0.032804 m；截至该落脚，关节 RMSE 峰值 0.185129 rad、根位置误差峰值 0.045247 m、根姿态误差峰值 0.297436 rad。

这个版本使用可选的提前卸载滞回：新脚实际承重且落点合格后启动，短暂载荷波动不会重置卸载计时，失去实际支撑或源帧落点超差时退出。物理接触约束和独立落脚验收仍由测量决定。第 9、10 步的控制落点调整均经过有界腿部 IK 检查，原始 PKL 验收参考不变。

复现：`bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_ten.sh`。证据：`acc_data/amp_walk_20261103/feedback_followup/corrected_ten_strict_prefix.json`。

完整 tracking 尚未完成：该试验在 3.934 s、源帧 901.16 因关节 RMSE 超限停止。第 11 步已验证离地、落点位置达标，但连续有效落地仅 20 ms，未满足 50 ms，因此仍只能计为 10 次落脚。

### 2026-11-04：修正接触力矩约束后的连续 11 次落脚

`feedback_followup/fixed_clock120_step11_body10` 在第 10 步配置上，从源时间 4.2 s 起用 0.06 s 的平滑过渡，将躯干姿态任务权重缩为 0.1。原始根姿态误差上限仍为 0.35 rad，物理约束和逐帧原始 PKL 验收均未放宽。该调整避免了第 11 步触地后立即因关节误差越界。

第 11 次落脚在仿真 3.954 s、源帧 905.96 验证完成，原始同帧脚位置误差 0.019000 m，实际承重 418.63 N，连续承重时间满足 0.05 s。到此为止，峰值关节 RMSE 0.185129 rad、根位置误差 0.150299 m、根姿态误差 0.297436 rad，严格前缀计数为 11。

复现入口为 `acc_data/amp_walk_20261103/feedback_followup/run_corrected_eleven.sh`；证据为同目录 `corrected_eleven_strict_prefix.json`。整段仍未通过：4.056 s／源帧 930.44 的关节 RMSE 达 0.202793 rad。第 12 步尚未验证离地，已测得的卸载离地高度最高 0.014190 m，低于 0.02 m 要求。不能将此结果解释为全片完成。

后续的 `fixed_clock120_body10_overdue` 保留前 11 次严格落脚，并将第 11 步误差降至 0.013391 m。参考仅在源帧 820 之后逐步增加 0.04 m 根部侧向偏移，以有界腿部 IK 保留双脚世界位姿。第 12 步实际卸载离地高度达到 0.035136 m，已验证离地，但尚未验证落脚；4.088 s／源帧 938.12 的根姿态误差达 0.354216 rad，因此整段仍失败。入口为 `run_corrected_eleven_recovery.sh`，证据为 `corrected_eleven_recovery_strict_prefix.json`。

该入口还启用 `--anticipate-unload-overdue`：当下一次计划抬脚时间已过去，但新支撑脚已经满足既有真实承重、源脚位误差和短时承重门槛时，允许继续发起卸载。连续 50 ms 落脚验收不变，且禁止越过下一摆动段的结束帧。4 项单元测试及原接触门控检查通过；启用此选项前后，本次已录制的源帧 0–938.12 的 `history_q` 完全一致。完整尾段仍需实测。

实验批处理现在以 `full_motion_tracking_passed` 判定整片成功（旧报告缺少此字段时才回退到旧字段）。原始片段末尾仍在运动中，整片验收不附加静止收尾要求；覆盖全部 2000 帧、29 次原始落脚、物理控制有效及全程 PKL 误差合格四项要求不变。

2026-11-04 后续诊断补充：当前最佳连续有效前缀仍为 11 次落脚。
`fixed_clock120_body10_recovery_uninterrupted` 保持相同物理约束、不在第一次
PKL 误差超限时中断，首次超限仍为 4.088 s；继续到 4.242 s 时 QP
不能满足约束并停止，未恢复或完成第 12 次有效落脚。不能将延长运行时间
解释为更多有效跟踪进度。

另查明一个控制优先级问题：首层 COM、上身方向和全部关节姿态的加速度
矩阵秩为 35，原来位于次层的 `joint_limit_avoidance` 无法改变首层已锁定
的加速度。`feedback_followup/upper_braking_priority_audit.json` 保存了
腰部已需制动、QP 却仍要求向限位加速的记录。需要区分：原生仿真的关节
位置限位一直保留，但该实验配置的 `acceleration_and_braking_limits`
为 false，额外的 QP 预测制动界并未开启。新实验选项
`--joint-limit-priority-start` 可将接近限位的软制动请求纳入首层；该单项
修改尚未提高有效步数，因此没有替换已验证的运行配置。

2026-11-04：辅助约束与参考动力学的进一步检查。保存的首层 QP 对照计算显示，
3.900 s 和 4.000 s 的根姿态跟踪管约束会显著改变腰部加速度；单独删除该约束
虽可减小部分关节的纠偏冲突，却会增大根部滚转加速度。实际关闭源时间 4.2 s
之后的全部跟踪管，仅保留 11 次落脚，且在 3.962 s 提前超限。因此没有将关闭
辅助约束作为修复，也没有更改原始 PKL 的验收界限。源时间 4.2 s 后将跟踪管
带宽改为 4 或 30 的两组试验分别在 3.972 s、4.066 s 超限，同样没有提高步数。
对应记录为 `feedback_followup/qp_individual_tube_audit.json` 和
`wrench_fixed_tube_braking.log`；4.060 s 的保存 QP 已软化关节管，最后两条硬管
分别是根位置和根姿态，不能按三条硬管解释其对偶量。

新增离线 `mujoco_reference_dynamics.py` 仅计算同一 URDF 的刚体动力学，不进行
仿真步进，也不改变 Isaac 的状态、碰撞或电机参数。其世界坐标广义速度映射和
电机转子惯量已与四个实际 QP 快照核对：质量矩阵最大绝对差约 6.1e-6，偏置力
最大绝对差约 1.2e-4，足部位置最大差低于 4e-7 m。检查文件为
`mujoco_reference_dynamics_audit.json`。在原计划单支撑、精确执行参考 q/v/a
的假设下，最佳候选的后段部分帧要求很大的踝关节力矩；这说明几何预检或净力
预检不能替代全身力矩检查，并不构成“容差内跟踪不可能”的证明。

`full_effort_linearization.py` 将参考的 COM、根姿态及上身关节变化映射到单支撑
接触力矩和各电机力矩，用于进一步准备参考。独立曲线扰动检查的相对误差低于
0.43%，见 `full_effort_linearization_audit.json`。这些离线检查不计为物理落脚
成功；目前发布的最佳入口仍为 `run_corrected_eleven_recovery.sh`，整片仍未通过。

后续 COM 诊断修正：本运行的 COM 反馈实际 K 为 [35,35,25]、D 为
[12,12,10]，反馈加速度上限为 [4,4,4]；`--motion-feedback-scale 9`
作用于关节、脚和躯干任务，不会将 COM 的 K 乘 9。离线重建与记录的
COM 目标加速度最大差为 4.77e-7，见 `com_feedback_clipping_audit.json`。
提高 COM 反馈上限的实际试验未超过连续 11 次落脚。

新增的全身有限时域预测仍是实验，不是已验证的完整 MPC 跟踪方案。
首个线性可行解在非线性复核中出现约 165.6 N 基座力残差和 67 mm
足底穿地，因此已增加全足底几何、非线性动力学及状态积分检查。
短时域修正版 `fixed_clock120_body10_full_body_preview42_refined6`
仍只验证 10 次落脚，在 3.900 s 超出原始 PKL 误差门槛。

进一步修正预测中遗漏的触地冲量，并检查冲量动量一致性和非增能条件。
预测接触点随原生足底几何更新，以表示脚跟到脚尖的滚动；接触速度约束
也加入对配置的导数。初始猜测不再作为必须满足的数值信赖域，后续 SQP
迭代仍围绕上次解限制步长，所有电机、源帧误差和接触约束保留。
`full_body_preview_rolling_audit.json` 中，3.7 s 和 3.8 s 实测快照的
0.3 s 预测通过非线性复核；这只是模型检查，不计为落脚成功。
`full_body_preview_audit_tests.log` 的 7 项检查覆盖穿地、状态跳变、
冲量动量不一致及产生能量的冲量拒绝，并验证耗散冲量可以通过。
实验的输出仍须经过原 WBC 的实际物理约束投影，未通过预测复核时回退
到已验证的原 WBC 输出；没有写入仿真状态或施加预测冲量。

滚动接触与预测控制的后续实测仍未超过最佳前缀：
`fixed_clock120_body10_full_body_preview42_rolling12` 验证 10 次落脚，
4.056 s 根姿态超限；补上摆脚位置/姿态目标和参考净空后，
`fixed_clock120_body10_full_body_preview42_feet12` 验证 11 次，
4.040 s 超限。增加数值信赖域恢复的 `full_body_preview42_adaptive12`
只验证 10 次，在 3.916 s 超限。完整 2000 帧跟踪仍未通过。

失败快照的离线诊断表明，SQP 的速度信赖域可能阻止后续迭代；恢复仅改变
优化器的数值搜索范围，不删除物理、电机或 PKL 约束。相同快照在更多迭代后
通过非线性复核，见 `full_body_preview_online_trust_audit.json`。
此外，即便脚级接触模式没变，实测接触点/线/面也可能变化，导致缓存加速度
需大幅投影。实验在线包装器现在在该偏差过大时立即重算；若重算失败，采用
先前已经验证的常规 WBC 输出。3 项在线包装器检查通过，见
`online_full_body_preview_tests.log`。

长时域（24 × 25 ms）在两个失败前实测快照通过非线性复核，见
`full_body_preview_long_audit.json`；连续物理验证仍另行进行。可选
`--full-body-preview-warm-start` 从上一次合格预测构造初值，保留重新求解
回退；初值移位不会把触地冲量分摊到触地前速度，3 项针对性检查通过。
可选 `--full-body-preview-track-wrench` 将第一阶段规划的接触力矩目标
一并交给原 WBC 投影，使用当前第一阶段 QP 的接触坐标。两项默认关闭。

预测模型复核的重要更正：上述早期“非线性复核通过”仅覆盖当时已有的检查，
**不能再作为未来接触力矩可实现的证据**。后续发现预测器把最低角点当成对称
支撑矩形的中心，可能允许压力中心超出原生足底；同时没有检验所有未来承重
点的接触加速度。独立复核见 `full_body_preview_legacy_wrench_audit.json`
和 `full_body_preview_contact_acceleration_audit.json`。真实执行的首阶段
WBC 始终保留实际接触约束，最佳连续 11 次落脚记录不依赖该预测模型。

现在实验预测使用 `native_patch_wrench.py` 从原生接触点的摩擦射线生成
耦合力矩锥，分别处理点、线和多边形的仿射空间，避免虚构支撑面积或独立
偏航力矩容量。5 项检查覆盖角点外侧压力中心、三角形边界、线接触的摩擦
与偏航耦合，以及合法逐点力的接纳。被用于传力的点还需满足接触高度和
速度约束；选点阈值缩为 0.5 mm，避免把重叠原生碰撞体中本来高低不同的
表面强行当成同一支撑面。预测另外考虑足底接触点增加时的冲量。

MuJoCo 的足部雅可比时间导数经独立差分核对，三个实测状态的最大误差
小于 2.2e-7，见 `contact_acceleration_bias_audit.json`。该项用于预测中的
接触加速度约束和独立复核。补齐这些检查后，晚段的完整接触预测尚未获得
可执行解；不再将旧的宽松模型结果用于声称预测可行。当前在离线改进 SQP
初值恢复，只有恢复全部硬约束并通过完整复核的结果才能用于控制。

最新两次完整尝试仍失败：`full_body_preview42_wrench_warm24` 在
3.930 s 关节误差超限；`full_body_preview42_guarded24` 在 3.958 s 超限，
均只有 10 次有效落脚。物理阈值、原始 PKL 和最佳 11 次落脚入口保持原值。
当前预测相关 18 项针对性检查通过，见 `full_body_preview_combined_tests.log`。

随后已在 3.7 s、3.8 s 的实测状态获得新的 0.3 s 可行预测，记录于
`full_body_preview_elastic_fixed_audit.json`。初值恢复允许临时求解带松弛量
的数值子问题，但该结果不能用于控制；接受前必须重新求解不含这些松弛量的
硬约束问题，并通过动力学、原生接触力矩锥、承重接触速度/加速度、冲量和
节点间净空复核。两个最终解的松弛量均为空，接触加速度残差分别约
0.870、0.882 m/s²，最低足底高度约 −0.5 mm。`test_preview_elastic_admission.py`
检查即便其余复核通过，也不能接受仍使用松弛量的解。

接触复核按预测中实际用于传力的原生接触点检查速度和加速度，不把几何上
接近地面、但没有用于传力的另一个碰撞体表面误当成刚性固定点；全部原生
表面的节点间不穿地检查仍保留。当前预测相关 20 项针对性检查通过。

逐点力预测进一步补齐了接触点位置变化对力矩、摩擦方向的导数，独立微扰
复核的相对误差分别低于 2.94e-5、1.87e-5。旧的聚合力矩锥预测在实测
3.77 s 状态迭代 30 次仍失败；逐点力模型在 13 次内通过，证据见
`full_body_preview_distributed_audit.json`。连续试验仍另行检查。

新试验在 3.79 s 又遇到接触点集合在 SQP 迭代间切换的问题。现在每次重规划
内部固定所选原生接触点集合，下一次重规划重新选取；没有放宽穿地、承重、
接触力矩或原始参考误差阈值。相同失败状态在 5 次迭代内获得无松弛量、通过
全部复核的解，见 `full_body_preview_frozen_audit.json`。落脚冲量映射导数
也通过独立微扰检查（误差低于 3.43e-5），但单独加入该项没有解决收敛，因此
未把该实验版本用于在线控制。完整跟踪仍未通过，最佳严格前缀仍为 11 次。

还补上了预测对落脚跟踪目标的检查。`prediction_source_feet_audit.json`
表明，之前两条动力学可行预测中的新落脚位置仍可偏离原始 PKL 约 6–11 cm；
这不是一次合格的 tracking。新落脚在 50 ms 保持时间及一个网格余量内，现需
满足原始脚链接位置的 3.5 cm 预测约束，并通过实际非线性误差小于 4 cm 的
复核。3.79 s 失败状态的 0.6 s 预测在 6 次迭代后通过，复核最大落脚误差
3.509 cm，见 `full_body_preview_landing_reaudit.json`。这一预测结果不算实际
落脚成功；连续物理跟踪仍需独立通过。新增两项落点接受/拒绝检查，总计
22 项预测相关检查通过。

逐点力但未固定接触集合、未加入落点硬约束的连续试验
`fixed_clock120_body10_full_body_preview42_distributed12` 在 4.038 s 失败，
严格有效落脚仍只有 10 次。已切换到固定接触集合、原始落点约束和 0.6 s
预测的 `fixed_clock120_body10_full_body_preview42_landing24` 实测。
重复计算的动力学/接触差分状态现可共享，同一失败状态的控制、状态和冲量
数组与优化前逐元素完全相同，见 `full_body_preview_cache_equivalence.json`。

`full_body_preview42_landing24` 连续测试仍只验证 10 次落脚，在 3.886 s
首次超过原始 PKL 的误差阈值。前 8 次在线预测通过，3.786 s 开始失败。
该状态暴露了首节点接触集合的不一致：几何上靠近地面的 4 个原生点被当成
已接触，但实测力和真实 WBC 只有右脚 9、11 两点承重，见
`initial_observed_patch_audit.json`。实验入口现从真实接触力（>0.5 N）、
分离距离（≤1 mm）及地面法向筛选承重点，并匹配原生碰撞点；预测以此判断
随后是否新增接触点，避免漏掉压平冲量。没有用几何接近代替实际承重。

单独修正首节点仍不能让该失败状态收敛，不能据此声称 tracking 已恢复。
正在以 `full_body_preview35_observed24` 从源轨迹 3.5 s 提前介入预测控制。
`test_observed_preview_patch.py` 的 4 项检查覆盖未承重近地点排除、真实平面
支撑、无载荷时无支撑，以及不匹配原生碰撞点时拒绝；当前预测相关 26 项
检查通过。每次成功预测另存最近一份状态/控制，以便复现后续热启动失败。

进一步把预测近端的两个区间细化为 10 ms，其余区间保留 25 ms。相同
3.786 s 状态在实测承重点初始化下，用该网格 7 次迭代通过所有复核，
而统一 25 ms 网格仍失败；见 `full_body_preview_nonuniform_audit.json`。
预测根部力/力矩残差分别约 1.80 N、0.37 Nm，落脚误差约 3.50 cm。
这是近端接触变化时间分辨率的修正，不是放宽物理或跟踪阈值。

`full_body_preview35_observed24` 已主动终止以切换网格修正，不计为成功或
一次完整试验，其记录见同名 `.superseded.json`。新测试为
`full_body_preview35_fine24`，源轨迹 3.5 s 开始介入、10 ms 重规划。
热启动按不等长时间网格移动状态，积分复核也使用每段真实时长；新增检查
覆盖非均匀网格中的运动、冲量时间和错误时间步拒绝。

`full_body_preview35_fine24` 在约 3.15 s 发生 CPython 原生扩展异常
`notimplemented_dealloc: deallocating NotImplemented`（退出码 -6），未输出
完整 tracking 报告，不能算作合格结果；同名 `.runtime_failure.json` 保留
异常说明。该轮前面仍有预测失败/恢复交替，不能只用某次预测通过推断成功。

为隔离预测库与 Isaac 原生库的同进程影响，实验预测器现通过私有本地管道
在独立 Python 进程运行。它只读取状态和真实首阶段 QP，返回预测结果；
Isaac 中的原 WBC 仍负责物理约束投影及最终控制输出。通信或求解进程失败
返回明确的预测失败状态，使用已验证的基线输出，不注入状态或外力。
同一失败状态的独立进程与原进程预测逐元素完全一致（状态、控制、冲量的
最大差异均为 0），见 `preview_process_equivalence.json`；30 项检查通过。
新连续测试为 `full_body_preview42_fine_process24`。

预测接触精度的进一步检查发现，旧预测允许选中的承重点离地约 0.6–0.7 mm
仍承担数百 N，见 `predicted_loaded_gap_audit.json`。因此不能把旧的模型
复核通过当成高精度的接触一致性证据。新候选将节点接触高度线性约束收紧到
±20 μm，并复核承重点离地/穿地的绝对误差小于 50 μm，全部原生表面在节点
也须满足该不穿地精度。3.786 s 状态的预测在 10 次迭代通过，承重点最大
绝对高度误差约 40.5 μm，根部力/力矩残差约 0.59 N、0.21 Nm；见
`full_body_preview_tight_contact_audit.json`。节点间的近似积分仍单独检查，
不能把有限网格的模型复核当作连续物理跟踪的保证。

新增可选 `--full-body-preview-reuse-verified`：新预测失败时，仅在上一条
合格预测未过期、年龄不超过 60 ms、接触模式相符、实测状态仍在近邻范围内
时提取其当前阶段目标，并重新通过当前真实 WBC 投影；原来的投影偏差判定
和物理验证仍保留。未来阶段的世界坐标力矩不会误作当前 WBC 接触坐标力矩。
该项默认关闭，38 项已有/新增检查通过，另有 3 项接触高度检查通过。

`full_body_preview42_fine_process24` 没有再出现原生扩展崩溃，但在约 3.91 s
遇到实测加载点无法在 3 mm 内匹配到原生候选点，因而未产生完整 tracking
报告。现在这一预测观测异常会保存原始接触诊断并使本周期使用原 WBC，
不会把未匹配点当成有效支撑，也不会直接终止整段仿真。

更关键的模型不一致已用真实执行快照定位：控制器运行在 500 Hz，仿真采用
先更新速度、再更新位置的半隐式积分，而旧预测/热启动/反馈使用连续恒加速
位置公式。在同一个 4 ms 实测区间，改用两个真实 2 ms 积分步后，最大关节
位置预测误差从 0.0094225 rad 降至 0.0001637 rad，根姿态误差从
0.0013575 rad 降至 0.00009985 rad；见 `semiimplicit_prediction_audit.json`。
预测的位移加速度系数、节点及节点间复核、热启动、缓存轨迹复用、在线反馈
已统一积分方式。近端仍为两个 10 ms 区间，后续改为 24 ms，使其对齐 2 ms
仿真步；窗口到达源末端时截短，保留末帧运动速度，不添加静止收尾。

带严格节点接触精度的新模型在 3.786 s 状态用 8 次迭代通过，见
`full_body_preview_semiimplicit_audit.json`。独立进程与本地结果逐元素一致，
见 `preview_process_semi_equivalence.json`。52 项针对性检查通过。
`full_body_preview42_tight_process24` 已主动终止以应用积分修正，不能计为
完整试验；当前连续测试为 `full_body_preview42_semi_process24`。完整 tracking
尚未通过，严格最佳实际前缀仍是此前的 11 次落脚。


进一步定位到预测缺少未来避碰约束。`full_body_preview42_semi_process24`
在 3.71 → 3.72 s 的实测根位置预测误差仅约 0.11 mm，广义速度最大误差
约 0.0933；但上一预测的第二阶段控制违反真实手腕/躯干 barrier 约 78.2。
这说明积分修正后，未来可行域与真实 WBC 仍不一致，不能把旧预测的
动力学/地面接触复核通过解释为未来真实 WBC 可执行。该连续实验已主动
终止并写入 `.superseded.json`，没有完整 tracking 结果。

新增实验选项 `--full-body-preview-collision-barrier`，将真实运行中同一套
球覆盖几何、配对和安全距离送入独立预测进程，在未来节点线性化并复核
同一个二阶避碰约束。`preview_collision_equivalence.json` 对保存的实测
状态核对了 317 条活跃约束：雅可比最大差 9.48e-7，边界最大差约 0.00484，
解析加速度偏置与独立有限差分最大差 1.75e-7；下一阶段最大残差 78.2042。
新增 4 项检查覆盖距离导数、根刚体运动不变性，以及动力学可行但避碰约束
不满足的预测必须被拒绝。原来的实际碰撞、力矩、速度和源运动验收标准
保持不变。该候选仍在失败状态复现阶段，尚无改善实际落脚数的证据。


新增避碰候选的初版还发现一个实现错误：避碰活跃掩码覆盖了支撑脚列表，
导致线性预测错误地允许摆动脚受力。该版本没有产出获准执行的预测，
`full_body_preview42_collision_process24` 已终止，相关离线长/短窗口和
线搜索结果标为无效原型，不用于判断可行性或比较收敛性。现已分离变量，
并给主实验模型及避碰候选增加独立的 `nonlinear_inactive_wrench` 复核。
回归检查验证单脚支撑时，即使避碰掩码同时包含 True/False，摆动脚仍为零力；
即使总力平衡，分配给非支撑脚的力也必须被拒绝。另外独立验证未来 barrier
确实改变优化加速度，而非只在事后检查。当前 59 项针对性检查通过。
重新运行的连续实验为 `full_body_preview42_collision_fixed_process24`；
实际完整 tracking 仍未通过，最佳严格前缀没有更新。


完整未来避碰候选在真实连续仿真中成功执行了多次预测，但保持单一加速度
常在 6–8 ms 后触发投影偏差重规划。`full_body_preview42_collision_fixed_process24`
已保存为未完成的被替代实验。进一步测试 `full_body_preview42_collision_fine_process28`：
前五个控制段为真实 2 ms，第六段 10 ms，后续 24 ms，总预见长度约 0.55 s；
在线按实际源帧读取轨迹的 q/v/a，每次仍经过当前物理 WBC。
未来世界坐标 wrench 不会误作当前 WBC 接触坐标 wrench。时间节点的浮点误差
采用 1e-8 帧的对齐容差，真实到达冲量节点之前仍使用冲量前状态，末节点不越界。

该细网格实验在 3.68 s 出现一次 NumericalError 后使用基线 WBC，3.72 s
恢复合格预测；3.736 s 又遇到预测不收敛并回退，之后多次恢复。3.800 s 的
实际源动作误差为关节 RMSE 0.138572 rad、根位置 0.074551 m、根姿态
0.168951 rad，见 `collision_fine_actual_snapshot_fidelity.json`。这些只是
该时刻的实测误差，并非完整落脚或整段成功。该轮也已标记 superseded，
没有完整 tracking 报告。

求解性能分析表明，全量未来避碰矩阵包含约 1.5 万行。新增按需约束生成：
从每个节点最紧的少量行开始，每次解后检查全部原始线性不等式，补入所有
违反的行；达到轮数上限时恢复全量。数值失败也用全量问题复核，不能将
缩减问题的数值错误直接视为完整问题不可行。另利用自碰撞距离对两连杆
共同祖先运动的不变性，显式清除这些方向的数值导数噪声。
同一线性化问题中，184 行求得的解满足全部 14933 行，目标函数相对差
1.42e-8、加速度最大差 0.00199 rad/s²，单次构建/求解从约 17.6 s 减至
6.9 s；见 `preview_collision_cut_equivalence.json` 和 profile 文件。
但完整非线性迭代可能走不同路径：首次冷启动得到的首加速度最大差约
92 rad/s²，两者都通过模型复核，不能将其解释为相同闭环控制器。
多线程 faer 在此问题更慢，未用于实际候选。

另一个接触问题有实证：旧失败状态的未来脚跟点 9 已离地约 0.155 mm，
预测法向力仅约 6e-7 N，却仍被固定在接触位置/速度/加速度约束中。
新增未来接触释放，只移除法向力 <0.5 N、已离地 >50 μm 的预测点，
且必须保留另一个至少承担 20 N 的点；当前真实观测节点绝不修改。
移除后重新优化所有力、动力学及接触约束，不能直接执行旧的无效预测。
同一 3.72 s 状态由原先 30 次迭代仍失败，改为 17 次通过，承重点高度误差
26.1 μm、避碰残差 0.0130，见 `full_body_preview_collision_release_current_audit.json`。
实测初始加载点高度误差只有 0.7–2.4 μm，因此没有放宽地面接触容差。

新增可选 `--full-body-preview-state-feedback`（要求细网格跟随）：在已复核
轨迹仍与实测 q/v 接近、接触模式一致时，允许经过物理验证的 WBC 修正
MPC 期望加速度，不再仅因瞬时加速度归一化差 >1 就立即重规划。
该选项明确改变的是重规划/投影差异门槛；物理求解验证、源动作逐步误差
检查和全部落脚验收保持不变。状态离开原有近邻范围、接触模式不匹配或
物理投影无效时仍重规划/回退。近邻范围为根位置 25 mm、姿态 0.07 rad、
单关节位置 0.1 rad，线/角/关节速度差分别 0.5 m/s、1 rad/s、3 rad/s，
并保留 60 ms 有效期。该项默认关闭。

当前组合候选 `full_body_preview42_collision_state_release28` 启用未来避碰、
按需约束、细网格轨迹跟随、未来零力接触点释放和上述状态反馈重规划。
使用 float32 运行时覆盖几何的首次预测在 10 次迭代通过，见
`full_body_preview_collision_fine_release_initial_audit.json`；80 项针对性
检查通过。后续实验会保存完整求解配置及最新规划输入/失败输入，便于
复现每个实际状态。完整 tracking 尚未通过，最佳严格实际前缀仍为 11 次落脚。

2026-11-04 后续连续预测与触地时序验证：

- `full_body_preview42_collision_state_release28` 连续执行了 3.680、3.720、
  3.760、3.800、3.814 s 的合格预测。3.720→3.760 s 的实际/预测根部
  位置偏差 0.000983 m、姿态偏差 0.00918 rad、关节偏差 RMSE 0.00717 rad；
  见 `collision_state_release_continuity.json`。这不是原始动作跟踪误差。
- 该轮最后记录的实际状态为 3.846 s / 原始帧 880.04，原始动作关节
  RMSE 0.13387 rad、根部位置误差 0.08086 m、姿态误差 0.09599 rad。
  之后预测热启动 20 次和冷启动多次仍不收敛，已保存 `.superseded.json`。
  该实验未完成，不能计为新增严格落地记录或完整跟踪成功。
- 源接触标记与准备后的原生脚底几何存在时间差：左脚在帧 884 被标为
  接触时仍离地约 16.915 mm，到帧 890 才降到 2 mm 内。新增默认关闭的
  `--full-body-preview-native-contact-timing` 仅在预测内部按原生脚底到达
  时刻延后触地；若因此出现无支撑间隔，延续上一支撑直到新脚到达。
  实际接触仍取实测，原始 q/v/a、时间覆盖、落地/腾空与源动作误差验收不变。
- 这一候选在实际启动 3.680 s 和卡住的 3.846 s 状态分别经 9、16 次
  迭代通过原有全部预测检查；支撑点高度残差分别 38.44、21.16 μm。
  证据为 `delayed_contact_initial_3680_audit.json` 与
  `delayed_contact_actual_3846_audit.json`。当前连续仿真候选为
  `full_body_preview42_collision_native_timing28`，尚待实际结果。
- 运行路径的默认接触模式保持不变；开启选项的模式/几何与独立离线计算
  一致。接触时序和网格共 13 项测试通过。新增只读 `observe_preview.py`
  保存每次重规划的实际状态和原始动作误差，避免最新状态文件覆盖诊断证据。
- 未采用的密集网格候选在冷启动发生 NumericalError，热启动后能通过
  3.814 s 的离线检查（416 s 求解）；这不代表实际跟踪效果改进。
  `full_body_preview_collision_anchored_release.py` 也仅为离线实验。

截至以上记录，完整原始 2000 帧 / 29 次落地仍未通过；已核验的最佳实际
严格前缀仍为 11 次落地与第 12 次腾空。最终判定继续使用
`full_motion_tracking_passed`。

后续实际结果与状态一致性修正：

- `full_body_preview42_collision_native_timing28` 已结束，严格落地 10/29。
  在 3.944 s / 源帧 903.56 因原始关节 RMSE 0.200305 rad 超过 0.2 停止；
  当时根部位置误差 0.117991 m、姿态误差 0.184436 rad。3.878 s 的预测
  失败后回退，3.918 s 双支撑重规划也失败（763.54 s 求解）。不能将
  “进入双支撑”当作通过连续 50 ms 的落地验收。该轮未改善最佳 11 步。
- 调整脚底触地阈值为 0，以及把初始实测支撑点保留 40 ms，均未解决
  保存的 3.878 s 状态的不可行性。这些实验未接入实际控制。
- 查明旧旋转状态转移近似可使一个真实 WBC 可行控制的 2 ms 脚跟高度
  从正确的 -2.497 μm 误算为 -57.990 μm，姿态积分误差 0.000877 rad。
  新 `preview_rotation_transition.py` 对相同半隐式四元数积分进行局部
  线性化，得到脚跟 -2.471 μm、姿态误差 5.16e-8 rad；证据为
  `rotation_first_step_3878_audit.json`。新候选把预测姿态积分审计阈值从
  0.02 rad 收紧至 1e-4 rad，实际原始动作姿态验收仍为 0.35 rad。
- 同一实测状态的右踝滚转角为 0.260399 rad，真实上限 0.261800 rad，
  速度 -0.359110 rad/s 已朝限位内部回退。旧 QP 又在固定的初始状态上
  强制 0.005 rad 内缩余量，导致直接矛盾。新增实验恢复规则仅对初始
  不在该规划余量内的关节，在 50 ms 内从 0 恢复至 0.005 rad；真实硬件
  限位始终保留，并增加所有预测节点的独立关节限位审计（<1e-6 rad）。
- 新默认关闭选项 `--full-body-preview-exact-rotation` 与
  `--full-body-preview-recover-joint-reserve` 已接入实验进程。当前候选
  `full_body_preview42_collision_rotation_recovery28` 从原始起点重新验证。
  `rotation_recovery_runtime_preflight.json` 记录相同 3.680 s 启动状态的
  完整预测检查通过，姿态积分误差 6.751e-6 rad、关节限位违反量 0；
  相关 52 项测试通过。保存的旧 3.878 s 状态仍未求得合格完整预测，
  因此不能宣称该卡点已被恢复或新一轮已经通过。
- `preview_status.py <run_name>` 优先读取实际结束报告并核对物理进程，
  避免把已经停止更新的预测器日志误认为仿真仍在运行。
- 降低 SQP 正则强度的离线实验未接入实际控制：固定早降方案失败；
  按残差降幅的方案仅从 16 次迭代降至 13 次，收益不足以改变当前方案。

2026-11-05：首步控制与离地接触点：

- `full_body_preview42_collision_rotation_recovery28` 在 3.680、3.684、
  3.702 s 采用合格预测；3.722 s 热/冷启动共 978.71 s 后仍因非线性
  接触检查不通过而回退。3.762 s 实际关节 RMSE 升到 0.178829 rad；
  此轮已保存 `.superseded.json`，未完成，未计为新的落地成绩。
- 在实际 3.722 s，右脚跟点 9 高度 1.43 μm、向上速度 0.02505 m/s，
  右脚尖点 11 高度 2.17 μm、法向速度 0.000377 m/s。实际第一步 WBC
  可行域上的线性优化及非线性积分检查给出脚跟下一个 2 ms 高度约
  36.6–60.3 μm；不能继续把两个点都当作未来固定地面点。
  见 `first_contact_step_rotation_3722_audit.json`。
- 新默认关闭的 `--full-body-preview-separating-points` 保持当前实测
  WBC 接触不变，仅在未来 40 ms 保守地去掉已明显分离的点；必须有另一个
  实测稳定点保留支撑，不能删除最后一点或构造新支撑。新力/几何轨迹仍须
  通过原来的全部审计。实际物理接触力与腾空/落地判据不受修改。
- 新默认关闭的 `--full-body-preview-damp-first-control` 将后续节点
  已有的 SQP 加速度近端正则同样用于第一个控制量。启动预测的最大根部
  首步角加速度从约 1275.5 降到 369.0 rad/s²；不是增加硬件能力或放宽
  任一物理限制。`first_damped_initial_3680_audit.json` 记录完整预测通过。
- 两项组合在保存的 3.722 s 卡点经 13 次 / 195.42 s 求解通过，支撑点
  高度残差 21.63 μm、姿态积分误差 1.75e-6 rad、真实关节限位违反量 0。
  见 `separating_actual_3722_audit.json`。启动状态没有被筛掉的接触点，
  同一启动预测也通过组合后的检查，见 `separating_runtime_preflight.json`。
  56 项相关测试通过。
- 当前实际连续验证候选为 `full_body_preview42_collision_separating28`。
  原始文件全段仍未通过；最佳严格实际记录仍为 11 次落地，第 12 次已腾空。

2026-11-05：球形脚底接触速度导数的有限差分核对。
在 `collision_separating28` 的实际 3.708 s 状态，预测器将球面当前接触点
当成固定材料点，对 `R p - r n` 的时间导数使用了 `ω × (R p - r n)`。
地面法向固定时，正确导数为 `ω × R p`。该状态的支撑脚旧偏置导数误差
为 0.0652238 m/s²，修正后与中心有限差分的最大差为 1.48e-9 m/s²。
证据为 `feedback_followup/surface_kinematics_actual_3708_audit.json`。
新增独立开关 `--full-body-preview-surface-kinematics`，修正未来接触速度的
配置导数、加速度偏置及相应非线性复核；实际首层 WBC 和验收阈值不变。
同一状态的修正模型冷启动预测在 9 次迭代、143.03 s 后通过所有预测复核，
最大承重点间隙 31.47 μm、根部力残差 0.785 N，见
`surface_actual_3708_audit.json`。这只是预测验证，不能算新增实际落脚。
包含滚动球面解析情形、零半径退化和有限差分的 59 项相关测试通过。
完整实际跟踪仍未通过，已验证最好前缀仍为 11 次落脚。

进一步的闭环诊断表明，球面导数修正不是此前 3.708 s 热启动停滞的唯一原因：
旧模型同状态冷启动也通过（165.86 s），见 `separating_cold_actual_3708_audit.json`。
新球面模型的真实运行在 3.686 s 触发重规划，6 ms 内根角速度预测误差
达到 1.16883 rad/s，姿态误差仅 0.003589 rad；实际与预测的支撑点均为
右脚 [9, 11]。同时 WBC 对预测加速度作了大幅改动。
`surface_projection_constraints_audit.json` 保存了该实际状态的 QP 对照：
预测加速度对最后一条根姿态跟踪 CBF 行违反量为 9.00712，其对偶量
14154.4；全部约束下最小加速度投影最大改动 158.53，去掉额外源跟踪
CBF 后为 5.51605，剩余物理等式/不等式残差 2.44e-7。
这些 CBF 是局部辅助控制约束，预测器已经有全时域原始源姿态误差约束。
现准备仅在 MPC 开始时通过既有 `--tracking-tube-end 3.68` 结束额外 CBF，
保留碰撞、摩擦、真实电机边界、原始源帧实际误差与落脚验收；尚待实际验证。

`surface_no_cbf_initial_3680_audit.json` 的同状态预测已通过：11 次迭代、
166.58 s，最大承重点间隙 22.30 μm、根部力残差 0.147 N、姿态积分误差
3.22e-6 rad。首个 10 ms 内最大根角加速度从 571.22 降为 261.87 rad/s²。
`run_wrench_fixed_full_body_preview_collision_surface_no_cbf.py` 已开始从源帧 0
实际运行；只有完整实际报告通过后才能更新成功结论。
此前 `collision_separating28` 和 `collision_surface28` 已保存为主动替换的
未完成试验（`.superseded.json`），不能计为新的连续落脚成绩。
新模型热启动若连续四次已收敛的动力学/碰撞迭代仍有停滞接触松弛，转入
原有冷启动完整复核，而非接纳带松弛的解；新增该行为测试后相关检查共 62 项通过。

### 2026-11-05：按用户要求停止 MPC，后续仅用 Fabric + WBC

用户明确要求“不要 MPC”。所有全身预测实际运行已停止，后续跟踪使用
`run_corrected_eleven_recovery.sh` 的直接 WBC 路径，并显式设置
`--full-body-preview-start inf --centroidal-preview-start inf`。历史预测文件仅留作
诊断记录，不作为正在使用的控制方案。
上一未完成的 `collision_surface_no_cbf28` 在 3.68 s 的实际初态与离线预检
不同：生成的时钟脚本将 `tracking-tube-end` 解释为源时间，3.68 会提前结束
跟踪 CBF。该试验已停止，不得使用其离线预检作为实际初态等价证明。

纯 WBC 记录显示，原最好版本在 4.040 s 的唯一支撑脚计划法向力只有
40.12 N，COM 预测与观测均接近向下 8.6–8.8 m/s²。新增默认关闭的
`--support-load-floor-start` / `--support-load-floor`：仅在已检测到至少 20 N 的
单脚支撑阶段，平滑增加支撑脚世界竖直力下限；双支撑、飞行和未检测到承重
的脚不增加该约束，不改变摩擦/力矩上限或制造外力。
`wbc_support_floor_clarabel_audit.json` 中，保存的 4.040 s 物理 QP 在 100 N
下限下仍可行（残差 2.98e-13）；瞬时加速度改动较大，必须实际验证，不能
据此宣称稳定。3 项约束测试通过。`wbc_support_floor100` 正从源帧 0 实测。

`wbc_support_floor100` 实际完成报告：仍为 11 次落脚，在 4.088 s／源帧
938.12 根姿态误差 0.355151 rad 越界，未提高有效前缀。因此载荷下限仍默认
关闭，未替换已验证配置。

继续纯 WBC 检查发现，两套源参考平滑尺度不一致：主控制 stride=6，而额外
跟踪管 stride=2。在源帧 780–980，后者根角加速度峰值 160.57 rad/s²，
stride=6 为 63.93 rad/s²；两条参考最大根姿态差 0.00616 rad，最大关节
RMSE 差 0.01617 rad（`wbc_tube_smoothing_audit.json`）。新增默认关闭的
晚段平滑过渡，将两条源样条用五次权重连接，并对同一条曲线解析求 q/v/a。
过渡前保持原参考精确一致，边界 q/v/a 连续，有限差分和约束相关 6 项检查
通过。`wbc_late_tube_smooth6` 正在源时间 4.2 s 后平滑切换跟踪管参考；
实际原始 PKL 验收与物理限制未改。此实验没有 MPC，载荷下限也没有启用。

`wbc_late_tube_smooth6` 完成实际报告：11 次落脚；4.084 s／源帧 937.16 的
关节 RMSE 0.200841 rad 越界，根姿态误差 0.322684 rad。没有达到完整跟踪。

转向检查控制参考落脚几何：原始源帧 884 的左脚原生最低点约 −0.208 mm，
而 Y40 控制参考整数帧为 +13.483 mm，stride=6 插值后为 +16.915 mm。
新增仅几何准备的 `wbc_native_landing880_floor3_drop20.npz`，平滑降低该段
摆脚末端并下调根高度 20 mm，以有界腿 IK 保留右脚和左脚 XY/方向。
预检最大 IK 位置残差 1e-7 m、最小密集关节余量 0.056 rad、最大密集关节
速度 13.283 rad/s（限值比例 0.664）；原始落脚窗口足位差最大 3.59 mm，
源帧 884 的插值最低点为 +1.976 mm。初次零间隙候选预检失败，未执行；
当前候选留出 3 mm 几何余量后通过。`wbc_native_landing880_trial` 正在实际
运行，不启用 MPC、载荷下限或新跟踪管平滑。尚未据此更新有效落脚成绩。

`wbc_native_landing880_trial` 实际未通过：10 次有效落脚，在 3.936 s／源帧
901.64 关节 RMSE 达 0.201775 rad。第 11 步已离地，落脚误差最小
22.57 mm，但连续承重仅 6 ms，未满足 50 ms。该候选不替换基线。
继续使用 Y40 基线，单独测试第 10 次落脚后（仿真 3.65 s）用 0.3 s
将源时钟从 1.2 倍平滑减至 0.9 倍，使用同一时钟解析生成 q/v/a；
验收仍对照该时钟对应的原始源帧，且禁用所有 MPC/preview。

`wbc_tail_slow090` 未改善：11 次落脚，仿真 4.074 s／源帧 918.32
根姿态误差 0.354145 rad 越界。减速没有替换基线。纯 WBC 入口
`run_corrected_eleven_recovery.sh` 现把两个 preview-start=inf 放在调用参数
之后，确保此入口不会被附加参数误启用 MPC。

`wbc_support_priority_audit.json` 的瞬时同状态 QP 检查显示，3.900 s 的
支撑脚自由轴目标 −16.525 rad/s²，原 QP 输出 +66.257，增加首层支撑
姿态任务后为 −15.491，物理残差约 1e-13。统一参考中的该任务跟踪源脚
姿态，并不是额外的固定水平足底目标。实际 `wbc_support_orientation415`
仍失败：10 次落脚，3.950 s／源帧 905.00 关节 RMSE 0.200075；此时
右膝源误差 −0.705 rad，虽侧向误差略减，未改善有效前缀，不替换基线。
全身统一参考相关 9 项测试和 git diff --check 通过。

### 2026-11-05：继续完整跟踪，保持无 MPC

按用户再次明确的要求，所有新实际运行均通过强制禁用 preview 的纯 Fabric + WBC
入口。新选项 `--orientation-tube-end` 默认 inf，只能结束实验辅助姿态 CBF；
原始源姿态独立验收不变。`wbc_orientation_task415` 用源时间 4.15 s 后的软
骨盆姿态任务替代该硬管，仍只有 11 次落脚，3.974 s／源帧 910.76 根姿态误差
0.351207 rad，未采用。

转身右支撑脚保持水平的几何准备中，根高度下降 0、10、30 mm 均因 IK 残差
失败，未执行；下降 40 mm 候选通过几何预检，但尚无实际跟踪证明。
新 `prepare_wbc_eleventh_clearance.py` 只降低第 11 步过高的摆脚净空，保留 XY
及姿态。下降根部 0 mm 的候选 IK 误差 <1e-7 m、密集速度占硬件限值 0.427，
几何预检通过。首次启动因候选漏写 baseline_sha256 被加载器拒绝，没有进入
跟踪；已补齐准备脚本和文件的来源哈希，用独立名称 `wbc_clearance065_trial_v2`
重新执行。`wbc_clearance065_trial` 仅是失败启动记录，不能计为跟踪试验成绩。

原最佳实际记录的第 10、11 步在卸载脚 <10 N／支撑脚 >=100 N 时原生最大
净空分别为 0.0499005 m（帧 791.24）和 0.0861376 m（帧 861.32）。第 10 步
减小净空候选也已单独准备，尚待验证；实际 0.02 m 离地验收门槛未更改。

第 11 步减小净空的实际试验 `wbc_clearance065_trial_v2` 仍为 10 次落脚，
3.958 s／源帧 906.92 关节 RMSE 0.206097 rad；第 11 步最长有效承重 40 ms。
第 10 步版本 `wbc_tenth_clearance065_trial` 为 9 次，3.732 s／852.68 帧
关节 RMSE 0.201402 rad。`wbc_support_floor250` 为 10 次，3.960 s／907.4
帧关节 RMSE 0.200624 rad。三者均未替换已验证配置。

#### 核实当前模型力矩范围与旧软件配置的不一致

`examples/test_g1_two_tasks.py::unitree_g1_effort_limits()` 返回旧
UNITREE_G1_29DOF_CFG 的软件限额：四个踝关节和腰部 roll/pitch 均为 25 Nm，
QP 再乘 0.8 得到 20 Nm。当前实际使用的 `g1_29dof_rev_1_0.urdf` 及
`g1_29dof_rev_1_0.usd` 的这六个关节均为 50 Nm；其余关节对应值一致。
因此以前记录的 25 Nm 应称旧配置降额，不能称当前资产的原生力矩上限。

新增 `examples/g1_model_effort.py` 按关节名称取当前 URDF 和运行中
`robot.get_max_efforts(joint_names=...)` 的较小值，不写入或扩大仿真模型限额。
相关 3 项测试覆盖命名顺序、六个并联驱动坐标的模型值、双重限额取小和非法
输入拒绝。实验参数 `--model-effort-start` 默认 inf，以保留历史配置；启用时
仅平滑解除旧配置的软件降额，QP 仍保留模型范围的 20% 余量，原始 PKL 和
所有落脚验收不变。每次运行写 `.effort_model.json`，并逐步记录有效 QP 上限。
`wbc_model_effort415` 从源时间 4.15 s 起过渡，正在进行实际全片验证。

模型力矩配置的第一轮实际结果（均未完成全片）：`wbc_model_effort415` 11 次／
937.64 帧，`wbc_model_effort415_body1` 10 次／909.32 帧，提前启用的
`wbc_model_effort370_body1` 9 次／842.12 帧，躯干权重 0.3 并解除同步
控制额外预留的 `wbc_model_effort415_body03_full` 10 次／929.00 帧。
完整失败指标保存在 `wbc_model_and_stance_trials.json`。第一组源时间 4.15 s
之前的 history_q 与原最佳记录逐位相同，证明限额过渡前前缀未改变。
这些试验始终未修改 USD/URDF 或超过二者和运行时限额的较小值；实际应用中
旧配置降额与模型极限必须明确区分。尚未将模型限额试验作为更好的跟踪结果。

还发现第 10 步右脚的 [15,-20] mm 落点修正原在帧 840–884（承重期间）撤回。
`prepare_wbc_stance_offset.py` 将其保留到帧 908，并在下一次摆动中撤回到帧
952；几何预检通过，参考脚与原始源的最大偏差 25 mm。实际
`wbc_stance_offset_trial` 为 11 次落脚，4.080 s／936.2 帧关节 RMSE
0.201913 rad，仍未通过。新默认关闭的 `--leg-posture-secondary-start`
用于检查腿姿态与足位姿在首层重复竞争的问题；源时间 4.15 s 后保留完整
腿姿态任务在末层，首层仅选上肢/腰部关节，原物理约束与 PKL 验收不变。

### 2026-11-05：纯 WBC 后续验证，仍未完整通过

以下实验均从原始动作第 0 帧运行，保持原始 PKL 审计阈值；两种预测控制入口均关闭。所有结果的 `full_motion_tracking_passed` 都为 `false`，没有替换已验证的 11 次落脚基线。

| 实验 | 落脚次数 | 最后源帧 | 关节 RMSE 峰值 | 根位置误差峰值 | 根姿态误差峰值 |
|---|---:|---:|---:|---:|---:|
| `wbc_leg_posture_secondary415` | 10 | 894.44 | 0.202779 | 0.119722 | 0.297437 |
| `wbc_model_effort_from_start` | 3 | 228.51 | 0.189552 | 0.200088 | 0.332670 |
| `wbc_landing_pause898` | 11 | 898.02 | 0.201722 | 0.167715 | 0.297437 |
| `wbc_tenth_descent_lead6_trial` | 9 | 812.84 | 0.201778 | 0.045247 | 0.297437 |
| `wbc_configuration415` | 10 | 929.00 | 0.196937 | 0.155022 | 0.352747 |
| `wbc_configuration_feet_model415` | 11 | 926.12 | 0.185129 | 0.158616 | 0.350166 |
| `wbc_configuration_feet_model415_rot4` | 10 | 928.52 | 0.198551 | 0.162534 | 0.351241 |
| `wbc_model_effort_smooth6` | 11 | 937.16 | 0.190299 | 0.192313 | 0.351645 |
| `wbc_unified_smooth7_model` | 11 | 937.16 | 0.185129 | 0.192402 | 0.352416 |

`wbc_angular_response_audit.json` 对比实际速度差分与 QP 的根角加速度：除接触冲击附近外两者基本一致，因此尚无证据把该段失败归因于普遍的执行器响应错误。

新增的实验选项 `--late-reference-smooth-stride/start/ramp` 使用已有 C2 曲线转换，统一生成主控制参考 q/v/a，默认关闭。离线足底检查中 stride 7、8 通过，9、10、12 未通过，不执行未通过的参考。stride 7 与源管约束同步平滑的真实试验仍在第 11 次落脚后姿态超限。平滑、单步 QP 可解或几何预检通过均不构成完整跟踪证明。

### 当前整片运行入口（纯 Fabric + WBC）

`examples/run_amp_full_motion.sh` 现在调用已验证 11 次落脚的 `run_corrected_eleven_recovery.sh` 实验配置，取代旧四步配置派生的整片尝试。该入口请求完整源帧覆盖，在参数末尾固定关闭两种预测控制并开启原始动作误差检查；不会将“尝试整片”标记为“整片已通过”。依赖本工作区 `acc_data/amp_walk_20261103/feedback_followup/` 中的实验脚本与参考文件。

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_full_motion.sh
```

默认输出 `acc_data/amp_walk_20261103/feedback_followup/full_motion_no_mpc.npz` 与同名 `.tracking.json`、`.steps.json`。整片结果必须查看 `full_motion_tracking_passed`；失败运行返回非零退出码。

后续四项实际对照均失败，未提升或替换入口配置：

| 实验 | 已验证落脚 | 最后源帧 | 关节 RMSE 峰值 | 根位置误差峰值 | 根姿态误差峰值 |
|---|---:|---:|---:|---:|---:|
| `wbc_flat_support_model` | 10 | 914.12 | 0.202245 | 0.162639 | 0.297437 |
| `wbc_model_balance_hierarchy` | 10 | 842.12 | 0.200656 | 0.045247 | 0.297437 |
| `wbc_half_flat_support_model` | 11 | 929.00 | 0.202799 | 0.177523 | 0.297437 |
| `wbc_half_flat_upper2` | 10 | 930.92 | 0.202181 | 0.177144 | 0.297437 |

半幅支撑脚姿态修正保住 11 次落脚，但最终腿部关节 RMSE 约 0.075 rad、腰部约 0.335 rad、双臂约 0.237 rad，全身 RMSE 仍超限。进一步把上半身权重加倍没有改善落脚数。当前证据指向转身段各跟踪目标与实际支撑之间的冲突，不能据此断言无 MPC 的方法不可能完成该动作。

入口参数检查确认，即便调用方传入两个预测控制起始时间为 0 或 `--tracking-steps 4`，最终生效值仍是两个 `inf` 与 0，完整动作及源误差审计保持开启。新增模型力矩读取、配置跟踪与平滑转换共 9 项单元检查通过；`bash -n` 与 `git diff --check` 通过。

整片新入口已实跑：`full_motion_no_mpc.tracking.json` 记录 11/29 次落脚，在仿真 4.088 s、源帧 938.12 因根姿态误差 0.354216 rad 超过 0.35 rad 停止；进程返回 1。保存的控制选项确认 direct WBC、两种 preview 起始时间均为 inf；源 PKL SHA256 未改变。`full_motion_tracking_passed=false`，完整跟踪仍未实现。

### 2026-11-05：继续纯 WBC，分离时间缩放与支撑力矩问题

本轮加快第 12 步并采用模型力矩范围的 `wbc_model_twelfth_retime` 仍只有 11 次落脚，在源帧 937.64 关节 RMSE 0.201956 rad 超限。半幅支撑姿态修正再加骨盆优先级的 `wbc_half_flat_pelvis` 只有 10 次落脚，源帧 925.16 关节 RMSE 0.205468 rad 超限。三轴角动量反馈的 `wbc_all_axis_momentum` 保留 11 次落脚，在源帧 925.64 因关节 RMSE 0.200137 rad 超限；该时刻根姿态误差为 0.197192 rad。三者均未替换基线。

`audit_wbc_path_retiming.py` 是离线诊断，不输出运行参考或控制信号：固定参考姿态曲线及原接触计划，在源帧 800–1000、速度 0.6–1.8 倍、每 8 帧一个三次样条节点的时间缩放范围内，检查必要的质心合力与地面力矩条件。它使用完整足底投影凸包、宽松的箱形摩擦约束，并忽略关节力矩限制，因此可行也不代表实际可跟踪。固定两端 1.2 倍速度时合力条件不可行；放开两端速度后合力可行，但两条被检查的参考仍需要约 0.1768 m 的体重归一化地面力矩松弛。这只说明所选固定路径/接触计划/速度参数化无法靠时间缩放精确满足这些条件，不是对所有 WBC 方法的不可行证明。报告分别是 `wbc_path_retiming_audit.json` 和 `wbc_path_retiming_free_audit.json`。

新增默认关闭的实验选项 `--contact-friction-refine-start`，结合 `--contact-friction-rays 8/16`：通过实际接触点生成圆锥内部的摩擦射线，对点、边、面接触均保留相同的接触几何及缺失力矩约束。边/点仍沿用原映射中的水平理想线/点，不用几何噪声制造额外力矩。摩擦系数不变，法向力上下限及原边/点偏航上限保留。5 项生成力、摩擦范围、接触力矩及载荷边界检查通过；此时尚未获得完整跟踪证明。

八方向摩擦近似实际结果：`wbc_friction8_model` 仍为 11 次落脚，源帧 935.72 根姿态 0.354666 rad 超限（此时关节 RMSE 0.179651 rad）；与三轴角动量反馈组合的 `wbc_friction8_momentum` 只有 10 次落脚，源帧 923.72 关节 RMSE 0.200515 rad 超限；提前到源时间 3.9 s 启用近似的 `wbc_friction8_before_turn` 只有 9 次落脚，源帧 899.72 姿态超限。均不推广到默认整片入口。新增极限力矩的非负接触力分解检查后，该近似共 6 项测试通过；测试通过不等于跟踪通过。

随后检查第 11 步起摆相位：初稿 `wbc_eleventh_phase_delay16` 的 20 帧淡入会使局部相位导数为负，虽然几何/速度检查通过仍不符合此修正的单调设计要求，已撤回并中止实际尝试，不计入可用结果。`prepare_wbc_eleventh_phase.py` 已补充单调性验收以拒绝该初稿。新候选 `wbc_eleventh_monotone_delay16` 使用 32 帧淡入、最小相位速度约 0.0625，双脚最小原生几何高度分别约 +0.069 mm 和 −0.697 mm，关节裕量 0.01767 rad，速度上限占用率 0.427。仅改变左脚局部参考相位，通过有界 IK 生成统一 q/v/a；原始 PKL 审计、实际起飞/承重/落脚证据保持不变。

单调起摆修正实测 `wbc_eleventh_monotone_delay16_trial` 保留 11 次落脚，源帧 931.40 处关节 RMSE 0.202976 rad 超限（根位置 0.177600 m，姿态 0.280478 rad）；第 12 步尚未取得有效起飞证据。也不替换默认入口。

本轮共完成 7 组新的全程起始实测对照，另撤回 1 组不满足单调相位设计的尝试；保存的控制选项逐项确认全部为 direct WBC、两种预测控制起始时间为 inf。6 项摩擦近似检查通过，改动脚本可编译，`git diff --check` 通过。当前最佳仍是原来的 11/29 次落脚、源帧 938.12；完整跟踪尚未实现，完整汇总见 `wbc_continuation_status.json`。


### 2026-11-05: twelve verified landings with direct WBC, no MPC

The full-motion entry now uses `run_corrected_twelve_no_mpc.sh`. This is an
improved experimental prefix, **not a full-clip success**. The original PKL,
all 29 landing requirements, and the original pose/error thresholds are unchanged.

`wbc_convex_catch_foot4.npz` and its independently recomputed
`wbc_convex_catch_foot4.strict_prefix.json` verify **12 ordered landings** while
all preceding original-source pose samples pass. Landing 12 occurs at 4.216 s,
source frame 968.84, with 7.52 mm foot-link error. Prefix peaks through that
landing: joint RMSE 0.185129 rad, root position 0.176128 m, root orientation
0.297436 rad. Step 13 has measured flight, but the run stops at 4.300 s / frame
989.0: root position error 0.201006 m exceeds 0.2 m. The final
`full_motion_tracking_passed` remains false.

The new profile uses the existing convex wrist/body collision geometry from
source time 3.5 s, verified native model effort limits from 4.15 s, bounded
instantaneous margin weighting from 4.4 s, and increased swing task weight from
4.5 s. It does not run a predictive controller. A bounded IK guide moves the
11th landing by [-15, -25] mm and holds that offset until the next left takeoff;
it also adds up to 35 mm clearance during swing 12. One interpolated q/v/a
curve supplies all reference tasks. Independent acceptance still uses the
unmodified original source and native physical contacts.

`turn_collision_geometry_audit.json` records a structural conflict in the old
sphere covers: among 23 sampled source poses from 3.2 to 4.08 simulation seconds,
14 have negative optional wrist/body sphere distance while the corresponding
minimum convex-mesh distance is at least 20 mm. Actual actuator acceleration
checks show the arm response generally follows the QP; contact impacts cause
larger ankle discrepancies. Replacing the conservative covers earlier reduces
arm error and advances landing 10 by 54 ms. Convex checks do not disable collision
barriers or alter robot geometry.

New optional helpers are `tracking_margin.py` (measured error and velocity-based
soft weights, bounded by 16) and `early_resume_clock.py` (C2 phase recovery after
verified landing, with nonnegative speed and bounded recovery acceleration).
Each has three passing unit tests. The early-resume clock improves over a fixed
pause but is **not selected** in the twelve-landing profile. Several intermediate
trials still fail; none are reported as complete motion tracking.


### 2026-11-05: continuation diagnostics after the twelve-landing result

The selected profile remains `run_corrected_twelve_no_mpc.sh`. Subsequent
changes to support-load floors, momentum regulation, body/leg task allocation,
turn timing, and root-reference transition timing did not improve the certified
12/29 prefix. Every completed trial and its independent original-source prefix
audit is retained in
`acc_data/amp_walk_20261103/feedback_followup/wbc_contact_tracking_status.json`.
Unselected experimental flags keep their inactive defaults.

Two diagnostic continuations checked whether the first fidelity violation was
followed by recovery. With the selected controller, the original root-position
limit was crossed at 4.300 s; both feet subsequently lost support and the QP
failed validation at 4.418 s (source frame 1017.32). The invalid result was not
applied. The diagnostic reached peak source joint RMSE 0.411 rad, root-position
error 0.265 m and root-orientation error 0.512 rad. Disabling the auxiliary
tracking tubes after source time 4.85 s also failed to recover: the robot fell
and the physical QP failed at 4.740 s. These are failed diagnostics, not completed
tracking results; all original-source acceptance limits remain unchanged.

The diagnostic-only `--diagnostic-continue-after-fidelity-loss` option changes
termination timing and records the first violation. The public
`examples/run_amp_full_motion.sh` appends `--stop-at-fidelity-loss` after caller
arguments, so that option cannot silently disable the public entry's stop rule.
The public entry also appends both disabled-preview settings, full-source mode,
and zero bounded-step count after caller overrides; an argument-order check
verified those guarantees.

Collision response bandwidth is now an explicit optional scalar (default 8,
exactly preserving previous bounds). The experimental 32 setting certified only
10 landings and is not selected. Unit tests verify the original formula,
activation masks and invalid-parameter rejection; the four convex-geometry
regressions pass. The eight dynamic-step audit tests also pass. A separate
experimental torso-to-pelvis weight transfer preserves references and total
orientation weight, but its physical run still stopped at 12 landings.


The public default output `full_motion_no_mpc.npz` was rerun after these changes.
Its state, velocity, source-frame and measured-foot-force histories are bitwise
identical to `wbc_twelve_reproduced.npz`. Its independent report still declares
`full_motion_tracking_passed=false` and 12/29 verified landings; it stops at
simulation time 4.300 s / source frame 989.0.

An additional offline whole-remaining-clip COM preparation experiment did not
produce an approved reference. Vertical acceleration bounds were feasible, but
horizontal planning with fixed reference angular momentum, prescribed source
support polygons and a 0.12 m COM-offset bound was infeasible for the tested
1.2-speed and smoothly reduced 1.0-speed schedules. HiGHS independently reported
infeasibility for the constrained reduced 1.0-speed formulation. Its separate
without-friction diagnostic returned an **unknown** solver status, so no
infeasibility claim is made for that relaxed case. These restricted-model results
are not a proof that the full robot motion is impossible. No candidate from
these planners was used in the runtime profile. Reports are
`wbc_remaining_com_plan.json`, `wbc_remaining_slow1_plan.json` and
`remaining_com_feasibility_audit.json` in the same experiment directory.


### 2026-11-05 continuation: transfer diagnostics and reference consistency

Nineteen additional Isaac Sim runs requested the full source with direct WBC and
both previews disabled. None passed the thirteenth ordered landing. The selected
public profile remains the independently reproduced 12/29 baseline. The source
PKL hash remains `a870f6642380a5d7e23ae34fbc8a9614bf4164d6a9ce0a26432ae35f3401b8c4`.
The independent summary now contains 61 completed physical trials.

| New trial | Verified prefix | Last source frame |
| --- | ---: | ---: |
| `wbc_twelve_forward_out20_trial` | 12 | 990.92 |
| `wbc_thirteen_contact_clock` | 12 | 954.92 |
| `wbc_twelve_com_recovery39` | 12 | 989.48 |
| `wbc_twelve_slow_before_turn` | 7 | 659.12 |
| `wbc_twelve_clearance10_trial` | 12 | 993.32 |
| `wbc_tenth_angular4_trial` | 9 | 917.48 |
| `wbc_low12_roll08_native` | 12 | 994.28 |
| `wbc_tenth_angular_window_back20` | 9 | 941.96 |
| `wbc_tenth_out33_low12_trial` | 11 | 1008.68 |
| `wbc_out33_pelvis20_native` | 12 | 997.16 |
| `wbc_out33_support150_native` | 12 | 1002.92 |
| `wbc_out33_soft_position` | 12 | 971.24 |
| `wbc_transfer_bridge808_trial` | 10 | 914.12 |
| `wbc_transfer_left_lead8_trial` | 10 | 973.16 |
| `wbc_transfer_left_forward30_trial` | 11 | 988.52 |
| `wbc_out33_foot42` | 12 | 1001.48 |
| `wbc_out33_early_pelvis10` | 12 | 1001.00 |
| `wbc_transfer_collision_iter1_physics` | 8 | 661.64 |
| `wbc_transfer_collision_iter1_nativeplan` | 11 | 911.72 |

All recordings, controller options and independent `.strict_prefix.json` reports
are under `acc_data/amp_walk_20261103/feedback_followup/`. A later source frame is
not a better result when an earlier landing fails. In particular, the 1008.68
frame run missed step 12 by at least 75 mm and therefore remains an 11-step result.

Reducing the additional step-12 lift from 35 to 10 mm reduced the measured right
landing force peak from 1509 to 1371 N and preserved the required flight/landing
evidence. It still failed original root orientation before step 13 landed.
Increasing foot orientation weight throughout the turn caused a roughly 51 mm
forward landing error at step 10. Restricting that weight to the right-foot swing
and correcting its reference backward did not recover a valid landing. Optional
angular-weight window/side flags therefore remain inactive by default. The
separate `--position-tube-end` diagnostic neutralizes only the auxiliary root
position acceleration row; physical limits and original-source acceptance remain
active. Removing that row earlier caused a root-position failure at frame 971.24.

The offline upper-body reference fit uses COM, root orientation and 17 upper
joint coordinates, with both original guide feet reconstructed by IK. A small
linearized ground-moment residual did **not** certify its reconstructed q/v/a:
the dense audit rejected excessive force, support-polygon error, and/or native
sole penetration. No such rejected candidate was executed or selected.
`reference_smoothing_basis.py` now applies the runtime stride-6 smoothing operator
to the planning basis. Two tests compare pose, velocity and acceleration with
the actual runtime sampler, including clock acceleration, and reject fractional
endpoint sampling that cannot be represented by this linear operator. Both pass.
The knee-reserve variant also removed the earlier nonlinear IK reach failure;
this alone does not establish dynamic feasibility.

The source directly switches support at frame 802 (step 11 takes off there).
A continuously differentiable reduced model with positive vertical force cannot
instantaneously move its COP between disjoint single-foot polygons. This is a
limitation of the particular offline formulation, not proof that the full robot
motion is impossible. Preparing a short geometrically supported transfer, followed
by left-foot timing/position compensation, certified at most 11 steps in physics.
The original source contact labels and runtime measured-contact gating were not
changed. Earlier small-weight pelvis feedback preserved 12 landings and reduced
terminal orientation error to 0.324 rad, but root-position error reached 0.201 m.

Continuous motor angle and speed checks now evaluate exact polynomial extrema,
including extrema between sampled frames. Their two regressions and the two
smoothing-basis regressions pass. The rejected `wbc_transfer_curve_fit` had positive
continuous motor margin (0.01098 rad), maximum speed fraction 0.7322, native sole
clearance -0.00167 m, and normal force range 139.2–781.5 N. Its independently
reconstructed COP still violated the support polygon by 0.04984 m; it was not run.

`runtime_centroidal_linearization.py` differentiates fixed-foot IK before the
runtime spline, including quaternion normalization. This corrects the earlier
approximation that smoothed the coordinate basis before applying the pose map.
The independent finite-difference regression re-solves IK for positive and
negative perturbations, then compares the actual runtime velocity, COM acceleration
and angular-momentum rate. It passes. A second regression compares the accelerated
offline MuJoCo centroidal evaluator with the independent NumPy URDF implementation
across ten source poses, with absolute tolerance 1e-8. No MuJoCo simulation steps
are used, and the Isaac controller is unchanged by these offline tools.

The collision-constrained transfer candidate passed the dense nonlinear dynamics
preflight (COP violation 0, friction ratio 0.50025, normal force 139.99–780.01 N,
continuous joint margin 0.00996 rad), and its modified turn interval passed the
sampled native convex distance check. Running it with the extra prepared contact
intentions reduced the result to 8/29 because the plan changed many early source
frames. Running the same q/v/a candidate with the native contact plan reached
11/29 and stopped before step 12: the swing foot never obtained the required
0.02 m unloaded clearance. It therefore remains rejected and the public 12/29
profile is unchanged.

### 2026-11-06: continue through `examples/test_g1.py`

The public launcher now runs `examples/test_g1.py --tracking-profile amp-full`.
This explicitly selects the measured twelve-landing direct-WBC configuration,
including its prepared guide and read-only helpers under
`acc_data/amp_walk_20261103/feedback_followup/`. It requests every source frame
and all 29 ordered landings. The original raw PKL remains unchanged. This is
an executable full-clip attempt, **not a complete tracking result**.

Run with the Isaac environment active:

```bash
bash examples/run_amp_full_motion.sh
```

`G1_PYTHON` can select the environment's Python. Results now go to
`acc_data/g1_full_tracking_20261106/full_motion.*`. Ordinary calls without the
named profile retain the original QP and clock; inherited experimental timing
stages are disabled by default. The original file is backed up as
`acc_data/g1_full_tracking_20261106/test_g1.before.py` with a recorded SHA-256.

The migrated baseline reproduced the old state, velocity, time and force
histories exactly: 12/29 landings, stopping at 4.300 s / source frame 989.0
because root-position error reached 0.201006 m. Independent raw-PKL audits
recheck joint/root poses and landing positions; measured flight and hold evidence
are retained from the step tracker. No tolerance has been relaxed.

New disabled-by-default diagnostics in `test_g1.py` are a landing-envelope
activation time, explicit physics frequency with integer control/physics ratios,
and a native-sphere rolling-contact bias correction. For a sphere on a horizontal
floor, its current contact offset from the collider center is `-radius*world_z`.
The correction subtracts the rigid-material-point centripetal term only when
all observed patch vertices match native sphere bottom points. It changes no
wrench/Jacobian map and does not model an impact impulse. Independent finite
differencing verifies the kinematics, including a rotated collider offset.

| Full-source trial | Verified ordered prefix | Last source frame |
| --- | ---: | ---: |
| Baseline | 12 | 989.00 |
| Contact damping 40 from 3.5 simulation seconds | 10 | 906.44 |
| Landing speed 0.5 m/s from 3.8 simulation seconds | 10 | 965.00 |
| Horizontal COM first from source time 3.5 s | 9 | 793.16 |
| PhysX velocity iterations 8 | 2 | 120.40 |
| Synchronous 1000 Hz control/physics | 1 | 233.63 |
| Rolling correction from 3.5 simulation seconds | 12 | 1002.44 |
| Rolling correction plus pelvis priority | 11 | 1002.92 |
| Rolling correction from initialization | 1 | 389.96 |
| Rolling correction plus secondary stance posture | 10 | 858.44 |

The rolling correction preserves twelve valid landings and advances the failure
to 4.356 s. It still fails: root orientation is 0.355206 rad versus the unchanged
0.35 rad bound, and step 13 has flight but no loaded landing. It is optional,
not selected as the default or claimed to solve full tracking. The combined
pelvis variant reaches a later source frame but loses landing 12 and is worse.

All commands, source hashes, recordings and independent reports are in
`acc_data/g1_full_tracking_20261106/`. `audit_trials.py` and
`audit_raw_source.py` regenerate the acceptance reports. The targeted suite has
39 passing tests covering profile guards, rolling kinematics, contact state,
wrench cones, dynamic initialization, source auditing and landing envelopes.
The outstanding work is still physically consistent tracking through step 13
and the remainder of the original clip; full tracking remains false.

The final public-launcher rerun (`launcher_baseline`) is bitwise identical to
the historical baseline in q, qd, source frame, time and measured foot forces.
A separate 6 ms ordinary-entry smoke run completes successfully without the
AMP profile. All eleven full-source requests fail full tracking and correctly
return nonzero exit codes; the short ordinary smoke is not a tracking claim.

### 2026-11-06 follow-up: contact diagnostics and turn reference timing
Seven more full-source Isaac trials ran through `examples/test_g1.py`. None
passed the thirteenth ordered landing or full-motion acceptance. The public
profile remains unchanged. The raw PKL and all original acceptance thresholds
are unchanged. A later failure frame is not a completed additional step.
| Trial | Verified prefix | Last frame (failed sample) |
| --- | ---: | ---: |
| `rolling_support150` | 12 | 1003.88 |
| `rolling_pelvis_after12` | 12 | 1002.92 |
| `rolling_contact_damping8` | 12 | 1001.96 |
| `rolling_stance_after12` | 12 | 1005.80 |
| `rolling_stance_weight50` | 11 | 1003.40 |
| `rolling_support_orientation_after12` | 12 | 1002.44 |
| `rolling_turn_offset_hold1070` | 12 | 1002.44 |

The furthest twelve-landing trial ends at 4.370 s / frame 1005.80, with
root-position error 0.200517 m and root-angle error 0.351857 rad. Both original
limits are exceeded, and step 13 has no verified loaded landing.

Contact diagnostics now retain the actual mapped contact velocity, active
constraint rows and acceleration bias. These values do not alter the QP.
During the inspected late single-support interval, contact translations are
usually nearly stationary, while point/edge rotational freedoms change and
observed angular acceleration increasingly differs from the one-step QP
prediction. This difference includes impacts and time discretization; it does
not identify one unique root cause. A late 8/s contact-velocity damping trial
did not improve the result. The new recordings report no self-contact events.

The default-disabled `--stance-posture-metric-start`, `--stance-posture-metric-scale`
and `--stance-posture-metric-ramp` options test a continuous compromise between
source posture and balance. They preserve the task hierarchy and use a metric
congruence to retain symmetry and positive semidefiniteness, including cross
terms. Swing legs and upper-body weights are retained. The 0.5-weight physical
trial lost landing 12; it is not selected as a better controller.

A separate candidate delays removal of the existing 4 cm lateral root offset
from frames 940–1000 to 1070–1150, retaining both guide feet with bounded leg
IK. `prepare_turn_offset_hold.py` audits the actual runtime spline, including
exact continuous joint position/velocity extrema. Preflight passed: guide-foot
change at dense samples 0.000373 m, minimum continuous joint margin 0.017201 rad,
maximum joint-speed fraction 0.557148, and minimum sampled native clearance
-0.001767 m. This is a kinematic check, not a dynamics certificate. Its physical
run still failed before landing 13, so the candidate is not selected.

The targeted suite now has 41 passing tests. The two added tests check dense
metric residual energy, untouched swing/upper-body coordinates, positive
semidefiniteness, flight behavior and invalid weights. Commands, code hashes,
recordings, independent raw-source audits and `continuation_report.json` are
under `acc_data/g1_full_tracking_20261106/`. Full tracking remains false.

### 2026-11-06 follow-up: task allocation, response audit and reference forces

The continuation remains in `examples/test_g1.py`. The original PKL, full
2000-frame / 29-landing acceptance, collision model, motor limits and independent
source-pose thresholds are retained. The default `amp-full` profile has not been
replaced by an experimental result.

A real history-serialization failure was found when a priority stage disappears:
previous samples contained `priority_stage_constraint_error` and
`priority_lock_error`, while subsequent samples omitted them. Saving the mixed
history raised `KeyError`. Both fields are now present on every sample, with
NaN and `priority_active=False` when no priority stage exists. A regression test
covers active/inactive/active histories, and the corrected physical run saved
its final tracking failure normally.

Three default-disabled experiments are available for reproducible diagnosis:

- `--root-position-task-start`, `--root-position-task-weight` and
  `--root-position-task-ramp` add a soft world-frame root-origin acceleration
  request, using the same reference q/v/a. They apply no external base force.
- `--support-flatten-start` blends the observed patch's soft level-sole request
  into its source-orientation request only for a measured loaded, planned stance
  foot. It preserves the actual point/edge constraints and wrench cone.
- `--weighted-tasks-start` and `--weighted-tasks-ramp` smoothly include secondary
  task costs in a single physical QP. Metric congruence preserves positive
  semidefiniteness and handles partially selected primary rows. It does not
  relax any physical constraint or the independent tracking criteria.

Saved QP/response pairs show zero difference between requested total torque and
actually applied torque in the inspected synchronous runs. Some inspected root
acceleration predictions closely match the next physical sample, even while the
requested task accelerations have large residuals. This rules out a torque
ordering/clipping discrepancy at those samples; it does not rule out impact and
contact-model errors elsewhere. `applied_response_audit.json` retains the actual
numbers rather than claiming global model agreement.

The current nominal reference was also reconstructed independently. At six
recorded turn samples, maximum q/v/a discrepancies are approximately
2.15e-7 / 4.22e-7 / 1.50e-5, consistent with float32 runtime output. Rigid-body
inverse dynamics then gives a necessary net-force check: at source frames
791–798 the exact nominal curve requests negative normal force (minimum about
-81 N in the 740–880 interval). At 1.2x source speed, 22 of 123 samples in
955–1073 exceed a diagnostic circular friction budget of 0.6. This budget is not an
exact re-solve of the changing QP cones; the ground material is configured with
static/dynamic friction 1.0/0.8. Negative normal force alone proves that those
nominal q/v/a samples cannot be followed exactly without external assistance,
but does not prove that bounded-error tracking is impossible.

A continuous 1.2-to-0.9 clock transition over simulation time 3.1–3.4 s removes
negative normal force in the inspected turn interval and those sampled circular-budget
exceedances in 955–1073. Its actual run still fails before landing 11. A necessary
force check alone therefore cannot certify contact transitions or full tracking.
The recorded reference, exact clock settings and sampled forces are retained in
`reference_force_audit.json` and `retimed090_force_audit.json`.

The targeted suite passes 60 tests, including five new tests for moving-root
feedforward, bounded feedback, metric interpolation and changing history schema.
`task_followup.patch` is the exact main-file change against this continuation's
starting file. `task_followup_report.json` records each experiment separately,
including the rejected diagnostic-argument invocation and the initial history
serialization failure; neither is counted as a completed physical tracking run.

Completed physical results (the final frame is the failed sample):

| Trial | Verified prefix landings | Final source frame |
| --- | ---: | ---: |
| `root_task415` | 11 | 970.76 |
| `root_task415_weight300` | 10 | 949.16 |
| `root_task415_support` | 11 | 953.48 |
| `flatten_support415_run` | 12 | 988.52 |
| `physics2000_servo` | 0 | 47.60 |
| `weighted_tasks415_history_fixed` | 11 | 944.84 |
| `retimed090_before_turn` | 10 | 886.04 |
| `weighted_after12` | 12 | 1004.36 |
| `weighted_root_after12` | 12 | 1003.40 |
| `weighted_root_prepare12` | 11 | 972.20 |

None of the ten completed physical trials passed full tracking. The best
prefix remains 12/29. In `weighted_after12`, the first 2101 measured q/v
samples through 4.200 s are bitwise equal to `rolling_contact`.
That run later stops at 4.364 s / frame 1004.36 with joint RMSE 0.141793 rad,
root-angle error 0.328694 rad and root-position error 0.200931 m.
The root-position violation alone is sufficient to reject it. Adding a
late root-position task or moving the combined tasks earlier did not
produce a verified thirteenth landing. No experimental configuration
is selected as a completed tracking solution.


### 2026-11-06 continued: force feasibility, native substeps and stance heading

Full tracking is still **not achieved**. The source PKL and the 2000-frame,
29-landing, same-frame fidelity acceptance are unchanged. The selected
`amp-full` profile remains unchanged. New experiments are disabled by default.
Results and rejected preflights are recorded separately in
`acc_data/g1_full_tracking_20261106/continued_tracking_report.json`.

A COM-only offline fit eliminated negative reference normal forces in the
inspected interval, but its complete physical trial stopped after nine landings
at source frame 806.12. An independent ground-wrench audit found that even the
fitted curve's required center of pressure can lie about 0.482 m outside a
*generous full-sole* support polygon. Positive normal force is therefore an
insufficient preflight. A later force-and-ground-moment fit was rejected before
simulation because leg IK exceeded its position tolerance; it is not counted
as a physical trial or a tracking solution.

Lowering the reference pelvis by 45 mm and advancing the thirteenth swing's
vertical descent preserved dense foot XY geometry and passed its kinematic
preflight. The actual trial reduced right-foot load oscillation but did not
produce a thirteenth landing: it failed at frame 999.08. The larger 55 mm
pelvis / 50 mm foot candidate failed the reference preflight and was not run.
Promoting the swing task after landing 12 produced a physically infeasible QP
at simulation time 4.282 s. Independent Clarabel and HiGHS feasibility checks
agreed, including diagnostic versions without the auxiliary root tube or
joint-speed inequalities. No unvalidated torque was applied.

`--late-physics-substeps-start` and `--late-physics-substeps` optionally split
one 2 ms torque interval into native physics steps. The same effort is issued
before each substep; contact impulse conversion uses the actual native step
length. Control/reference time still advances by the full torque interval.
The implementation checks elapsed simulation time and records both periods.
Starting four substeps at 3.5 s reduced verified landings to ten; starting after
landing 12 retained twelve and failed at frame 1002.44. Reissuing the held effort
and relying on the native effort buffer produced bitwise identical physical
states in the compared runs. Smaller physics steps alone did not fix tracking.

`--orientation-tube-start` / `--orientation-tube-radius` can tighten the
auxiliary orientation constraint while retaining the original 0.35 rad audit.
Combining a 0.15 m position tube with a 0.28 rad orientation tube failed at
frame 981.80. Other completed trials and failure frames are in the JSON report;
none qualifies as complete tracking.

The right support foot's measured yaw after landing 12 is approximately
-0.72 rad, while its source yaw is approximately -0.41 rad. The default-off
`--stance-heading-start`, `--stance-heading-limit`, and `--stance-heading-ramp`
experiment freezes a measured loaded heading once per source stance. It adapts
reference leg coordinates while preserving the foot-link position and the
reference root/upper body, with quintic fade-in/fade-out and differentiated
reference rates. It does not move physical contact anchors or change raw-source
audits. The earlier sole-point rotation formulation was rejected by derivative
tests near a straight-leg configuration. The foot-link-preserving version
passes loaded-observation, geometric-invariant and derivative tests. Its first
physical run (`stance_heading_turn_v3`) still failed after twelve landings at
frame 1002.44. Two preceding launch errors are not physical trials.

The baseline response audit in `turn_com_feedback_audit.json` also shows
persistent lateral COM lag beginning around simulation time 3.7 s. At many
samples the QP produces substantially less lateral acceleration than the task
requests, and measured acceleration closely follows that QP result. This is
sample-specific evidence of task allocation / physical feasibility limitations,
not proof that all contact-model errors have been eliminated. A new optional
`--late-balance-hierarchy y` selects only the world-Y COM row for the first
priority stage; its physical validation remains experimental.


Further checks distinguish instantaneous feasibility from sustainable tracking.
`lateral_authority_audit.json` independently verifies that four saved states can
satisfy their lateral COM requests under the saved constraints. In the actual
`lateral_com_first_before_turn_v2` run, prioritizing only world-Y COM produced
large upward acceleration, subsequent support loss, and a rejected QP at
3.770 s (ten verified landings). Prioritizing all three COM rows with a 100 N
single-support load floor instead failed raw joint fidelity at frame 859.40
(ten landings). Neither hierarchy is selected. The original raw-source checks
remain active and no rejected QP command is executed.

The `twelfth_heading_hold` offline candidate completes the incoming right foot
heading before touchdown, preserving its foot-link path with bounded IK.
Its dense foot-position change is at most 0.001389 m, continuous joint margin
0.05927 rad, speed fraction 0.53838 and raw joint reference RMSE 0.14258 rad.
The complete physical run still fails after twelve landings at frame 1002.44.
The `turn_angular_smooth16` candidate passes kinematic checks but does not
clearly improve necessary ground-wrench behavior; no physical run is selected
for it. Kinematic preflight success is not a dynamics certificate.

`--torque-continuity-start`, `--torque-continuity-weight` and
`--torque-continuity-ramp` enable an optional soft effort-continuity cost inside
each QP stage. The target is the preceding validated joint torque. All stages
share that target; retries do not accumulate the objective. The exact modified
QP is saved for independent replay. No torque output is filtered after solving,
and no physical constraints, solver acceptance or source audits are relaxed.
Analytic-optimum tests verify the objective, unchanged constraints, stage
consistency, retry behavior and invalid inputs. Physical effectiveness remains
under test. `torque_variation_audit.json` records the measured increase in
per-step effort variation near the baseline failure, and
`torque_continuity_sweep.json` records independent saved-state solves.


The two effort-continuity physical trials did not improve coverage:
`torque_continuity10_turn` failed joint fidelity at frame 977.96;
`torque_continuity100_after12` failed root orientation at frame 1001.96.
The latter's 2111 recorded state/velocity/effort samples through 4.220 s
are bitwise identical to the rolling-contact baseline.

An eight-ray contact experiment also retained only twelve landings and failed
joint fidelity at frame 990.92. On finding the existing equivalent
`--contact-friction-refine-start` / `--contact-friction-rays` interface, the
new duplicate implementation was archived and removed from the active entry.
Its exact sources are in `contact_rays8_trial_sources/`; the existing helper
and its normal-load/yaw limits are unchanged. No friction coefficient or
native material property was increased.

`waist_stop_audit.json` identifies a repeatable late event: baseline waist roll
reaches approximately 0.396 rad with outward speed 4.8 rad/s at 4.30 s, then
hits the actual 0.52 rad stop. The baseline is within 0.002 rad of that stop
at 4.326 s; several root-margin variants reach it earlier, at 4.274–4.284 s.
This is not claimed as the cause of every failure: `balance_pose_after12`
fails joint fidelity at frame 980.84 without having reached the stop.

Enabling the existing upper-body braking constraints at source time 4.75 s
produced a rejected QP at simulation time 4.286 s, with twelve landings and
source fidelity still passing. Independent LP checks show that saved failure
is infeasible with all upper-body braking rows, but feasible with the waist
rows alone. This counterfactual was not executed. The new optional
`--physical-braking-waist-only` scopes those *additional* prediction bounds
to the three waist joints. It cannot narrow bounds when the global
`--acceleration-and-braking-limits` option is enabled. The robot's physical
joint stops, existing speed limits, torque/contact constraints and raw-source
acceptance remain in place. Physical validation is ongoing.


Step-13 investigation (2026-11-06): the verified prefix remains **12/29**.
The runs in `acc_data/g1_full_tracking_20261106/step13_progress.json` all start
at source frame zero. The default `amp-full` profile and launcher have not
been promoted to an unsuccessful candidate. Faster/slower clocks, increased
waist posture weights, full model effort within native caps, a flattened-foot
reference and a local COM-force reference have not established landing 13.
The flattened-foot and COM-force candidates passed their recorded kinematic
preflights; those preflights are not physical tracking certificates.

`--support-knee-min-angle`, `--support-knee-guard-start` (source seconds), and
`--support-knee-guard-bandwidth` add an optional acceleration inequality on
active support knees. The option defaults off. It leaves physical joint,
contact and torque limits and raw-source acceptance unchanged. The
`thirteenth_support_knee20` run satisfies the added rows (maximum recorded
violation zero), and its 2111 samples through 4.220 s are bitwise identical
to the rolling-contact baseline. It nevertheless stops at frame 1001.96
with root orientation error 0.352765 rad and only twelve landings. A 0.30 rad
auxiliary orientation tube also fails before landing 13. The knee constraint
therefore has no demonstrated coverage benefit.

The knee run supplies a sharper failure observation: at 4.328 s, waist roll
is 0.5173 rad with outward speed 4.813 rad/s. At 4.330 s it reaches the real
0.52 rad stop and its velocity is arrested. Across that interval the root
angular-acceleration prediction error is approximately
[666, -1309, 185] rad/s^2. This precedes the orientation-fidelity failure;
it does not prove that avoiding this stop alone will complete the step.
Earlier waist braking at source time 4.0 s rejects the QP at simulation time
4.232 s, after twelve landings while raw-source fidelity still passes.
`step13_waist_stop.json` independently identifies the auxiliary orientation
acceleration row as sufficient to make both saved waist-braking failures
infeasible. Removing it in a physical run instead fails the unchanged raw
orientation check earlier, at frame 967.40; feasibility alone is insufficient.

Reference-generation failures now preserve explicit false completion fields
and, when sufficient history is present, produce the same prefix landing and
full-tracking reports as QP failures. A failed state is not counted as a valid
history sample. Regression tests cover preserved prior landings and empty
history. The 114-test suite, including the optional DAQP tests and the new
support-knee mathematical checks, passes with zero skips. These are software
checks, not a claim of physical tracking success.


The global soft-avoidance weight 200 experiment changed early behavior and
failed at frame 467.24; it does not isolate step 13. The optional
`--joint-limit-priority-scale` (default 1) now scales the selected soft braking
rows only after the existing source-time activation gate. Before that gate,
the original weight is preserved. With upper-only selection, the additional
scale affects only the upper-body block. The factor-10 trial with baseline global weight 20 preserves all recorded
state, velocity, torque and frame samples through 4.136 s bitwise. It still
reaches the waist stop at 4.340 s and fails orientation fidelity at frame
1001.96; twelve landings remain the verified limit. Thirty-five focused profile/joint-control
regressions pass after this scoped option was added.

### 第 13 步：转向任务与离线参考检查（2026-11-06，继续）

追加的真实物理试验仍未通过第 13 次落地，未更新默认配置：

- `thirteenth_yaw_priority_turn`：12 次落地，源帧 977.48 时关节 RMSE
  0.201299 rad 超限；根朝向误差降到 0.057766 rad，但不能据此判为通过。
- `thirteenth_restore_torso_metric`：11 次落地，源帧 995.72 时根朝向
  0.354743 rad 超限。
- `thirteenth_yaw_torso_feedback`：12 次落地，源帧 976.04 时关节 RMSE
  0.201679 rad 超限。该段命令已经是双支撑，不能归因于漏接纳落地接触。
- `thirteenth_late_yaw_waist`：12 次落地，源帧 1003.40 时根朝向
  0.358590 rad 超限。
- `thirteenth_smooth12_turn`：11 次落地，源帧 986.60 时关节 RMSE
  0.200748 rad 超限。

离线逆动力学复核发现，固定原根部/腿部，仅修正上身不能解决所有电机限幅。
`step13_upper_wrench_fit25_audit.json` 的源帧 960 左髋 pitch 需求约
113.5 Nm，超过 80% 预算的 70.4 Nm。局部平滑左摆动腿后降到约
81.3 Nm，处于原模型 88 Nm 上限内，但超过原 80% 预算；尚未据此修改
默认预算。局部平滑候选的 `clearance_audit` 检查实际最低离地高度，
而不是禁止空中轨迹任何向下变化；原物理验收的腾空、落地和 PKL 误差
阈值均保持不变。sigma=5 的最大脚路径变化为 15.52 mm，落地附近变化
2.62 mm，最差向下变化发生在仍离地约 46.3 mm 的空中样本。

半帧间隔检查覆盖到源帧 1035，并在源接触标签从 1013 帧变为双支撑后
显式分配双脚接触力。`step13_smooth5_contact_wrench_source_cost.json`
仍未通过：最差归一化力/力矩残差 0.09924，腰 pitch 需求 52.66 Nm
超过原模型 50 Nm，上身拟合后的原始源关节参考 RMSE 达到 0.19894 rad。
该候选仅保存为拒绝/未执行诊断，未下发 Isaac，不能计入 tracking。

新的 `fit_step13_whole_body.py` 在固定根朝向的前提下，联合小幅调整
根平移和 29 个关节，并保持两脚参考路径；直接用二阶锥约束将原始源
关节参考 RMSE 限制在 0.18 rad 内。它只做本地离线参考拟合，仍需
独立几何检查及从第 0 帧开始的 Isaac 验证。名义整脚面动力学可行
不等于实际点/线接触可行，更不等于第 13 步完成。

联合拟合的第一轮 (`step13_whole_body_fit`) 在半帧采样上将最大归一化
动力学残差降到 0.05966，但仍不是可执行结果。独立 HiGHS 以每 0.125 帧
重新分配接触力后，在落地前的 1012.875 帧发现右踝需求 61.58 Nm，
超过 50 Nm；随后加入 1013 帧左极限样本，边界拟合残差降到约 0.028。

独立凸体检查同时发现第一轮联合拟合在第 973 帧将左橡胶手与左髋 roll
凸体重叠约 40.6 mm；父参考在检查区间的最小对应距离仍有 30.4 mm。
因此停止了无碰撞约束的边界拟合，保留 checkpoint 和中断原因；未执行
该候选。`fit_step13_whole_body_collision.py` 将运行时使用的原球体对和
新增手腕/身体凸体对、20 mm 距离要求一起加入离线约束，并在非线性
复核中重新计算距离。求解中的软松弛仅用于寻找参考，最终预检查仍要求
物理残差消失及碰撞距离达标；不会把松弛带入真实控制器。

证据：`step13_whole_body_fit.candidate_audit.json`、
`step13_whole_body_fit.collision_audit.json`、
`step13_whole_body_boundary.interrupted.json`。

继续的零松弛拟合补入了足坐标系转动对摩擦/力矩约束的导数，8 个有限差分
核对样本最大绝对差约 2.4e-10。加入迭代步长惩罚后，
`step13_whole_body_proximal.json` 在自身采样点上的最大残差降到 4.41e-5，
最小碰撞距离约 20 mm，原始源关节参考 RMSE 仍为 0.14258 rad。
但更密的独立检查拒绝了这版：1008.75 帧的归一化残差 0.003947，
963.75 帧的最小碰撞距离 19.9754 mm。该版没有进行物理试验。

`fit_step13_whole_body_dense.py` 补入这些采样点，额外收紧名义力矩预算
0.05%、接触不等式归一化余量 0.0005、碰撞距离余量 0.2 mm；这仅收紧
离线拟合，不放宽原控制器或原始 PKL 验收。独立审计先将参考量化到
控制器实际加载的 float32，再重新计算 q/v/a 和刚体动力学。

另一个需要保留的限制：名义参考右支撑脚在第 960 帧最低点仍以约
0.55 m/s 运动，其中向下约 0.42 m/s；源接触标签在 955 帧已经切换，
但原参考右脚直到约 969 帧才达到地面。当前整脚面力可行性检查不包含
实际滚动点/线接触或冲击；通过该检查不能代替真实的接触门控与物理运行。
详见 `step13_nominal_stance_motion.json` 和 `step13_parent_stance_geometry.json`。

`step13_whole_body_dense.npz` 最终通过独立名义检查：601 个 0.125 帧
间隔样本的 HiGHS 最小物理松弛均为 0，基座动力学等式残差峰值
1.85e-13；按实际 float32 输入复核。801 个 0.25 帧间隔几何样本的
最小新增凸体距离为 20.184 mm。它仍仅是名义参考检查，通过后才启动
`thirteenth_feasible_reference` 从第 0 帧做 Isaac 物理试验；该试验是否
完成第 13 步由独立的实测落地/原始源保真度审计决定。

`thirteenth_feasible_reference` 的真实运行仍只验证了 12/29 次落地：
4.348 s / 源帧 1000.52 时原始源关节 RMSE 0.14701 rad、根位置误差
0.19588 m、根朝向误差 0.35383 rad，因朝向超限停止。腰 roll 在 4.32 s
已达到 0.4884 rad、6.63 rad/s，随后撞到 0.52 rad 限位；无自碰撞事件。
独立原始 PKL 审计也只确认 12 次，名义参考可行性不能算作第 13 步通过。

保存状态的第一优先级 QP 复算显示，4.3 s 时腰 roll 请求为约
-133.1 rad/s²，但 QP 实际预测为 +54.6 rad/s²。仅将 pelvis 姿态提高到
同级并不能在该晚期状态刹住腰部：全轴权重加大反而使预测腰 roll
加速度升高。新试验 `thirteenth_feasible_root_braking` 因此同时提前
启用腰部物理制动边界和 pelvis 姿态同级跟踪，从未失稳的更早阶段验证；
没有将离线复算的解直接发送给机器人。证据见 `step13_pelvis_priority_probe.json`。

该组合首次启动因重复诊断时刻参数格式错误在仿真前退出（`thirteenth_feasible_root_braking`，非物理失败）。已使用重复 `--diagnose-qp-time` 参数重新启动 `thirteenth_feasible_root_braking_v2`，原启动日志保留。

## 第13步：名义参考与实际支撑的差异（2026-11-07）

`step13_whole_body_dense.npz` 在 601 个密集采样点通过独立全脚掌名义动力学复核，
但 `thirteenth_feasible_reference` 实跑仍仅通过 12 步，在源帧 1000.52 因根姿态
误差 0.35383 rad 停止。这说明名义力矩可行并不保证实际滚动接触下的闭环跟踪。
在 4.3 s 腰 roll 已达 0.3618 rad、速度 5.8309 rad/s；姿态任务要求负加速度，
QP 折中后的预测加速度仍为 +54.63 rad/s²，随后碰到腰部限位。

- 同一参考提前加入根姿态优先级及腰部制动：
  `thirteenth_feasible_root_braking_v2` 仅通过 10 步，在帧 943.88 因关节 RMSE
  0.20089 rad 停止；此前不带 v2 的启动记录是参数语法错误，未运行物理仿真。
- 平滑合并全部任务到同一层：`thirteenth_feasible_all_tasks` 仅通过 9 步，
  在帧 805.64 因关节 RMSE 0.20084 rad 停止。
- 固定右脚支撑的 `step12_settled_support` 未通过可达性预检查，不执行。
- `step12_grounded_landing` 仅修正右脚落地高度过渡，保留水平路径与脚掌滚动；
  通过运动学和已配置碰撞几何检查，但名义动力学检查未通过。其物理试验是对照实验，
  不将参考预检查或源帧推进计为真实落地。实际 QP 验证和原始 PKL 验收均保持启用。

新的转弯段联合参考拟合和落地时序对照结果写入
`acc_data/g1_full_tracking_20261106/step13_progress.json`。第13步尚未通过，默认 profile
没有替换为这些未通过的候选。

落地参考后续对照：`thirteenth_grounded_landing` 在帧 1009.64 停止，但仅通过 11 步，
第12步最小承重落点误差为 0.11612 m，不能按源帧进度算作完成12步。
`thirteenth_landing_bridge` 恢复12步，在帧1003.88因原始根姿态误差0.35411 rad停止。
第13步单独提前左脚的参考超出腿部可达范围，未执行。
新的整体参考相位提前实验最多领先12帧，原始验收时钟和接触标签保持不变；
局部左肩roll增加0.03 rad的平滑余量后，密集参考碰撞检查最小间距20.566 mm。
它仍只是闭环对照，单支撑名义动力学预检查未通过，不能作为参考可执行性证书。

转弯段联合拟合查明旧端点约束互相冲突：696帧原参考手—髋间距仅6.609 mm，
固定端点无法立即满足20 mm要求。新离线拟合在720..1035帧要求20.2 mm，
连接段按五次曲线从原有间距过渡，且不低于原有较小间距。
这是离线参考连接段的处理，实际仿真碰撞约束和原始tracking验收没有放宽。

相位提前实跑 `thirteenth_phase_lead12` 仅通过12步，源帧999.56因根姿态误差
0.35439 rad停止，第13步没有承重落地。接触速度反馈对照
`thirteenth_contact_velocity8` 仅通过9步，帧845.96停止；没有把该参数提升为默认值。

对 `step13_whole_body_dense` 的名义接触运动学补查发现：即使离接触标签切换已6帧，
最低碰撞球材料点仍可能有1.176 m/s的速度。该参考此前的力/力矩可行性检查
不包含静止或纯滚动接触速度条件。实际记录也有较大的瞬时接触点滑速和速度跳变，
但跳变同时含碰撞冲量、模型误差和离散误差，不能全部归因于某个单一因素。
对应报告为 `step13_whole_body_dense.contact_kinematics.json` 和
`step13_contact_slip_and_impulses.json`。

转弯段后续搜索加入原始根位置16 cm参考上限及最多0.12 rad的根roll参考修正，
并继续保留原始关节参考RMSE 0.18 rad约束。实际tracking验收仍是
0.2 rad关节RMSE、0.2 m根位置及0.35 rad根姿态。离线候选只有通过非线性复核后
才会标记名义预检查通过；所有实际通过步数仍来自从第0帧开始的独立审计。


## 2026-11-23 Kinodynamic Fabrics 启发的顺序对照

**已回退到本轮实验前的 baseline。** 以下为历史实验记录，新增控制逻辑和开关已从当前代码撤回。
实验代码和对应测试保存在 `acc_data/g1_kinofabrics_trials_20261123/archived_source/`，
日志及结果仍保留；下列实验命令仅适用于归档实验代码。当前运行使用 `--tracking-profile amp-full`，
对应此前 1.356 s、3/29 次合格落脚的 baseline。

按要求不引入 ALIP 或落脚预测。新增实验开关均默认关闭，默认 amp-full 未替换：

- `--qp-certificate-polish`：DAQP 单次原生求解后，仅在数值证书失败且原始残差不超过 1e-3 时，精修原生对偶变量选定的活跃约束。
  精修必须满足原始完整约束和现有 KKT 门限；不是放宽容差，也不进行第二次原生 QP 尝试。
  它包含额外的线性 KKT 求解，不能描述为完全没有额外数值求解。显式不可行仍拒绝。
- `--fabric-stance-posture-scale .5`：在 Fabric 求和之前将支撑腿六个关节的姿态软权重减半；
  使用当前接触掩码，保持摆动腿、上身、任务目标及物理约束。原有 `--stance-posture-metric-scale`
  作用在另一条缓存任务路径，不能替代本项 amp-full 实验。
- `--contact-reference-bridge-duration .08`：在接触状态机确认离地时，将该脚的任务参考从测量位姿/速度
  经 C2 桥接回运行中的原参考，起始参考加速度取零。落地取消桥接。原源时钟、关节/身体参考和
  PKL 审计不变；桥接改变的是足端任务参考，因此额外保存 foot_task_* 及桥接活跃掩码。
  原系统的 Fabric 掩码和 QP 已使用同一拍接触状态，本项是参考衔接实验，并非修复已证实的一拍延迟。

复现命令（在仓库根目录，已配置 Isaac/GPU 环境）：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/test_g1.py --tracking-profile amp-full --qp-fabric-metric --qp-certificate-polish --output-path /tmp/g1_metric_polish.npz
```

独立实验命令、源码哈希、日志、NPZ、落脚审计和汇总保存在
`acc_data/g1_kinofabrics_trials_20261123/`；`summarize.py` 核对原始验收门限、单次原生求解、
无任务缓存/优先级阶段，以及实际施加力矩等于 QP 输出。
`offline_certificate_check.json` 证实原完整 metric 失败帧可精修通过，baseline 不可行帧仍拒绝，
且精修实验在原失败前与旧完整 metric 实验的 q/v/加速度/施加力矩逐位相同。
单元及集成验证记录见 `tests.log`（37 项通过）。

闭环结果（各自从第 0 帧独立运行，未完成的源落脚不跳过计数）：

| 实验 | 停止时间 s | 合格落脚 / 29 | 停止原因 |
|---|---:|---:|---|
| `baseline` | 1.356 | 3 | QP validation failed |
| `fabric_metric` | 0.334 | 1 | QP validation failed |
| `metric_polish_gpu` | 1.228 | 1 | Original-source fidelity limit exceeded |
| `metric_stance50` | 0.648 | 1 | Original-source fidelity limit exceeded |
| `metric_bridge80` | 0.906 | 1 | Original-source fidelity limit exceeded |
| `baseline_stance50` | 1.526 | 2 | QP validation failed |
| `baseline_bridge80` | 1.340 | 1 | QP validation failed |
| `baseline_bridge80_polish` | 1.752 | 1 | QP validation failed |

完整 metric 的数值阻塞被精修越过，但未提高落脚数。支撑腿降权到 0.5 在完整 metric 路径
更早超出关节误差门限；在 baseline 中虽运行更久，但合格落脚数由 3 降到 2。
80 ms 离地桥接在完整 metric 路径使第二步出现合格离地，但落点仍超限；baseline 上第二步
最低承重落点误差 4.834 cm，超过 4 cm，不能计为成功。baseline 桥接＋精修延长到 1.752 s，
仍只有 1 次合格落脚，最终失败帧经独立 LP 确认不可行。
该组合的根位置峰值误差为 10.607 cm、关节 RMSE 峰值 0.17547 rad、根姿态峰值 0.30579 rad。
精修前缀与未精修桥接实验逐位相同，见 `bridge_polish_failure_check.json`。
这些结果只覆盖所测权重和桥接时长，不证明所有相位分配或参考衔接方案都无效；
本轮没有得到优于 baseline 合格落脚数的方案，所有新增开关保持 opt-in。


### 2026-11-23 baseline 失败原因补查（控制器未修改）

证据保存在 `acc_data/g1_baseline_diagnosis_20261123/`。
需要细化此前“物理不可行”的表述：1.356 s 不可行的是含附加 tracking tube 的完整 QP；
移除三条 tube 行后，在相同冻结状态保留动力学、接触、摩擦、速度、防碰撞和力矩限制，LP 仍可行。
三条 tube 的启用阈值为源时钟 1.34 s；本次日志核对对应仿真时间 1.296 s，分别限制关节 RMSE、根位置和根姿态误差的二阶演化。
失败时 QP G 行 628/629/630 即这三条，570..627 为速度，40..569 为碰撞，0..39 为接触力约束。
单独去掉关节 tube 或根姿态 tube 也可恢复该帧可行性；单独去掉速度或摩擦行仍不可行。
一组删除过滤得到的冲突子集含碰撞行 454/512/560、关节 tube 628、根姿态 tube 630，
以及保留的接触/动力学等式和力矩边界。该子集不是唯一冲突解释，不表明实际已经发生自碰撞。

与最新 baseline 状态逐位相同的保存 QP 上，1.34 s 原输出根角加速度 pitch 为 +89.185 rad/s²，
只移除关节 tube 后为 -45.441 rad/s²；1.354 s 对应为 +762.182 与 -256.180 rad/s²。
这属于冻结状态反事实，不证明去掉 tube 后闭环能完成 PKL。
实际根 pitch 角速度从 1.34 s 的 1.213 上升到 1.356 s 的 8.092 rad/s；后者参考值仅 0.580。
完整管道在位置误差仍未超验收阈值时，也可能因为速度误差和二阶收敛要求而不可行。

更早的 COM 漂移仍由日志中的任务取舍损失支持：1.20 s 原 COM Y 目标 -4.555、
Fabric -4.582、投影 -4.023、QP +0.241、观测 +0.206 m/s²。
amp-full 的 `solve_single_tracking_qp` 清空叶任务和优先级，原命令行 balance_priority 等标志
不构成当前路径里的任务优先级保证。控制参考与原 PKL 也并非完全相同：1.20 s 控制/原始关节
RMSE 为 0.10995 rad，实际/控制为 0.11198 rad，实际/原始为 0.15441 rad；不能把这些 RMS 标量相加。
本次没有证明原 PKL 全段动力学不可执行，也没有更改默认控制器或任何验收阈值。

### 2026-11-23 软 tracking tube 与参考一致性对照（无 MPC）

实验记录：`acc_data/g1_soft_tube_trials_20261123/results.json`。全部独立从源起点运行，
保持 2000 帧、29 次落脚及原始 PKL 验收阈值（关节 RMSE 0.2 rad、根位置 0.2 m、
根姿态 0.35 rad）。没有加入 MPC，没有 QP 重试或任务优先级阶段。

新增两个默认关闭的实验选项：

- `--tracking-tube-soft-weight w`：`w > 0` 时，仅将三条辅助 tube 改为带非负松弛量
  的不等式，在同一个 QP 中增加 `0.5*w*sum(s**2)`；默认 `0` 保留原硬 tube。
  关节/位置行除以各自半径平方，姿态行除以 `sin(radius/2)**2`，故松弛量单位为 s^-2。
  动力学、接触、摩擦、碰撞、速度和力矩约束均没有松弛量。
- `--tracking-tube-reference control`：tube 与控制器共用同一个解析 q/v/a 采样器，
  原始 PKL 验收参考不变；默认仍为 `source`。

实现位于 `examples/g1_soft_tracking_tube.py`。完整增广 QP 通过原 DAQP 的独立
可行性/KKT 校验，再返回原物理变量给 WBC 验证。失败快照和显式 QP 快照中的
`soft_problem_*` 保存增广问题，`problem_*` 保存不含 tube 的物理问题。
历史 `tracking_tube_soft_slack` 记录每步的三维松弛量。

| 独立实验 | 停止时间（仿真 s） | 合格落脚 | 停止原因 |
|---|---:|---:|---|
| 默认 baseline 复跑 | 1.356 | 3/29 | 完整硬 QP 不可行 |
| 软 tube，w=1 | 1.364 | 3/29 | 根位置误差超限 |
| 软 tube，w=10 | 1.396 | 3/29 | 根位置误差超限 |
| 软 tube，w=100 | 1.404 | 3/29 | 关节 RMSE 超限 |
| 共用控制参考，硬 tube | 1.342 | 3/29 | 完整硬 QP 不可行 |
| 共用控制参考，软 tube w=10 | 1.402 | 3/29 | 根位置误差超限 |
| 上一组再将平滑跨度从 6 降至 2 | 1.344 | 3/29 | 根位置误差超限 |
| 软 tube w=10，从源时间 0.6 s 启用 | 1.214 | 3/29 | 关节 RMSE 超限 |

没有一组提高合格落脚数或完成完整 PKL，因此没有提升为默认配置。默认复跑与旧
baseline 的全部共同历史字段逐元素一致，唯一排除项为求解墙钟耗时；默认结果仍为
1.356 s / 3 次落脚。其余六个原控制器源文件哈希未变化，原入口代码另行归档。

关键证据：

- `frozen_soft.json`：原失败冻结帧中 w=0.1/1/10/100 均一次求解通过，物理约束
  最大残差低于 4e-10；这不构成闭环完成证明。
- `activation_check.json`：原 tube 首次启用发生在仿真 1.296 s、源时间 1.3402 s，
  当时根位置误差已为 0.16954 m；非提前/非平滑实验在这一状态之前与 baseline 完全一致。
- `reference_audit.json`：用真实源时钟重建并对照控制日志的 q/v/a；仿真 1.34 s 时
  控制/tube 关节位置、速度、加速度 RMSE 分别为 0.02192 rad、0.82666 rad/s、
  41.9087 rad/s²。接触几何检查仅是参考一致性诊断，不能当作实际地面滑移测量。
- `aligned_hard.failure.feasibility.json`：共用参考的硬 tube 失败帧仍是完整 QP 无解，
  去掉三条 tube 后物理问题可行。参考一致不等于辅助硬屏障必然可行。
- 提前启用实验在同一仿真时间 1.214 s，根位置误差由 baseline 的 0.13216 m
  小幅降至 0.12926 m，但关节 RMSE 从 0.15670 rad 增至 0.20019 rad，先触发验收失败。

本轮只实现 tube 软化、共用参考和已有平滑参数的对照，没有声称完成接触约束下的
全段重定向优化。结果支持继续检查更早的任务取舍和参考接触一致性，不能据此认定
原始 PKL 本身不可执行。32 项相关测试通过，见实验目录的 `tests.log`；
`summarize.py` 还检查每步一次原生求解、无任务缓存/优先级/恢复重试，以及实际下发力矩
逐元素等于 QP 输出。全部实验保留原始验收，不用松弛后的 tube 判定 tracking 成功。

### 2026-11-23：论文式 Fabric 任务优先级（可选，尚未改善完整 tracking）

按照 [Kinodynamic Fabrics 第 V-C / V-D 节](https://arxiv.org/html/2303.04279#S5.SS3)
的两级任务组织和式 (8)–(9)，新增 `--fabric-priority-mode paper`。
实现位于 `src/fabrics_sim/fabrics/priority.py`，优先级在 Fabric 内合成，最终仍然
只有一次 DAQP 逆动力学求解，不增加 MPC 或第二级 QP。

| 层级 | 当前 G1 的任务映射 |
| --- | --- |
| 一级 | COM 跟踪、质心角动量调节 |
| 二级 | 摆动脚、手、身体/源 pelvis 姿态、关节姿态、软避碰 |
| 独立硬约束 | 原有动力学、支撑接触、摩擦/力矩、碰撞 barrier、速度和 tracking tube |

论文把稳定性行为设为一级，其余行为设为二级；上述 COM/角动量映射是对已有 G1
任务的适配，没有新增论文实验中的 ZMP limit 行为。同级保留原任务 metric。
支撑脚仍由物理接触约束处理，表中的脚任务使用原有激活掩码。

将激活且权重非零的一级 Jacobian 合并为 `J_H`，构造
`N = I - D^-1 J_H^T (J_H D^-1 J_H^T)^dagger J_H`。
一级保留 `J`，二级采用 `J*N`；完整 `dot(J*N)*v` 用中心差分计算，包含
`dot(N)`，没有只对原始 `J` 求导。`D` 来自 URDF 全身惯量，并在仿真初始化后加入
Fabric 关节顺序下的实际电机 armature。零行/冗余行用伪逆处理，惯量检查考虑输入
float32 的舍入误差，投影及自然形式合成使用 float64。

本实现保留现有 35 维广义加速度根空间，因此选择矩阵 `S=I`；由逆动力学处理
浮动基座的欠驱动性。直接照搬“只保留关节列”会改变现有物理 COM 目标的含义。
原有物理反馈目标保持不变，式 (9) 使用既有 `1e-4 I` 正则化。
这属于优先级方法的适配，不是论文全部控制器的逐项复现。

`paper` 自动启用完整 Fabric metric 传递：单次 QP 使用合成后的 metric 和
Fabric 加速度目标，避免随后用固定加速度对角权重覆盖优先级结构。
硬约束和其他既有正则项仍可改变最终加速度，因此不能将 `J_H*N=0`
解释为最终接触受限 QP 对所有物理一级任务的严格字典序保证。
`TaskRequest.priority` 标签本身仍不触发额外求解。

可直接运行：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_paper_priority.sh
```

原入口 `run_amp_full_motion.sh` 和参数默认值继续使用 `none`，保留 baseline。
新入口显式设置 `paper`，也可通过末尾参数 `--fabric-priority-mode none` 切回。
尚未把此实验设为默认，因为闭环结果退化。

证据目录：`acc_data/g1_paper_priority_20261123/`，包含修改前/最终源码、源码 hash、
完整命令、日志、原始 PKL 验收和 `results.json`。

| 配置 | 停止时刻 | 已验证落脚 | 原始 PKL 结果 |
| --- | ---: | ---: | --- |
| baseline (`none`) | 1.356 s | 3/29 | QP 验证失败 |
| 初版：只保留关节列，已弃用 (`paper.npz`) | 0.140 s | 0/29 | joint RMSE 0.202691 rad |
| 最终广义坐标版本 (`paper_final.npz`) | 0.436 s | 1/29 | joint RMSE 0.200678 rad |

中间 `paper_generalized` 运行遇到过严的 float32 惯量对称性检查并退出，不能作为
完整验收记录；该检查已修复并补充回归测试。最终版本在 0.436 s 的 root 位置误差
为 0.041819 m，姿态误差为 0.037664 rad，但关节误差已超过原始 0.2 rad 阈值。
同一时刻 baseline 的 joint RMSE 为 0.102903 rad、root 位置误差为 0.023948 m。
最终版本偏差最大的关节为右髋 pitch（0.602652 rad）、右踝 pitch（0.411423 rad）。
因此这组优先级不能认为改善了动作保真度，也没有完成整段 PKL。

最终运行的 219 个控制更新均为一次原生 DAQP、KKT certified，无任务缓存、无
下游优先级求解、无 tube 恢复重试、无额外 PD 力矩；实际下发力矩与 QP 输出一致。
`max|J_H*N|=2.19e-14`，`max|N*N-N|=1.46e-14`，说明零空间投影实际生效。
新日志字段为 `fabric_priority_mode`、`fabric_priority_nullspace_error` 和
`fabric_priority_idempotence_error`。关闭优先级的 baseline 与前一轮 142 个共同
history 字段逐元素一致（排除求解耗时）。

66 项针对性测试通过，覆盖非单位惯量/冗余行的投影恒等式、冲突任务、
`dot(N)` 曲率、禁用任务释放、浮动基座列、float32 惯量舍入、真实 G1 模型及
单次 QP/原始验收回归。更宽的 `test_g1_*` 共 110 项中 109 项通过；原有
shoulder-roll 离线分析测试因当前环境缺少 `clarabel` 未能运行，未改变其实现或环境。

### 2026-11-23：单 QP 框架继续搜索，实测并复验 5/29 步

用户要求尽力测试可完成多少步。本轮完成 **22 次有完整验收报告的运行（含复验）**，
另有 5 次参数拒绝、日志错误、初始化停滞或原生崩溃，单列在 `incomplete_runs`，
不计入性能结果。证据位于 `acc_data/g1_priority_search_20261123/`。
所有有效运行都从原始片段开头出发，尝试完整 2000 帧/29 次落脚，未改原始 PKL。
关节 RMSE / root 位置 / root 姿态阈值仍为 0.2 rad / 0.2 m / 0.35 rad，
落脚仍要求 2 cm 离地、4 cm 位置误差以内且连续加载保持 50 ms。
两项减速实验在命令中标注；最终 5 步配置使用原来的速度时钟。

| 配置 | 已验证落脚 | 停止时刻 | 结果 |
| --- | ---: | ---: | --- |
| 上一轮原始论文式优先级 | 1/29 | 0.436 s | 关节误差超限 |
| 原 baseline，重新验证 | 3/29 | 1.356 s | QP 不可行 |
| 本轮主要任务合成，原硬 tube | **5/29** | **2.308 s** | 关节 tube 与物理约束冲突 |
| 延后软化关节 tube + 单次 proximal 求解 | 5/29 | 2.376 s | 关节误差超限，第 6 步未通过 |
| 更快载荷转移 | 5/29 | 2.436 s | 仍未通过第 6 步，关节误差超限 |

**5 步不是“原封不动的论文映射”的结果。** 保留的是
`Fabric 合成加速度/metric → 单个逆动力学 QP` 架构，成功配置重新组合了任务：
在 Fabric 内恢复已有的局部跟踪反馈，选取历史主要任务组（COM、身体姿态、
关节姿态，以及按当前阶段选择的摆动脚行），将其自然形式合成完整根目标。
没有把任务列表传给逆动力学，没有恢复旧的第二级 WBC。
历史次要软任务在此配置中的权重为零；既有物理硬约束保持不变。

与旧的固定加速度权重不同，根 metric 随合成目标传入最终 QP。
本配置使用 `1e-10 I` 根正则化，并去掉常规软力矩/接触力惩罚对主要任务的干扰；
QP 自身的数值正则化、硬力矩/摩擦/接触约束、阶段性卸载力目标继续保留。
这是一个单次优化的主要任务目标，不是对原两级 QP 的严格等价替代。

可复现入口：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_single_tracking.sh
```

默认输出 `acc_data/g1_priority_search_20261123/continued_tracking.npz`。
原 `run_amp_full_motion.sh` 仍是 3 步 baseline；
`run_amp_paper_priority.sh` 仍是上一轮论文式映射。新入口没有将停止条件设为 5 步，
而是继续尝试整段，直到原始误差检查或 QP 验证失败。

复验文件 `best_verify.failure.npz` / `best_verify.tracking.json` / `best_verify.steps.json`
与 `primary_pure_repeat` 的 **145 个共同 history 字段完全一致**（排除求解耗时），
共 1154 个通过认证的控制更新。第五步在 1.712 s、源帧 367.88 被验证；
失败发生在 2.308 s、源帧 510.92。已记录前缀的峰值误差：
joint RMSE 0.183898 rad、root 位置 0.066541 m、root 姿态 0.191715 rad，均未超限。
`baseline_verify` 与修改前 baseline 的 145 个共同 history 字段也完全一致。
详见 `verification.json`；最终两次复验还保存了逐文件源码快照与 SHA-256。

失败分析与进一步尝试：

- 冻结 2.308 s 的 QP，用独立 LP 检查得到原问题不可行；仅移除辅助关节 tube
  即可行，单独移除 root 位置或姿态 tube 仍不可行。
  证据：`primary_pure_repeat.failure.feasibility.json`。
- 新增可选的延后启用、仅关节行松弛。它仍在同一个 QP 内完成；root tube 和所有
  物理约束不获得松弛，原始 PKL 审计不受松弛影响。它可推进到约 2.34 s，随后
  原生 DAQP 数值失败。该失败冻结问题采用预先指定的 proximal factor 后，
  一次原生调用即可通过原有 KKT 标准，见 `soft100_numerics.json`。
- 可选 `--qp-single-proximal` 在指定时间后预先选择单次求解方式，不是失败后重试。
  可行性、stationarity、complementarity 的验收阈值未放宽，也未启用 active-set 修补。
  闭环因此延长到 2.376 s，但关节 RMSE 达 0.201169 rad，不能判为继续成功。
- 第 6 步是右脚落地：最佳加载位置误差 0.031822 m，但连续有效加载最长只有
  34 ms，不到要求的 50 ms。此时源轨迹已开始下一次左腿摆动，保留上一支撑与
  继续前进的关节参考发生冲突；最后左膝误差约 0.668 rad、左髋 pitch 约 0.476 rad。
  更早/更强的实际载荷转移，以及第 5 步后提高摆动脚位置权重，均未增加验证步数。
- 原论文式映射的权重、角动量开关、减速播放、物理目标补偿、接触一致零空间等
  对照也未超过 1 步。完整命令与所有结果见 `results.json` / `results.txt`。

新增实现和开关都保留为可选实验：`paper-tracking` 使用高优先级加速度补偿低级任务的
物理目标；`contact-tracking` 先在接触仿射空间中合成。两者均不增加 QP 或 MPC。
`--fabric-tracking-feedback` 把原有局部反馈恢复到根目标合成前；
`--fabric-secondary-scale` 控制历史次要任务组权重；
`--qp-task-only-objective` 控制额外常规软力目标；默认均保留原 baseline 行为。
主实现为 `examples/g1_priority_tracking.py`，新入口不启用实验性软 tube 或 proximal。

本轮 **73 项针对性测试通过**，见 `tests.log`。新增测试覆盖物理目标补偿、
接触仿射偏置、次要任务 metric 的 PSD、实际单次 DAQP 和硬力矩限制、仅关节行松弛，
以及单次 proximal 选择。`summarize.py` 对每个有效闭环验证：每次原生求解一次、
QP 调用处没有任务缓存或第二级优先级求解、没有额外 PD 力矩、实际下发力矩与 QP
输出逐元素一致。**本轮测得最好是 5 步，不是理论上限，尚未恢复历史两级版本的
12 步，也没有完成整段 29 步。**


### 2026-11-23：定位落点参考偏移，单 QP 复验 7/29 步

本轮从上一节的 5 步配置继续，不加 MPC，也不恢复第二层 WBC。
证据目录：`acc_data/g1_tracking_fix_20261123/`。这是对原始 2000 帧 PKL
整段运行的尝试；所有运行仍未通过 full-motion tracking。步数沿用原有连续
原始 PKL 审计，包含起始半步，不能把停止前的局部成功说成整段成功。

**第 6 步的最早偏差在控制参考中。** 预处理候选
`wbc_convex_catch15_25_clear35.npz` 的第 450 帧右脚 Y 相对原始 PKL 偏了
+0.030 m。实际落地受力后偏差一度达到约 0.048 m，超过 0.04 m 落地门槛；
单纯提高追踪权重无法消除这个目标偏差。逐步参考偏移见
`control_landing_bias.json`，实际/控制/原始参考分解见
`step6_error_decomposition.json`，对照图见 `step6_comparison.png`。

新增 `examples/prepare_g1_landing_reference.py`：通过离线腿部 IK 修正
完整关节位姿，再从同一全身参考派生各控制任务，不单独覆盖脚目标。
`landing15.npz` 在第 6 步右脚摆动中平滑加入 Y=-0.015 m 修正，支撑期间
保持，到下一次同脚摆动末端平滑退出（源帧 373/450/572/640）。骨盆、
另一条腿及目标脚朝向不变，源 PKL 与验收保持不变。IK 最大位置残差
1.88e-7 m，最大朝向残差 2.04e-7 rad；完整参考的源关节 RMSE 峰值仍为
0.14258 rad。候选生成器校验父候选与原始源哈希，拒绝不可达 IK 和覆盖旧文件。

| 配置 | 连续验收步数 | 停止时间 | 最后源帧 |
| --- | ---: | ---: | ---: |
| 上轮 5 步配置 | 5/29 | 2.308 s | 510.92 |
| 仅第 6 步落点修正，原速度 | 6/29 | 2.544 s | 567.08 |
| 落点修正 + 第 7 步落地前减速（当前入口） | 7/29 | 2.682 s | 559.98 |

当前入口在源时钟 2.55 到 2.80 s 区间平滑减速，等待第 7 步通过原有落地
验证后恢复；q/v/a 均使用同一时钟及其导数。第 6、7 步验收位置误差分别为
0.02865 m、0.02437 m。从控制时间 2.5 s 起启用 DAQP 的单次 proximal
求解，仍只有一次原生 QP 调用且须通过原有 KKT 校验，不是失败后再解一个 QP。
不启用软 tube。物理约束、0.2 rad 关节 RMSE、0.2 m 根位置、0.35 rad 根朝向、
0.04 m 落点、0.05 s 连续落地保持门槛均保持原值。

`align15` 与 `align15_verify`，以及 `align15_pause7` 与 `seven_verify`
分别复验成功：排除求解耗时后，152 项历史记录逐元素完全一致。
7 步复验的原始 PKL 误差峰值为：关节 RMSE 0.183898 rad、根位置
0.129573 m、根朝向 0.218952 rad。停止前 fidelity 通过，整段覆盖未通过。

停止原因仍需解决：冻结 7 步停止帧的 QP，完整约束不可行；仅移除辅助
关节 tracking-tube 行即可满足其余硬约束（`seven_feasibility.json`）。
最后已接受帧的关节 RMSE 仅约 0.09983 rad，但该二阶 tube 还约束相对速度和
加速度，不能由“当前位姿误差未越界”推断下一帧一定可行。没有放行被拒绝的
QP 输出。另有单次 proximal、不同落点偏移、提前恢复时钟、辅助 tube 软化
等对照；逐次结果及完整命令见 `results.json` 和各 `*.command.json`。

运行当前复验配置：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_single_tracking.sh
```

该入口尝试完整 PKL，遇到 QP 或 fidelity 失败会报错停止，并不保证走完整段。
上轮 5 步入口保留为 `examples/run_amp_single_tracking_5step.sh`；最初
`examples/run_amp_full_motion.sh` 的 baseline 行为未改。本轮 35 项针对性测试
通过（`tests.log`），包括新 IK 的几何保持和不可达拒绝。`summarize.py` 检查
所有完整实验的原始阈值、单次 QP、无额外 PD、无二次恢复 QP，以及实际施加
力矩与 QP 输出逐元素一致。


### 2026-11-23：继续排查第 8 步，保留已复验的 7 步入口

本轮证据目录为 `acc_data/g1_tracking_continue_20261123/`。仍使用完整原始
PKL、同一组误差及落地门槛、单个逆动力学 QP，不使用 MPC。以下是实验结果，
不是已经实现完整 tracking 的声明；汇总和完整命令见 `results.json` 与各
`*.command.json`。

新增可选开关 `--qp-cold-start-time`（控制/仿真时间，默认 inf）：从指定时间
开始，在每次唯一的原生 QP 调用之前清空热启动缓存。不做失败重试，不放宽
物理/KKT 校验，也不改变当前帧目标函数和约束。原速软 tube 7 步试验的冻结
QP 用冷启动能通过 KKT；用日志中的上一帧输出重建热启动则被拒绝。
重建输出可能比求解器内部缓存精度低，因此这是复现数值敏感性的对照，不能
视为完整缓存的逐位复现。见 `audit_warm_start.py`、`warm_cold_audit.json`。
闭环 `soft_cold` 越过了原先 2.662 s 的数值停止点，到 2.748 s 才触发原始
关节 RMSE 门槛；这只解决了一处数值障碍。

主要后续结果：

- 直接停用辅助关节 tube，或仅在第 7 步之后恢复参考时钟，未提高验收步数，
  反而暴露了后续关节误差越界；物理关节限制与独立 PKL 审计始终保留。
- 连续减速配合提前载荷转移，让第 8 步右脚真正离地。`slow_transfer` 的
  净空为 0.07055 m，但在落地前、源帧 613.08 处关节 RMSE 达到 0.20031 rad。
- `slow_landing25` 使用已验证几何可行的第 6 步 Y=-0.025 m 修正，配合
  控制时间 2.3 s 开始、0.25 s 渐变到 0.8 倍速度。推进至 3.104 s / 源帧
  647.64 时触发 fidelity：RMSE 0.20009 rad。第 8 步净空 0.08578 m，虽然
  已经触地，最小已加载落点误差仍有 0.16185 m，超过 0.04 m，不能计为第 8 步。
  停止时实际右脚相对原始目标约为 [-0.16356, +0.02978, -0.00583] m，主要
  是前向少迈。相应控制目标偏移约 [-0.01046, -0.02995, +0.00047] m，不能
  将这次 16 cm 偏差全归咎于参考落点偏移。见 `eighth_step_diagnosis.json`。
- 上身权重、质心权重、支撑脚姿态稳定、提前减速和后段脚位置权重等对照均
  保留完整原始审计；不能用多运行一点时间或脚更高替代完整落地验收。

QP 快照给出了进一步的任务冲突证据（`audit_com_authority.py`）：在
`slow_com8` 的 2.3 / 2.5 / 2.7 s 冻结状态，质心三轴目标单独可行，但再同时
  固定 17 个上身关节为 Fabric 请求的加速度时不可行。它们是单帧线性可行性
检验，不是动态轨迹可行性证明，也不表示增大某一个任务权重即可解决问题。
单轴可行范围和组合目标状态见 `com_authority.json`。

第 8 步的两种离线落点修正还在后续支撑段触发了固定骨盆腿部 IK 不可达
（源帧 758 / 761）；没有保存或执行这些候选，也没有放宽 IK 容差。
记录见 `rejected_reference_candidates.json`。

36 项针对性测试通过，包含冷启动后单次 DAQP、最优解保持、默认关闭、原有
物理限制与原始 PKL 审计。已验证入口 `examples/run_amp_single_tracking.sh`
没有切换到这些未提高完整落地步数的实验参数。

最终完成 **18 组新对照 + 1 组原配置回归复跑，共 19 组闭环**。原配置仍在
2.682 s 停止并验收 7/29 步，排除求解耗时后与上一轮的 152 项共同历史记录
逐元素完全一致（`baseline_verification.json`）。没有新试验完成第 8 步验收；
当前默认入口保持原有 7 步配置。全部试验通过单次原生 QP、实际力矩一致、
原始全身及落地阈值、冷启动时序检查，见 `results.json`、`acceptance_audit.json`。

### 2026-11-23：碰撞约束排查与第 8 步复验

最新证据目录：`acc_data/g1_tracking_contacts_20261123/`。本轮完成 13 组闭环，
将同一原始 PKL、同一验收规则下的连续步数从 **7/29 提升到 8/29**。
`examples/run_amp_single_tracking.sh` 已切换到独立复验通过的 8 步配置；
原 7 步入口完整保留为 `examples/run_amp_single_tracking_7step.sh`。
这里的步数沿用同源帧脚目标（`source_pose`）及既有初始部分步计数，不能
解读为整段 tracking 通过；另记的固定接触起始帧落点审计仅通过 2 步。

**本轮定位到的两个问题：**

- 前轮 `slow_com8` 在 2.3 / 2.5 s 的冻结状态中，COM 三轴目标和 17 个上身
  关节加速度目标无法同时满足约束。仅移除碰撞约束行，组合目标即可行；
  单独移除力矩、速度、接触等约束没有这一效果。碰撞约束独立重建与快照
  的系数及右端项误差均为零。保持同一球覆盖几何和 2 cm 安全距离，将
  加速度 barrier 带宽从 8 改为 20，也可使这两个冻结状态的组合目标可行。
  这改变了加速度约束的制动响应，并非所有约束系数保持不变；也不是整条
  动态轨迹可行性证明。旧轨迹 2.7 s 的组合冲突仍存在。见
  `constraint_conflict.json`、`barrier_bandwidth.json`。
- 闭环采用带宽 20 后，第 8 步原先约 16 cm 的前向不足消失，剩余横向偏差
  导致脚接触后只能维持 42 ms 合格落点，未满足 50 ms。全身参考的第 8 步
  右脚落点再沿 Y 回调 +1.5 cm，才完成原有落地验收。增大上身权重未提高
  步数且出现肩膀与躯干接触；新增落地速度限制反而退回 7 步，均未采用。

**参考和控制设置：** `prepare_landing8.py` 从前轮 `landing25.npz` 出发，
对源帧 573–801 做离线全身 IK，保存为 `prepared_landing8_y15.npz`，各任务
仍从同一全身位姿派生。允许根位置相对父参考最多移动 4.84 mm；IK 最大位置
残差 0.37 mm、朝向残差 0.000424 rad。独立腿部 FK 验证另一脚最大移动
0.119 mm，第 8 步支撑期目标脚 Y 修正为 14.89–15.09 mm；窗口外参考逐元素
不变。完整候选相对原始源的关节 RMSE 峰值仍为 0.14258 rad。原始 PKL 未改。
见 `prepared_landing8_y15.json`、`reference_geometry_verification.json`。

启动配置使用连续参考时钟，在控制时间 2.3 s 开始、0.25 s 内渐变至 0.8 倍
速度；沿用前轮的提前卸载、单次 proximal 和 2.5 s 起冷启动。只软化辅助关节
tracking tube，力矩、接触和碰撞约束不加 slack，原始 PKL fidelity 独立审计。
不再使用旧 7 步配置的等待落地时钟。碰撞带宽在源时钟 2.0 s 起用 0.1 s
渐变到 20。仍是一个逆动力学 QP，无 MPC、无额外 PD、无失败重试；当前
`fabric-priority-mode none` 是加权目标配置，不声称实现了严格分层任务优先级。

| 配置 | 验收步数 | 停止时间 | 停止原因 |
| --- | ---: | ---: | --- |
| 只调整碰撞带宽到 20 | 7/29 | 3.140 s | 原始关节 RMSE 越界 |
| 加第 8 步 Y +1.5 cm 全身参考修正 | 8/29 | 3.134 s | 原始关节 RMSE 越界 |
| 独立复验 `eight_verify`（当前入口） | 8/29 | 3.134 s | 同上 |
| Y +2 cm 修正 | 8/29 | 3.214 s | 同上；有肩膀/躯干接触记录 |
| 1.5 cm 版本再增大上身权重 | 8/29 | 3.284 s | 同上；有肩膀/躯干接触记录 |
| 1.5 cm 版本再限制落地速度 | 7/29 | 3.114 s | 原始关节 RMSE 越界 |

第 8 步在 3.100 s / 源帧 647 通过，落点误差 **3.158 cm**，当时关节 RMSE
0.16413 rad、根位置误差 0.15439 m。50 ms 连续保持、4 cm 落点、2 cm 离地
净空，以及 0.2 rad / 0.2 m / 0.35 rad 全身审计门槛均未修改。
落地验收时测得右脚法向载荷约 2633 N，后续仍存在明显的冲击与跟踪问题。
3.134 s / 源帧 652.44，关节 RMSE 达到 0.20046 rad，停止于第 9 步起步前。
最后有效状态中右肩 roll 误差约 -0.512 rad，肩部及左腿侧向关节贡献较大
（`post_landing_errors.json`）；仅凭这些误差不能断定冲击是唯一原因。

两次 1.5 cm 运行排除求解耗时后的 **153 项历史记录逐元素一致**，自碰撞启用，
均无自接触事件；1.85 s 后额外硬碰撞覆盖的最小距离为 20.175 mm。
`verify_eight.py` 校验了参考/源哈希、复跑一致性和原始验收门槛；
`summarize.py` 对全部 13 组检查每次原生求解次数为 1、实际施加力矩与 QP
输出完全一致、无二次恢复求解。36 项针对性测试通过（`tests.log`）。
启动脚本通过 shell 语法检查，其 32 项显式运行选项与 `eight_verify` 一致，
见 `preset_verification.json`。上述结果不代表完成完整 2000 帧 / 29 步。

复现命令仍为：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_single_tracking.sh
```

原始两组 `landing8_y15/y20.command.json` 的输入候选名与后来仿真输出名重合；
候选已在写出仿真结果前保留为 `prepared_landing8_y15/y20.npz`，输入哈希及
映射记录在 `reference_inputs.json`，保存的 control reference 也逐元素一致。
复现请使用独立的 `eight_verify.command.json` 或上述入口；runner 现在拒绝
覆盖已有输出文件。保留原始命令记录，不改写历史。

### 2026-11-23：持续完整 tracking 任务，已独立复验 9/29 步

本节为进行中的工作记录，目标仍是完整 2000 帧 / 29 步，尚未完成。
证据目录为 `acc_data/g1_tracking_complete_20261123/`。
`pause9_early` 与 `nine_verify` 在排除求解耗时后，154 项历史记录逐元素一致，
自碰撞开启且无自接触事件。第 9 步在 3.670 s / 源帧 773.495 通过，落点误差
1.846 cm，承重 487.0 N；当时关节 RMSE 0.11277 rad、根位置误差 0.16123 m。
3.714 s / 源帧 775.527 时关节 RMSE 达到 0.20262 rad，停止于第 10 步起步前。
原始 0.2 rad / 0.2 m / 0.35 rad、2 cm 离地、4 cm 落点和 50 ms 保持门槛不变。
入口已更新到此复验配置；8 步入口保留为 `run_amp_single_tracking_8step.sh`。

本轮实现了可选的 DAQP 变量缩放：`--qp-equilibration-start 2.5`（控制时间）、
`--qp-equilibration-cap 100`。令 x=D*y，按 Hessian 对角线选择有界正对角 D，
等价变换目标、约束、边界与热启动，求解后把原始变量和边界对偶映射回来。
每周期仍只有一次原生 DAQP，使用原有 native 容差；在原始坐标中检查全部
物理约束、stationarity 和 complementarity，不修改 KKT 验收阈值、不失败重试。
默认 cap=1、start=inf 保留原行为。初版实验曾把 native 容差设得过紧而出现
错误拒绝，现已恢复原 native 容差；初版 `*_scaled` 证据未覆盖。
41 项求解器/单 QP/软 tube/完整审计相关测试通过，见
`equilibration_tests_final.log`。冻结失败问题的等价缩放对照见
`equilibration_audit.json` 和 `scaling_tolerance.json`。

另新增了仅在指定 QP 快照时保存的 `composed_task_*` 数值数组，记录任务被清除
前的实际权重和目标。任务对象仍不传入逆动力学 QP。快照版本与原 8 步复验的
153 项共同历史逐元素一致（`snapshot_noninterference.json`）。快照消融显示，
接近第 8 步落地时辅助根位置 tube 会把肩部加速度推向跟踪目标的反方向；
质心目标单独满足全部物理约束的瞬时可行解存在，但加权目标之间存在竞争。
这些是局部数值诊断，不能替代闭环结果。

9 步参考从前轮 8 步候选继续修正：第 7 步左脚 Y=-3 cm 的全身 IK 修正，
再在源时间 2.5–2.8 s 平滑加入骨盆 Y=+5 cm、Z=-5 mm 偏移，保持到 3.2 s，
于 3.7 s 前退出；腿部 IK 保持两个世界脚位姿，验证位置残差约 1.02e-6 m。
所有任务仍从同一个完整 control_q 派生，源 PKL 未动，完整参考对原始源的
关节 RMSE 峰值仍为 0.14258 rad。使用的文件为
`prepared_combined_root_y50.npz`，生成器与逐项几何检查均保留在证据目录。

源时钟保留前段减速，并从控制时间 3.0 s 起、0.2 s 内恢复到 1.2 倍速度。
无等待的 `combined50_clock` 第 9 步已到正确落点，但载荷中断使最长合格保持
只有 44 ms；源时间 3.7–3.88 s 的平滑落地等待才完成 50 ms 验收。
`--contact-pause-landings 9` 是恢复参考时钟的条件，不是运行终止条件。
另一组 `landing9_load` 在同一个 QP 中加入已有的承重下界，也通过 9 步且让
第 10 步取得 3.93 cm 离地证据，但在 3.776 s 因原始关节误差停止，未完成第 10 步。
后续试验继续保存在此目录，不将更晚停止或更大离地高度计为成功落地。

新增的可选 `--root-position-task-reference control` 让已有软骨盆位置任务
使用统一 control q/v/a；默认仍为 source。首次开启该任务的实验退回 7 步，
未纳入上述入口。将 5 cm 骨盆偏移延长至后续步态的固定根朝向 IK 候选未通过
脚姿态保持预检，未输出可运行候选；其失败报告也保留，未放宽几何门槛。


后续独立复验 `landing9_transfer_snapshot` 与 `landing9_load` 的 154 项共同
历史逐元素一致（排除耗时），证据为 `load9_snapshot_verification.json`。
仍为 9/29 步，不能把第 10 步离地计为落地成功。`nine_preset_check.json`
也确认此前 9 步等待入口的 394 项解析参数与对应复验配置等价。

可选的 `--unload-deadline-start` 根据下一次同脚摆动的源时间与当前源时钟速度，
在实测承重转移开始时确定更短的卸载过渡；暂停时钟保留普通过渡。
`--unload-deadline-weight` 可单独提高该时段的软卸载力目标权重。
两项默认不启用，不直接解除接触、不改落地验收、不增加 QP。
单独缩短过渡仍只有 9 步，较强力目标的闭环效果仍在验证。
44 项相关测试通过，见 `unload_and_solver_tests.log`。

第 10 步原始摆动窗口为源帧 780–802。`landing9_load` 中右脚实际到约
3.73 s / 源帧 798 才卸载，摆动明显滞后。冻结的 3.64 s QP 中，较强的
卸载目标可在全部原有物理约束下把右脚载荷由约 123 N 降到约 4.4 N；
这只是同一状态的局部诊断，见 `transfer_unload_weight_audit.json`，并非
闭环成功证明。关闭辅助根位置 tube 的诊断实验则在根位置误差超过
0.2 m 时停止，没有放宽独立的原始源验收。

延长骨盆修正的有界根朝向 IK 已通过严格双脚位姿预检，随后闭环仍只有
9 步，未采用。较激进的提前落地 IK 候选未通过预检而未写入可运行 NPZ；
较小的摆动相位提前候选通过预检，但已完成的闭环仍未增加步数。
无界面模式现在跳过纯视觉的远程天空 HDRI；有界面场景保留原天空设置。
两个长时间停在初始化的进程被终止并以新名字重跑，原运行记为未完成，
不纳入跟踪步数结果。


当前 `run_amp_single_tracking.sh` 已切换到独立复验的 `landing9_load` 配置，
396 项解析参数核对一致（`load9_preset_check.json`，忽略输出路径与诊断快照时间）。
原等待版本保留为 `run_amp_single_tracking_9step_pause.sh`。两者都只有 9 步；
当前入口额外取得第 10 步离地证据，不声称完整跟踪。
无 HDRI 的候选在控制变更生效前，1744 个控制样本的 154 项共同历史与原轨迹
完全一致，见 `headless_prefix_check.json`。

较强卸载在闭环中把右脚实际释放从约 3.74 s 提前到 3.64–3.65 s，
但仍出现摆动末段横向滞后和提前触地。更早提高质心软权重的
`combined_com4_early` 通过 9 步，达到源帧 815.96，停止时根位置误差
0.15727 m、根朝向误差 0.17970 rad，关节 RMSE 0.20248 rad；未独立复验，
未替换入口。与足端权重 4 组合的 `early_com4_foot4` 第 10 步最小承重
落点误差为 0.12179 m，仍大于原 0.04 m 门槛。躯干权重 0.1 的候选
达到源帧 824.12，但根朝向误差 0.35263 rad 超过原 0.35 rad 门槛，仍失败。
接触子步、角动量阻尼、额外摆动净空等已完成候选均未超过 9 步；
保留全部结果和失败条件，后续继续基于这些约束诊断迭代。


`early_com4_foot16_body03` 将第 10 步最小承重落点误差降到 9.18 cm，
但主要因右肩侧摆、腰和支撑腿误差而失败。其上肢权重 16 版本
`foot16_upper16` 达到 4.235 cm，仍超过 4 cm，且在源帧 803 / 3.750 s
关节 RMSE 0.20309 rad 超限；尚无第 10 步连续合格承重证明。

实际电机预算诊断见 `model_effort_frozen_audit.json`。原控制预算仍使用
0.8×旧软件上限（踝/腰部分关节为 20 Nm），而已核对的活动 URDF/Isaac
上限中对应关节为 50 Nm。后续候选通过现有 `--model-effort-start`
提前采用 0.8×活动模型上限，或通过现有同步 direct-WBC 预算选项释放
剩余比例；不修改 URDF/Isaac 上限，力矩始终不超过活动模型上限。
这些预算候选尚未替换默认入口，也不等同于原软件限幅下的结果。


### 后续推进：10/29 步已独立复验，完整 tracking 仍未完成

`foot16_upper16_model80` 与 `ten_verify` 排除求解耗时后的 155 项历史
逐元素一致，均无自接触事件。第 10 步于 3.798 s / 源帧 814.52 完成
原有连续 50 ms 承重验收，落点误差 2.4719 cm、实测载荷 548.1 N；
当时原始源关节 RMSE 0.16185 rad、根位置误差 0.10866 m、根朝向误差
0.23447 rad。第 11 步已取得 4.84 cm 离地证据，但在 3.848 s / 源帧
826.52 根朝向误差达到 0.35136 rad，超过原 0.35 rad 门槛而停止。
详见 `ten_verification.json`；`full_tracking_passed` 仍为 false。

当前入口已切换到此十步复验配置，396 项解析参数核对一致
（`ten_preset_check.json`）。九步承重入口另存
`run_amp_single_tracking_9step_load.sh`，九步时钟等待入口也保留。
十步配置在九步承重版基础上采用：COM 软权重 4（源时间 2.5 s 起）、
卸载时间匹配及权重 2（3.7 s 起）、足端权重 16（3.85 s 起）、
躯干权重 0.3（3.2 s 起）、上肢权重 16（3.7 s 起），并在 3.2–3.3 s
过渡到活动 URDF/Isaac 电机上限的 80%；等价 QP 缩放 cap 为 10。
单个 QP、无 MPC、原始 PKL、原始 fidelity/落地验收均保留。
全部 29 步尚未完成，后续继续处理第 11 步之前的根部滚转增长。

进一步诊断发现十步版本在约 3.798 s 撞到腰部滚转 0.52 rad 的机械
限位。QP 仍预测向限位外的加速度，而实际关节速度被机械限位阻止。
`ten_verify.joint_stop_audit.json` 记录了这一现象：该关节在 25 个更新中
静止于上限、QP 却请求超过 10 rad/s² 的向外加速度。现有
`--joint-limit-barrier-bandwidth 8` 仅设置参数，并不启用可选物理制动
边界；十步配置也未把软避限任务纳入有效任务权重。仿真机械限位
始终存在，因此不能把未启用预测制动理解为取消了机械限位。

`audit_stop_response.py` 对 `ten_momentum_late` 的 3.800 s 快照做了
固定电机力矩、相同足端约束下的离线质量矩阵计算：加入测得的腰部
限位反力后，根角加速度预测从 [-12.72, 59.18, -28.12] 变为
[11.24, 19.14, -44.29] rad/s²，接近实测 [11.68, 19.25, -43.68]。
结果在 `waist_stop_mass_response.json`，这是故障诊断，不是运行时
新增 QP，也没有改变物理模型或源验收。已添加制动启用状态和上下界
日志，以及 `--physical-braking-joints` 按名称选择关节的可选接口；
后续候选继续验证提前避限是否能在保留十步的同时通过第十一步。


后续机械限位反力模型作为默认关闭的可选功能加入同一个原生 QP。
`ten_no_stop_verify` 与十步版本的 155 项共同历史完全一致，见
`stop_model_disabled_verification.json`；启用反力模型后仍为 10 步，
因此尚未推广到当前入口。反力是机械限位反力预测，不是额外电机力矩；
实际施加的电机力矩仍与单个 QP 输出逐项一致。

参考动力学诊断发现：原控制参考在源帧 792 的质心竖直加速度为
-12.464 m/s²，需要地面法向合力 -88.49 N，不能由单边地面接触实现。
离线向下调整质心可消除此问题，但高度修正闭环只到 9 步。
`prepare_force_com.py` 联合拟合三轴质心加速度和摩擦合力，再通过 IK
保持双脚位姿；此过程仅生成离线控制参考，不是运行时 MPC/WBC。
`prepared_force_com_xy30_mu55` 的最大三轴质心修正分别为
4.34、5.79、11.52 mm，全段 IK 双脚位置残差小于 0.8 μm。
运行时插值后，在源帧 740–884 的负法向力和摩擦合力越界均为零。
但这只是必要合力条件：闭环 `ten_force55` 仍只有 9 步，源帧 806.84
关节 RMSE 0.20559 rad 超限；当时根朝向误差 0.06550 rad。第 10 步
取得 4.64 cm 离地、3.44 cm 承重落点证据，但合格连续承重仅 4 ms，
未满足原 50 ms 要求，不能记作第 10 步。

新增的离线全身角动量模型已与现有 Torch/URDF 模型核对，七个姿态的
角动量误差小于 1e-5。包含角动量变化后的 CoP 诊断显示，合力修正
仍未消除支撑力矩冲突，详见候选的 `.momentum_audit.json`。
CoP 审计假定地面 z=0、使用乐观的完整名义脚底多边形；它是参考
可实现性诊断，不代替实际闭环落地验收。当前最佳仍为独立复验的
10/29；原始 PKL、原始 fidelity/落地门槛和默认入口均未替换。


进一步对照仍未超过 10 步。`ten_pelvis_roll2` 仅提高骨盆世界 x 轴角运动
任务，推进到源帧 838.52 / 3.898 s，但原始根朝向误差 0.35473 rad
超限，仍是 10/29，未替换默认入口。其末段根位置误差 0.18398 m，
关节 RMSE 0.18759 rad。新增 `--late-pelvis-axes` 可选择该可选任务的
世界角坐标；未指定时保持原行为。`--late-foot-metric-end` 等可选项
允许结束临时增大的摆脚任务权重，默认禁用；相应对照也未突破 10 步。

第四段平滑速度过渡 `--clock-fourth-*` 默认关闭，三个解析导数/连续性/
输入检查测试通过。0.8 和 1.0 末段速度对照分别为 9 和 10 步；
两者在 3.5 s 生效前的 1751 个样本、155 项共同历史与十步轨迹逐元素
相同，见 `late_clock_prefix_audit.json`。机械限位五项测试再次通过。

已分别尝试更强 IK 旋转正则、骨盆四元数平滑、第 10 步摆脚平滑、
延续横向根部修正、后段使用额定电机剩余预算、提高上肢姿态权重、
提前向外修正第 10 步落点，以及触地后足底姿态任务。全部闭环结果
保存在 `acc_data/g1_tracking_complete_20261123` 的独立结果文件中；
未通过全段 IK 预检的候选不会生成可运行参考。当前原始 PKL SHA-256
再次核对为 919eaff9b2521cce1e8b40e263fceb3f3f1873a7bc42500d9932be62f007eca9。

离线 `audit_frozen_weights.py` 在相同保存状态下重解任务权重对照，
仅用于局部诊断，不参与运行时控制。包含原始角动量变化的线性化
CoP 拟合，在水平质心改动限制 5 cm 时仍至少需要约 38 cm 最大
CoP 约束违背；这一结论只针对该冻结模型。将参考角动量变化做
20 帧平滑后的模型存在可行解，但这并不证明实际参考可实现：必须
重新计算 IK/插值后的完整角动量并进行闭环验证。零角动量筛选也仅是
近似诊断。所有离线优化都没有增加运行时 MPC 或第二个 WBC。


2026-11-24 继续实验：`ten_pelvis_xy2` / `ten_pelvis_xz2` 分别推进至
源帧 833.24 / 836.60，均为 10 步后根朝向超限。将内部卸载目标距离
由 3 cm 改为 4 cm 没有改变十步基线结果；原始落地门槛保持 4 cm。
检查旧“提前摆脚”参考发现其相位偏移在落地前归零，没有真正提前到达
落地点。新 `prepare_early_landing9_full.py` 生成提前到达、短暂停留、
接回原参考的候选，并检查双脚位姿及原始关节偏差。固定骨盆的旧 IK
不可达；允许全身调整并短时降低 COM 1.5 cm 后通过预检，但提前 4/8 帧
及联合右脚平滑的闭环均为 9 步。8 帧候选实际左脚承重只提前约 2 ms，
见 `true_early9_timing_audit.json`。脚部任务 16 倍增权提前至源时间
3.55 s：原参考为 9 步，提前落地参考在源帧 743.96 出现 QP 验证失败。
持续降速对照 `ten_global_clock7` / `ten_global_clock9` 分别在 3/4 步
后出现 QP 验证失败。以上均未替换独立验证的十步入口。

新增离线 `CentroidalGeometry.centroidal_matrices` 和 `inverse_dynamics`：
前者与独立逐链接角动量计算一致；后者在三个仿真快照上与 PhysX
质量矩阵及偏置力核对，上半身力矩最大误差小于 0.00005 Nm，见
`numpy_inverse_dynamics_validation.json`。合力及全身角动量变化也与
原独立计算核对，最大误差约 0.000055，见 `rnea_centroidal_validation.json`。
这些方法只用于离线参考准备，不参与运行时控制。

仅约束支撑合力/力矩的上半身调整 `prepared_upper_momentum_analytic_b20`
未通过闭环（9 步），其左肩在源帧 798 要求约 317.2 Nm，而当前预算
为 20 Nm。位置预检查通过不能代表动力学可执行。新
`prepare_upper_dynamics.py` 同时拟合接触、上肢力矩和速度约束；所有
物理约束使用显式报告的松弛量，线性拟合不是动力学可行性证明，必须
重新计算完整非线性参考并做闭环验证。当前仍未完整 tracking。


上半身力矩约束拟合的大幅候选 b20 在非线性复核仍达到预算的 2.64 倍，
未送入闭环；小幅 b05 的峰值约 1.15 倍，检查区间内消除了负法向力，
但 CoP 冲突未全部消除。b05 原权重/骨盆横滚增权/完整额定力矩预算的
闭环分别为 9/10/9 步，未改善独立验证的十步成绩。固定骨盆朝向、
COM 降低 2.5 cm 的提前落地参考分别为 8 步，配合较早 4 倍脚部任务为
9 步，仍未突破。

`baseline_single_support_effort_audit.json` 使用源单支撑接触和给定参考
运动反算关节力矩：源帧 800 左踝横滚约 113.89 Nm，对比当前 40 Nm
预算；源帧 882 右踝俯仰约 166.15 Nm。它说明精确执行该准备参考的
动力学冲突，不证明原始误差容许范围内不存在可行 tracking。
新增 `prepare_root_upper_dynamics.py` 尝试骨盆平移和上半身联合修正，
双腿通过脚位姿雅可比联动并在输出时做有界 IK。其双支撑力矩估计
明确采用等分支撑力假设；所有候选仍须非线性复核和真实闭环验收。
最初两个离线拟合达到迭代上限，未生成可运行参考。


`ten_waist_metric4` 单独提高腰部姿态权重，将腰横滚首次停在上限的源帧
从 814.52 推迟到 822.68，但仍在源帧 831.32 因根朝向超限停止（10 步）。
实测支撑点修正 1.5 cm / 3 cm 的闭环分别为 10 / 8 步。

`ten_torque_continuity01` / `ten_torque_continuity10` 的最初两次试验都在
权重启用之前的源帧 624.92 分叉并停止，不能用于评价力矩连续性策略。
原因是 `TorqueContinuitySolver` 只代理属性读取，没有代理写入，导致
运行时的 `tube`、`single_proximal` 和 `equilibration_cap` 写到了包装层，
未传入实际求解器。现已修复写入传递；连续性自己的 weight/previous
与底层软管道权重保持独立。新增配置透传和零权重原生软约束求解一致性
两项回归检查，连同原有四项共六项通过。完整零权重十步复跑进行中；
默认关闭连续性选项的十步入口未改变。

零权重闭环复验已完成：`ten_continuity_zero_verify` 与 `ten_verify` 的
155 项共同历史字段（排除执行耗时）逐元素完全一致，10/29，停止源帧
826.52。证据为 `torque_continuity_zero_verification.json`。因此上述配置
写入修复保留零权重行为；随后才重新测试非零连续性权重。


修复后的全关节力矩连续性权重 0.01 / 0.1 均为 10 步；前者在源帧
819.80 因关节 RMSE 超限，主要误差集中于左髋偏航/横滚；后者在源帧
827.48 根朝向超限。新增默认关闭的 `--torque-continuity-stance-only`
只对当前 QP 支撑腿的六个电机施加相同连续性目标，摆动腿和上肢不受
该目标影响。包含支撑切换和掩码检查在内的八项测试通过，全段 tracking
相关十五项测试通过；两个 stance-only 对照仍为 10 步，未改善成绩。
原有球形接触运动学修正对照也是 10 步。

离线 root/leg/upper 边界约束候选 `prepared_root_upper_dynamics_xy10_up05_boundary`
通过位姿预检并改善部分参考动力学需求，闭环仍为 9 步。增大根部范围的
候选以及加入膝屈曲余量的候选未通过严格脚位姿检查，不生成可运行 NPZ。
`refine_root_upper_ik.py` 保存失败 IK 的独立诊断数组；例如局部附加降低
骨盆会使左踝俯仰下限约束更紧，不能简单通过继续降低重心解决。

`slow_clock_frozen_feasibility.json` 对早期降速失败状态独立检查：0.7 倍
速度的保存 QP 在旧踝 20 Nm 预算下不可行，在实际 URDF 的 80% 预算
（踝 40 Nm）下 LP 和 DAQP 均可行；0.9 倍的保存状态仍不可行。闭环
从起步改力矩预算会改变前段轨迹，不能据此推断整段降速成功。保留原
起步、源时间 1.3 s 后再改预算的两组 0.7 倍实验仍只到 3 步。

`prepare_stop_margin.py` 准备小幅 C2 腰部/左踝姿态偏置，针对源帧约
805/815 的实测机械限位。腰偏置不改变双脚位姿；踝偏置最多改变脚
位置约 1.76 mm、脚朝向约 0.128 rad。这里明确允许控制参考脚朝向变化，
并未改变原始关节/根部/落地验收门槛，原始 PKL 不变。三组预检通过并
完成参考上肢力矩审计，正在进行闭环验证。

限位余量闭环已完成：腰/踝/组合为 9/10/9 步，没有突破。
`prepare_early_landing10.py` 在源帧 798/796 达到原第十步 802 帧的右脚
位姿，再于 822 帧接回原参考，区别于只在摆动中段加速的相位钟形偏置。
固定根部和其余关节，有界六维 IK 残差小于 0.4 微米，原始关节 RMSE
预检不超过 0.142581 rad；最大中段脚目标变化约 0.150/0.201 m。
这只是位姿预检通过，落地仍由原始 PKL 同源帧验收，不能算提前成功。

真正提前第十步落地的 f4/f6 候选都只通过 9 步；四次物理子步在当前
十步框架上也为 9 步。COM gain 2 + leg metric 4 + upper metric 32、
gain 1.25 + leg metric 2、腰偏置 + waist metric 4 三组均为 9 步。
默认入口仍保留独立验证的 10/29，未完整 tracking。

新增离线 `prepare_freeleg_dynamics.py` 使用缓存的精确 RNEA 导数，
在固定根朝向下允许根平移、双腿和上半身独立修正；一阶脚位置各轴
变化限制 1/1/0.3 cm，朝向各轴 0.1 rad。输出不再做恢复原脚位姿的 IK，
而是重新计算完整 FK，要求实际脚位置变化 <2.5 cm、朝向 <0.20 rad、
原始关节 RMSE <0.19 rad 和 URDF 关节限位。此处是控制参考自由度，
原始 PKL 和实际验收未改变；动力学不等式仍显式报告松弛，不能据此
声称参考可执行。后续需要非线性动力学审计和真实闭环。

`ten_soft_ori25` 在 3.2 源秒起使用 0.25 rad 辅助朝向半径且三类管道均
允许松弛，达到 10 步、源帧 837.56；朝向误差 0.2744，但根位置
0.200031 m 超过原始验收。soft 0.30 为 9 步 QP 失败，hard 0.30 为
10 步朝向超限。新增默认关闭的 `--tracking-tube-soft-joint-orientation`
允许关节/朝向两类辅助管道松弛，保留根位置管道为硬约束。三类原始
源数据验收均不变，同一个 QP 中物理约束没有松弛系数；九项管道测试
和十五项 full tracking 测试通过。

`prepare_root_roll.py` 的根横滚 -0.06/-0.12 rad + 根下降 2 mm 都未通过
固定双脚 IK；下降 15 mm 的 -0.06 rad 候选通过（最大脚位姿误差
小于 0.6 微米/微弧度），-0.12 rad 仍失败。通过候选的精确参考力矩
并非全部可执行，不视为动力学证书。

`prepared_freeleg_dynamics_r10_l10_u05` 求解收敛但脚朝向最大变化
0.2115 rad，未生成可运行参考。`scale_freeleg_candidate.py` 将其修正
缩小至 0.8 后，位置/朝向变化为 0.01982 m/0.17004 rad，通过原有预检。
非线性逆动力学在源帧 780–802 的峰值约为当前预算 1.187 倍，
802–884 区间仅三帧超预算；其余区间仍存在明显超预算需求，双支撑
力矩还采用等分载荷假设，不能称为物理可行。正在闭环验证。

保留硬根位置的 `ten_soft_ori25_hard_position` 为 10 步、源帧 830.36；
`ten_root_roll_minus06_z15` 为 10 步、829.40；均朝向超限。横滚任务
权重 8 为 10 步、834.68；soft ori25 + root task1000 为 9 步、关节超限。
`soft_ori30_frozen_numerics.json` 独立复核失败时保存的同一个增广 QP：
cap10/prox1e-4 重现最优性检查拒绝，其他多组缩放/近端参数通过同一
KKT 检查；这是数值原因，未修改验收或给运行时加入重试。

`ten_freeleg_scale80` 为 9 步，源帧 822.68 朝向超限；第十步右脚
最佳合格位置误差 0.03226 m，但仅连续维持 6 ms。
`late_candidates_stop_timing.json` 显示它在停止前未碰到左踝两轴限位，
而基线在 804.92/814.52 帧碰限位；腰仍在 816.44 帧触限。
新增 `prepare_freeleg_waist.py` 将已验证的腰偏置和自由腿参考组合，
仅改变腰横滚，脚位姿保持自由腿参考值；原始关节预检通过。

自由腿候选 + foot metric32 + 额定力矩100% 仍为 9 步。自由腿 + waist
metric4 为 10 步；再叠加腰横滚 -0.1 rad 余量为 10 步、源帧 835.16。
后者左踝横滚首次触限推迟到 813.56，左踝俯仰未触限，腰横滚触限
推迟到 828.44，最终仍朝向超限。相同组合正在检查 -0.2/-0.3 rad 腰
余量，准备阶段原始全身关节 RMSE 上限未变。

大范围自由腿拟合 r30/l20/u05 达到 200000 次迭代并返回 solved inaccurate，
原始输出脚朝向变化 0.2855 rad，预检失败。修正缩小至 0.65 后位置/
朝向变化为 0.01718 m/0.18393 rad，才生成可运行候选；精确参考力矩
仍有超预算区间，求解器状态不代表运动可执行。

`prepare_freeleg_dynamics.py` 新增默认零的离线脚修正曲率代价
`--foot-acceleration-penalty`，按真实重定时后的采样时间二阶差分脚
位姿线性修正（角度乘 0.1 的量纲权重），抑制小幅但快速变化的脚
目标。原先自由腿 scale80 右脚 Y 修正在帧 802 为 +8.7 mm，810 为
-17.3 mm；位置偏差小不能保证加速度平稳。该离线正则项不改变
原始 PKL，不是运行时 MPC，也未解除物理或 tracking 验收要求。

自由腿 waist -0.2 + metric4 为 9 步；-0.3 为 10 步、源帧 847.16，
腰触限推迟至 840.92，仍未验证第十一步，不能作为步数新纪录。
大范围自由腿 scale65 为 10 步、831.32。

`freeleg_native_clearance_audit.json` 用实际启用的 USD 球/立方体脚底
几何检查参考：自由腿 scale80 右脚底在帧 802 为 +4.37 mm（基线
+0.41 mm），810 为 -7.25 mm（基线 -0.21 mm）。位置预检并不
保证原生脚底接触净空。新增 `prepare_freeleg_clearance.py` 保持候选
COM、脚 XY 和朝向，通过全身有界 IK 恢复基线逐帧净空；通过候选
最大净空误差 0.512 微米，COM 误差 0.295 微米，附加根旋转最大
0.04252 rad。这只解决参考接触几何，不是闭环完成证据。

`ten_transition_swing37` 为 9 步；margin30 + transition 为 10 步、846.68；
margin30 + COM gain1.25 为 10 步、852.44，是更远源帧但不是更多步数。
增益1.5为9步 QP失败，腰权重8为9步关节超限，腰余量0.5为9步，
加实测止挡模型也为9步。恒定 motion-speed0.5/0.8、三个 clock speed
均为1的全程降速（初始速度也缩放）分别为2/1步。激活阈值仍按未乘
motion-speed的 phase-time 计算，因此这些不是严格保持源帧激活点的
同控制器对照；未据此断言所有恒定低速方案都不可行。

平滑自由腿/平滑并恢复净空分别为9/9步。净空+margin30 数值失败
冻结 QP 同样在 cap10/prox1e-4 被原始 KKT 检查拒绝，多组其他参数
可解。新增默认关闭的 `--qp-late-equilibration-start/cap` 在求解前
切换数值缩放，不重试、不降低KKT标准；cap30从控制时间3.7s启用
后该候选达到10步/842.84，未完整 tracking。十五项 full tracking
测试通过。

新增默认关闭的 `--soft-joint-braking-start/weight/joints`，默认只选
腰横滚。`g1_soft_joint_braking.py` 生成单侧加速度制动不等式；
`SoftTrackingTubeBackend` 在同一个 QP 添加独立惩罚松弛变量。
原动力学、接触、摩擦、电机和硬限位行补零，不获得任何松弛。
原 tracking tube 三个松弛诊断保持原形状，另记制动权重和松弛。
区别于固定减速度的等式软任务：已经更强的向内制动不被拉回较弱
目标。三项新增测试验证该性质、解析解、单次原生求解和物理等式
优先；连同原九项管道测试共十二项通过，十五项 full tracking 通过。
关闭选项的完整十步复验和两组开启实验正在运行，默认入口不改。

单侧制动关闭复验完成：`soft_braking_disabled_verification.json` 记录
155 项共同历史字段逐元素等于 `ten_verify`。单侧腰制动 weight100
在基线上仍10步、833.72；margin30+COM1.25+brake100 为9步、
851.48，关节误差超限。未提高步数，默认关闭。

`prepare_lower_swing10.py` 在780–802之间用C2中段偏置降低右摆脚
15/25 mm，端点、根、左腿和上身不变。25 mm候选的原生最大净空
仍约3.25 cm；实际2 cm离地验收不变。基线lower15为8步QP失败，
lower25为10步/829.88；margin30+COM1.25+lower25为10步/853.40，
关节误差0.20166超限，不能计为完整tracking或步数新纪录。

`early9_foot355_frozen_tube_feasibility.json` 用HiGHS和DAQP独立确认
原先提前第九步+提前脚任务候选的冻结QP：原硬朝向辅助管道不可行，
只允许该辅助管道有惩罚松弛后可行；原始姿态验收和物理行不变。
复跑early9 foot355/370 soft orientation为8/9步；仅基线foot370为
10步、关节超限。不能把这些失败一概归因于求解器数值。

新增 `prepare_early_landing9_gentle.py`，从680帧逐渐建立相位提前量，
替代旧730帧起、8帧内完成的快速相位变化；COM升降过渡也延长，
于822帧恢复。lead8/z15通过严格IK预检，lead12/z15失败，lead12/z25
通过（原始关节预检峰值0.16846 rad）。gentle lead8闭环9步，但
左脚100N首次时刻由770.36提前至764.60，右脚卸载由777.08提前至
773.72，step9在776.12通过。lead8+foot370+soft orientation仍9步。
lead12+foot370+soft orientation在746.36处数值拒绝；冻结原QP在
多组其他缩放参数通过原始KKT，已开始cap30从control3.45的复跑。

`prepare_early_landing10.py` 现支持指定准备参考作为父文件；旧无参数
行为保持。gentle lead8上叠加第十步提前4/6帧的两份参考均通过原有
IK/原始姿态预检，用于检验实际更早卸载后能否更早建立右脚承重。


### Continued full-source tracking: COM diagnostic attribution and support tradeoff

Full tracking is still **not achieved**. Verified default remains 10/29 landings;
the furthest observed source frame remains 853.40, with the original thresholds.
The original PKL and default single-QP runner are unchanged.

Gentle early ninth touchdown plus earlier tenth touchdown (4 or 6 frames)
verified only 9 landings and stopped on joint RMSE at source828.92/793.40.
The numerically repaired larger early-ninth reference also reached9/source806.84.
COM-only lowering25mm and approach speed envelopes0.5/0.8m/s reached9 but
failed tenth-flight clearance. These candidates are not promoted.

The gentle early-ninth trial loses BOTH measured ground supports around790,
where the baseline still carries roughly100N on the left. Predicted COM
acceleration and measured interval acceleration agree during ordinary support;
the discrepancy is chiefly a task/contact trajectory tradeoff at these states.

A diagnostic regression was found after policy composition began discarding
local task objects: `com_acc_requested` fell back to the legacy attractor
(clipping the total acceleration), and `com_task_metric` reported base weights.
These fields in historical composed-policy logs are not the actual task target.
Saved `composed_task_center_of_mass_*` QP snapshots remain authoritative.
At baseline control3.700s, the actual target is az=-9.6689m/s² with vertical
weight320; the QP chooses az=-6.8109 with the100N support floor active.
An offline frozen-QP counterfactual, keeping physical constraints unchanged,
raises Fz to182.99N by bounding the desired az below at-3 and multiplying
vertical task weight by4 (`frozen_vertical_com_tradeoff.json`). This is a
local counterfactual, not a closed-loop success or a feasibility certificate.

`test_g1.py` now saves detached numeric final COM target/metric before deleting
local leaves. `com_diagnostics.py` reads those values and labels the source
`composed_policy_task`; no task object passes into the sole inverse-dynamics QP.
Six COM diagnostic tests and15 full-tracking profile tests pass. An independent
baseline rerun is in progress to verify identical physical control histories.

Applying vertical support weight4 from source-clock3.6s yielded9/source806.36
on the baseline and9/source830.84 on the improved waist/free-leg reference.
Support floor250N yielded9/source825.08. All three failed original tenth-flight
requirements, so the next test delays the change until source780 rather than
changing the ninth landing and unloading transition.

COM diagnostic verification completed: all152 other history fields are
bit-for-bit identical to `ten_verify`; the three changed diagnostic fields
now match the saved composed COM QP target and diagonal metric exactly at3.7s.
See `com_policy_diagnostic_verification.json`. No physical control changes.

Delaying vertical support adjustment until source-clock3.9s preserved10
landings and reached852.92 with jointRMSE0.18960/rootposition0.18797m;
rootorientation0.35552rad stopped the run. Pelvis roll weight2 from3.7s on
the modified reference reached only8, breaking ninth loaded hold. It is not
promoted. Next tests isolate pelvis roll after the tenth landing, and reduce
the turn clock smoothly from1.2 to1.0/0.8 over control3.2–3.6s.
Independent RNEA audit of the modified reference (`audit_turn_retiming.py`)
predicts the minimum nominal Fz across780–802 changes from-34.60N at1.2
to75.92N at1.0 and166.33N at0.8. Peak nominal motor ratios remain above1;
this comparison motivates simulation but does not certify feasibility.

Turn slowdown to1.0/0.8 from control3.2s yielded only8 landings, ending
on root orientation at773.80/767.64. Delaying slowdown to control3.78s
preserved10 but reached846.35. Releasing the unused20% effort reserve
within actual URDF limits after source-clock3.95/4.05s reached10/source855.80
(3.970s); neither completed landing11. The later release stopped at
rootorientation0.350813rad with jointRMSE0.199538 and rootposition0.186733m.
This extends the furthest observed source frame, not the verified step record.

At the3.900s frozen state of the delayed-vertical-support trial, waist roll
q=0.36176rad and v=2.84291rad/s imply measured acceleration bounds
[-98.71894,-38.55894]rad/s² with bandwidth8. Adding these bounds remains
feasible and changes predicted root roll acceleration to-16.01rad/s².
`frozen_waist_braking_after_landing10.json` records this local counterfactual.
Next tests activate named waist-roll braking after source810, preserving
the earlier ninth landing. The first launch approval timed out before creating
any simulations; the single permitted retry succeeded.

Named hard waist-roll braking from source810 preserves10 landings but reaches
838.04 (80% effort) or846.68 (rated effort after810), both stopping on
jointRMSE while rootorientation remains0.23462/0.25472rad. Soft braking
weight1000 reaches10/source850.04, jointRMSE0.20222, orientation0.26714.
At846.68 hard-braking shoulder-roll errors grow to approximately-0.49/-0.47
rad versus-0.26/-0.24 without braking. It redistributes the balance burden;
it does not by itself establish tracking feasibility.

The improved-reference right stance is native point/edge support after802.
Measured right foot roll is0.138rad near815, almost zero near830, then0.201
rad near845. Its free angular support task has zero policy metric in the
current entry. Existing support-orientation/flatten options are now tested
from source802 with weights4/16, including a soft-braking combination.
Actual point/edge contact rows and moment cones remain unchanged.

Support flatten/promotion simultaneously from source802 reached10 but failed
at823.64/824.12 on jointRMSE, with right foot roll0.33–0.38rad and continued
point support. The original orientation target is promoted during the0.1s
blend into flattening, so this does not isolate a fully established level-sole
objective. Next variants preblend flattening from780 while its policy weight
is still zero, then promote the completed flattening task from820, after
landing10. Native geometry, motor limits and all original audits remain.

Preblending flattening then promoting it at source820 preserves10 landings,
reaching852.92 with rootorientation0.30752/0.30723rad for scales4/16,
but jointRMSE0.20257/0.20314 exceeds the unchanged0.2rad threshold.
The combination with soft waist braking reaches849.56/jointRMSE0.20526.

Before the turn the COM already lags horizontally by0.07–0.10m.
`--qp-com-feedback-axes xy` now optionally restricts the existing extra COM
quadratic objective to XY via a positive-semidefinite row/column mask.
Default `xyz` and zero-weight behavior are unchanged. This permits staged
horizontal weighting without duplicating a negative vertical feedforward
target over the existing vertical support task. It does not add a solve.
`com_acc_requested` still describes the composed local COM task;
`qp_com_feedback_scale/axes` identify this separate original-feedback term,
and `com_acc_qp_reference` identifies the combined generalized objective center.
Seven acceleration-objective tests and15 full-tracking profile tests pass.
Three closed-loop tests activate horizontal weight4 at control3.2/3.4s.

Additional XY objective weight4 failed8/source773.72 when enabled3.2s,
and8/source789.56 when enabled3.4s. The latter with delayed flattening is
identical because it stops before flattening activation. Rootorientation
reaches0.35 before the ninth loaded hold completes. All161 prior history
fields are exactly identical to the parent before the optional activation
(`com_xy_prefix_verification.json`). New objective remains optional, unused
in the verified default.

`prepare_turn_com_offset.py` prepares nearby full-body references on the
improved free-leg/waist/lower-swing parent: COMY+15/+30mm rises680–760,
holds through900, fades by1000. Both original control foot poses are held
within0.6micrometres; source joint RMSE peak remains0.14258rad; maximum
root displacement is0.05521/0.07227m and added rotation0.02421/0.04579rad.
This is bounded offline IK, not dynamic feasibility or full tracking.
Both candidates are now tested with the parent late-vertical/rated-budget
controller. A third trial promotes the unmodified source foot orientation
from820 without mixing into a separate flat-foot target.

COMY+15mm reaches10/source857.24 (3.976s), jointRMSE0.19706,
rootposition0.18736m and rootorientation0.35520rad. COMY+30mm reaches
only9/source821.24. Source-foot support orientation from820 reaches
10/source853.40 but jointRMSE0.20145. Default runner is still verified10.

The right swing foot already has approximately0.10rad world roll error
at source798 relative to its control reference, before touchdown. The
configured foot angular feedback norm is25rad/s² (stiffness scaled450).
`--late-foot-angular-feedback-limit` optionally replaces that feedback cap
within the existing angular metric window and foot-side selector. The
source angular acceleration is retained; only the soft task target changes.
DefaultNone leaves previous control untouched. A pure helper and two tests
verify world rotation-error sign, norm clipping and preserved feedforward;
15 full-tracking profile tests pass. Three right-foot swing experiments
compare angularweight4, angularweight4+cap100 and cap100 alone.

Right swing angular weight4 alone reaches10/source857.24, jointRMSE0.19117,
footroll at802=0.06857rad. Weight4+feedbacknorm100 reaches10/source858.20
(3.980s), jointRMSE0.18258, rootposition0.18651m, rootorientation0.35528rad,
and footroll at802=0.01140rad. Cap100 alone reaches10/source856.76.
All161 history fields before source760 exactly reproduce the parent
(`foot_angular_prefix_verification.json`). These are improvements in
landing pose/joint margin, not an additional verified landing.

Two optional offline right-roll references were also prepared with fixed
root and other joints, unchanged foot origins, negative worldroll0.08/0.12
during770–826. Raw foot pose residuals are below0.6micrometres/radians,
COM changes below0.61mm, native right swing clearance peaks0.0352/0.03565m,
and all leg limits pass. They are untested in simulation and not promoted;
feedbackcap100 already nearly removes the pre-impact roll error.
Next simulations combine the latter with soft waist braking100/1000 or
source support-foot orientation from820. An approval timeout occurred before
launch; the permitted retry uses the inspectable batch manifest
`batch_ang100_combinations.json` and shell-free `run_batch.py`.

Rightcap100 combinations remain10 landings: waistsoft1000 reaches854.36
(jointRMSE0.20012/rootorientation0.28515), waistsoft100 reaches857.72
(0.20266/0.30288), stance orientation reaches857.24 (0.20400/0.32920).
The new joint margin buys several frames but does not complete landing11.

At source850 the LEFT swing-foot world rotation error relative to the
control reference is approximately(0.378,-0.045,-0.375)rad, while its
foot-origin error is millimetres. This foot still uses the25rad/s² angular
feedback norm limit. `batch_both_angular.json` extends cap100 to both feet
with angularweights1/4 and activation source760/770, without a late cutoff.
No original-source acceptance or physical constraint is changed.

Both-foot angular feedback cap100/metric4 reaches10/source854.84; starting
at770 instead of760 reaches10/source859.16. Metric1 loses step10 flight
and stops at814.52. No additional landing is verified.

Moving torso orientation weight into pelvis from810 gives10/source858.68,
but reduces stop jointRMSE to0.15834 (rootorientation0.35263). Starting790
reaches857.72/jointRMSE0.15315. Combining with both-foot feedback reaches
858.68/jointRMSE0.17814. Lowering stance posture weight to0.25/0.1 from810
reaches858.20/858.68. Changing all auxiliary tube bandwidths to5 from810
reaches857.72 and fails both joint/orientation limits.

`turn_task_conflict.json` re-solves the same archived physical QP at3.8/3.9s.
Original acceleration reproduces within2.8e-5rad/s². At3.9s, softening only
the position tube changes COMZ acceleration from+3.06 to-2.38m/s²; changing
stance weights alone does not remove it. This is a local objective/constraint
conflict diagnostic, not an alternate successful controller or a change to
original-source acceptance. All ablations retain physical constraints.

Optional pelvis allocation metric scaling and angular feedback cap now
preserve the same source feedforward and fade in with existing allocation.
Defaults1/None preserve previous behavior. Three allocation tests, two
angular feedback tests and15 full-tracking tests pass. The first pytest
invocation could not run because pytest is absent; unittest completed20.
The allocation helper is now included in each new trial source archive.

Offline `prepare_fullroot_dynamics.py` extends the existing 32-coordinate
fit with root rotation freedom. For Exp(delta) R(t), angular variations
include delta_dot-[omega]x delta and delta_ddot-[omega]x delta_dot-
[alpha]x delta. Independent finite differences verify both derivatives;
22 combined tests pass. Exact exported-reference dynamics and closed-loop
simulation remain required; the offline soft inequalities are not a physical
feasibility certificate. This is reference preparation, not runtime MPC.

Pelvis feedback cap100 alone reaches858.68; metric4+cap100 reaches859.64,
with jointRMSE0.17019 and rootorientation0.35293. Both exactly reproduce all
163 existing history fields before source810 (`pelvis_allocation_prefix_verification.json`).
Pelvis replacement + soft waist1000 initially fails QP certificate at839.96;
frozen numerical audit finds cap1/3 passes, cap10/prox1e-4 rejects. Closed-loop
cap3 fails at another nearby state; cap1 reaches858.20/jointRMSE0.20114,
orientation0.25381. Waist100 reaches861.56/jointRMSE0.20163/orientation0.32310.

Late clock speed1.4 fromcontrol3.82s reaches10/source863.32;1.6 fails QP at
source862.36;1.0 reaches858.20. Actual advancement is still10/29, not11.
The completed290-trial audit verifies original tolerance values, one native
solve/update, no cached task requests at physical QP, no second QP/PD, and
exact equality of applied torque to the saved QP torque. Three startup/crash
records are kept separate from completed tracking results.

Full-root fit r10/l10/u05/a10 converges (79100 iterations). Full-scale foot
position change0.025508m exceeds unchanged0.025m preparation bound, so no
full-scale candidate was exported. Scaling0.8/0.9 yields max foot position
0.020258/0.022874m and angle0.14530/0.16281rad, both pass. Composing waist-0.3,
right swing-25mm and COMY+15mm passes geometry and native clearance. Exact
exported dynamics still has force/CoP/effort violations, particularly after
composition. Three explicit GPU cases are in `batch_fullroot_com15.json`.

To fit those corrections together, `prepare_fullroot_dynamics.py --parent`
can linearize the complete current reference. `batched_reference_dynamics.py`
matches scalar RNEA, COM angular wrench, nominal contact allocation and
armature on8 random states (all four support modes), to1e-10 absolute
precision without input mutation. It vectorizes offline calculations only.
Direct full-root fitting around `prepared_margin30_turn_com_y15` is running;
nonlinear export checks and closed-loop verification remain required.

First full-root composite closed-loop trials all stop at9/source827–827.48;
right10 unloaded native peak is0.01773/0.01736m, below unchanged0.02m.
Reference geometry alone did not preserve actual flight evidence.

Direct full-root fitting of the complete COM15 parent solves in12225
iterations. Linearized maxima: motor0.01689 of budget, friction0.000536,
normal0.000537, CoP0.08759m. Exported full-scale foot angle0.23232 exceeds
unchanged0.20 preparation bound. Scale0.8/0.6 passes geometry, native right10
reference clearance0.02586/0.02741m, and exact exported dynamics is audited.
Three GPU cases are `batch_fullroot_direct.json`. Batched cache's angular
variations independently match scalar nonlinear RNEA to2.6e-7 maximum
(`prepared_margin30_turn_com_y15.fullroot_linearization_verification.json`).

CoP residuals were measured in metres whereas force/torque residuals were
ratios. Optional offline `--cop-slack-length .05` makes their soft penalty
length explicit; reported violations remain in original units. A separate
`--swing10-clearance-margin .002` protects each of13 native right-foot points
at source782–800 with a C2 height margin, using full floating-base/leg
point Jacobians. Independent native-height finite differences verify the
rows. These are offline reference constraints; runtime acceptance is unchanged.

Direct full-root fits scale0.8/0.6 still reach only9/source830.36; actual
right10 native clearance is0.01269/0.01412m. Clearance-constrained versions
scale0.8/0.6 restore nominal native peaks0.03551/0.03470m, with worst
native point height loss only0.00031m outside the constrained sample set.
Exact exported motor demands are audited before `batch_fullroot_clearance.json`.
23 combined unit/profile tests pass (`fullroot_and_pelvis_tests.log`).

A late-only offline fit uses first free Greville knot828, so the cubic basis
has no effect before source816. It aims to preserve all ten measured landings
before changing the remaining turn. Runtime-spline prefix equality and full
closed-loop results are still required. CoP-normalized full-window fitting
converged but has a35.7microradian joint-bound residual, exceeding the fixed
1e-5 preparation tolerance; it was not exported. A warm-started offline fit
with tolerance5e-7 is running instead of relaxing that bound.

Effective amp-full preset already sets global posture feedback limit150 and
body angular feedback limit100. The optional pelvis norm100 override differs
from the existing component cap; it does not raise a25rad/s² body cap. The
25rad/s² cap diagnosed earlier applies to the original foot angular task.

Clearance-constrained scale0.8/0.6 improves actual flight to0.01801/0.01892m
but still only9/source828.92–830.36. Thus those candidates are not promoted.
Early position-margin trials use control tube radius0.17/0.16m fromsource
3.4/3.6s while the independent original-source limit stays0.2m.

Tighter solver tolerance did not remove the35.7microradian bound residual.
`fullroot_between_knots_limit_diagnosis.json` identifies left ankle roll at
source911, between bounded Greville knots. Parent positions are not linear
between knots, so knot-only bounds do not guarantee exported raw-frame bounds.
The full-root preparer now constrains every affected integer-frame joint
position using exactly the export interpolation operator. This fixes the
preparation constraint rather than relaxing joint limits or clipping output.

Late-only CoP-normalized fit hits150000 iterations and is not exported.
The ordinary CoP metric late-only fit converges22675 iterations; full-scale
footposition0.02562m fails preparation, but scale0.9/0.6 passes. Nominal
linearized residuals after816 are torque0.00286, normal0.000242, friction
0.00115 and CoP0.03216m. Exact exported dynamics and prefix q/v/a are checked
before simulation; no closed-loop result is claimed for this fit yet.

Late-only full-root scale90/60 preserves every163 prior history field through
source816 (1903 samples); the brake variant preserves165 fields. Results remain
10/source858.68/859.64; brake variant reaches854.36 and fails jointRMSE.
See `late_fullroot_closed_loop_prefix_verification.json`.

Position tube radius0.17m fromsource3.4/3.6 produces identical10/source850.52,
rootposition0.16795m and rootorientation0.35507. Radius0.16 reaches846.20,
rootposition0.15503m. Combining0.17 with pelvis4/waist100/waist1000 reaches
10/source853.40/851.96/849.08. No additional source landing is verified.
Raw-frame-bounded CoP fit hits100000 iterations and is not exported.

`prepare_support10_early.py` now accepts an explicit bounded XYZ offset while
retaining original default[0,-.025,0]. Three COM15-parent fixed-COM/left-foot/
foot-orientation IK candidates pass geometry and exact dynamics screening:
X+20mm, Y-20mm, and X+15/Y-15mm. Nominal native right10 clearance remains
0.033088m. X and combined variants give9/source831.80/830.36; Y-20mm gives
10/source863.48 (4.002s), jointRMSE0.18628, rootposition0.18862m,
rootorientation0.35158rad. Original-source landing acceptance stays40mm.

`g1_support_flatten.py` adds optional measured-stance flat-sole feedback gains,
using the separate sole heading and measured edge/point coordinate frame.
Defaults gain1/cap25 leave existing code untouched. Optional gain3/cap100
uses stiffness150/damping14*sqrt(3), still only as a soft free-axis task and
only on measured loaded planned stance. Physical contact rows/cones are not
changed. Two helper tests (including distinct sole/edge frames) and15 full
tracking tests pass. `batch_support_flatten_feedback.json` compares existing
gains versus gain3/cap100, with flat preblend3.9 and promotion4.01 rather
than starting source-foot-to-flat blending only at contact promotion.


Support flatten promotion4.01 gives10/source823.64 (default gains),825.56
(gain3/cap100),825.08 (metric16), failing joint RMSE. Later promotion4.10,
metric0.25/1 gives10/source863.00/860.12; promotion4.15 metric1 gives859.16.
None improves verified landing count or the863.48 best observed phase.

`prepare_path_retiming.py` optimizes speed squared along a fixed prepared pose
path offline. Exact RNEA phase scaling G+X*v²+U*a is independently checked to
3.41e-13. CoP/friction/normal/actuator conflicts have explicit slacks; a solved
fit is NOT a physical feasibility certificate. `g1_prepared_clock.py` adds an
optional hash-bound fixed C2 Hermite source clock, preserving the original
clock prefix and a constant-speed tail. It performs no runtime optimization.
Original-source same-frame fidelity and all landing criteria are unchanged.
Two clock tests plus15 existing full-tracking tests pass. Quintic smoothing
can increase the discrete acceleration bound6 to about7.26rad-equivalent
phase/s²; exported preflight reports this rather than claiming the original
bound still holds. Small optimizer equality residuals (below1e-6) at the
initial knot are explicitly removed to preserve exact C2 boundary values.

The whole-turn clock (source660..1032) reduces nominal motor residual1.98 to
0.44 and normalized CoP residual5.36 to3.23 but fails closed-loop at774.32,
rootorientation0.35009, with only8 verified landings. Late clock variants
starting780/816 are separately optimized and checked before simulation;
initial inaccurate/not-converged fits are retained, and refined solved fits
are used. A root-acceleration audit finds substantial roll error already at
source750..780 and upward root velocity1.82m/s at850; most stable-contact
accelerations match the QP, while impacts near816/850 give large deviations.
The independent orientation-radius0.27 trial starts atsource3.2; raw PKL
acceptance remains0.35rad.


Late prepared clocks retain10 landings: start780 stops852.90 on jointRMSE,
start816 stops863.41 on rootorientation. All163 history fields other than
wall time are exactly identical to the parent through their intended prefixes:
1577/1828/1903 updates for starts660/780/816, respectively. See
`prepared_clock_prefix_verification.json`. No clock variant is promoted.
Orientation-radius0.27 fromsource3.2 gives10/source816.92, stopping at joint
RMSE0.20015 despite orientation0.20975 and position0.15675. The next motor
budget combination uses100% actual URDF limits earlier; it does not raise
those physical limits. A separate optional `--late-position-tube-start` and
`--late-position-tube-bandwidth` changes only the root-position auxiliary CBF
rate over0.1 source seconds, explicitly retaining joint/orientation rates.
Defaults are disabled;15 full-tracking tests pass after integration. Two
trials compare rates2/5 fromsource4.08, with all source acceptance unchanged.


Position-only auxiliary bandwidth2/5 gives10/source860.12/862.04, still
orientation failure. Earlier full URDF budget with orientation0.27 gives
8/source796.28 joint failure. Fourth speed1.8 (control3.81,ramp0.12) reaches
10/source869.24, joint0.20841, position0.18465m, orientation0.33458rad; speed1.6
with preselected scaling cap1 fromcontrol3.82 reaches10/source867.00,
orientation0.35694. No additional landing has been verified. The rate-row
isolation test exposed an orientation diagnostic using generic bandwidth;
it now uses the actual orientation bandwidth. The physical inequality was
already correct. One independent row-isolation test passes.

`prepare_stance10_anchor.py` uses the actually loaded right-foot pose near
source816 for a bounded offline reference correction. The parent reference
continues moving/yawing its stance foot while the physical foot stays nearly
fixed through830. The initial version faded after884 and fails bounded IK
near898; no candidate is exported. The revised860..884 fade passes with
COM/foot residuals below6e-7, and raw-source peak jointRMSE remains0.14258.
Both position-only and full-pose variants preserve runtime q/v/a exactly
through816 at1634 samples. See `stance_anchor_prefix_verification.json`.
Exact nominal whole-body dynamics still exceed motor limits (equal-force DS
assumption,80% URDF audit), so neither geometric success nor preparation
is a physical feasibility certificate. Closed-loop checks are required.


Measured margin weighting after816 gives10/source860.60 (ordinary clock,
maximum4),867.08 (speed1.8/maximum4, QP certificate failure),869.24
(speed1.8/maximum16, jointRMSE0.20790). The position-only stance anchor gives
10/source862.52; full stance-pose anchor gives863.00. Combining full-pose
anchor with fourth speed1.8 reaches870.68, jointRMSE0.19730,
rootposition0.18657m and rootorientation0.35051rad. Still10/29 landings.

An earlier stance anchor starts raw-frame802; its runtime B-spline influence
begins792, not796 (first changed Greville coefficient804 has support from792).
The initial overlong prefix assertion correctly rejected the claimed796
prefix; `stance_anchor_next_prefix.json` records the actual exact792 prefix.
Late swing11 lowering by25mm preserves COM and right foot/foot orientations,
retains exact q/v/a through816, and has dense native peak clearance0.05595m.
The40mm version fails bounded IK (0.59mm foot/0.38mm COM residual) and is not
exported. No candidate with failed preparation is simulated.

`prepare_fullroot_dynamics.py --prepared-clock` now recomputes exact reference
q/v/a and all RNEA derivatives at the offline retimed clock. Clock hashes form
part of the derivative-cache key and the preparation report. The initial
clock660 full-root fit solves in6625 iterations, with only24.2mm root and
0.0206rad root-rotation correction; geometry passes. The pair is explicitly
bound by `audit_prepared_clock_reference.py`: it records the original clock
planning parent and the subsequent pose fit, moves original clock slack into
`planning_physical_slack`, and hashes the new pose/clock pair. This does not
claim old clock slack is valid for the child pose. Runtime q/v/a remains exact
through660. Exact every-frame audit finds maximum nominal effort ratio1.3304
and CoP residual0.1710m at883, an odd frame omitted by the original two-frame
fit grid. Native right10 clearance peak is0.02653m. A separate
`--frame-stride 1` fit now includes every integer source frame and has a
separate cache; the default two-frame mode is unchanged. These are offline
fits, never runtime MPC or extra inverse-dynamics solves.


Earlier stance anchoring with speed1.8 gives10/source866.36. Late swing11
lower25 plus speed1.8 improves the observed prefix to874.28, still10 landings,
jointRMSE0.19676, rootposition0.19457m, rootorientation0.35141rad. It is not
full tracking and is not promoted over the independently verified default.
Five stance/lowering runs preserve all163 prior history fields exactly (wall
time excluded) through their intended prefixes:1903 samples through816 for
late edits and1853 through792 for the earlier anchor. See
`stance_anchor_closed_loop_prefix.json`. The two next comparisons use fourth
speed2.0 or pelvis orientation replacement scale4 aftersource4.08. Existing
single-QP and unchanged original-source acceptance remain enforced.


Clock-aware coarse full-root reference remains8/source770.04, orientation
failure; nominal dynamic fit improvements do not guarantee feedback recovery.
Fourth speed2.0 on lower25 fails QP validation after869.40; all12 frozen
numerical variants fail. Frozen auxiliary-row ablation certifies only when
both root-position and root-orientation barriers are softened, while physical
constraints remain unchanged. This is a diagnostic, not an adopted relaxation.
An offline necessary-feasibility probe with one scalar phase acceleration in
the three tube rows also fails for bounds6/20/100; no runtime phase controller
is implemented. Pelvis scale4 with speed1.8 yields10/source872.84, stopping
on jointRMSE0.20223, orientation0.28678. Best observed phase remains874.28.


Dense clock660 full-root fit solves40750 iterations. Geometry passes:
maximumfootposition0.02249m, footrotation0.16984rad, rootcorrection0.03392m.
Exact exported audit reduces maximum nominal effort ratio1.33040 to1.05296
and CoP residual0.17103m to0.09096m; source883 is now included. Native right10
reference clearance remains0.02701m. Closed-loop comparisons use current
bodymetric0.3 versus1; the reduced-speed clock may need different feedback
allocation. A separate clock816 dense fit starts free knot828, preserving
source816 q/v/a and the ten-step prefix. Swing11 lowering30mm passes IK with
residuals below6e-7 and dense native peak0.05485m;40mm remains rejected.


Dense clock660/current body metric stops8/source762.51; body1 stops8/753.80.
Lower30 retains10/source874.28 with joint0.19447, position0.19419m,
orientation0.35085; no additional landing. The independent late clock816 dense
fit solves91325 iterations and passes geometry, with rootcorrection0.06366m,
rotation0.13655rad, footposition0.02143m, footrotation0.14374rad. Exact audit
is required before simulation. A new offline analytic fourth-clock option
fits the best lower25 parent under its actual1.8 schedule, recording the
configuration and using a distinct cache identity; no runtime code is added.

`prepare_lower_swing11.py` optionally drops COM together with the foot. The
40mm/20mm and50mm/30mm combinations pass bounded IK. A separate floor3 mode
keeps the lowering through the landing phase, caps it by the parent native
foot height minus3mm, and fades900..924 instead of860..884. Dense native floor
minima remain1.70–1.73mm after spline interpolation, with swing peaks above
0.0515m. All three selected candidates preserve q/v/a exactly through816
(`lower11_comdown_prefix.json`). Only the method text of two preparation JSON
reports was corrected to describe their explicit nonzero COM drop; numerical
results/candidate arrays were unchanged and the correction is annotated.


Lower40/COMdown20 gives10/source877.16; without landing floor it fails joint
RMSE0.20750 and rootposition0.20046. Floor3 keeps joint0.19016 and
orientation0.31215 but rootposition0.20074 still fails. Lower50/COMdown30/floor3
stops10/source861.32 on jointRMSE0.20068. No11th landing is verified.
At the floor3 best stop, root errors are[-0.1321,-0.1357,+0.0665]m; reference
position progression now limits the result. Old source-tube smoothing at a
frozen3.9 state reduces predicted adverse roll acceleration25.17 to14.49/11.30
for strides4/6, though COMZ stays near+8.7; no physical rows are removed.
Both smoothings are separately tested aftersource4.08. Frozen failed speed2
state stays infeasible for all tested smoothing scales; smoothing is not a
numerical rescue of that state.

The late clock816 dense exact audit separates unchanged prefix from new tail:
816..884 nominal maximum motor ratio0.95352 and CoP residual0.05453m;
885..1032 ratio0.89713 and CoP0.05139m. Prefix660..815 still has ratio2.03735,
which is not caused by the late fit. Aggregate maxima alone obscure this.
The revised exact auditor records all three windows and prefix equality.

The late clock816 fit originally encountered a numerical certificate rejection.
A preselected equilibration cap of1 from control3.806 removes that rejection;
closed-loop tracking then stops at858.67 on orientation0.35207rad, with10
landings. This is numerical progress, not an additional verified landing.

`g1_foot_linear_feedback.py` adds an optional norm bound for swing-foot linear
feedback while preserving reference acceleration. Defaults remain unchanged.
The left cap60 from source4.08 on lower40/COMdown20/floor3 reaches877.16 with
jointRMSE0.17658 and orientation0.30349, but rootposition0.20065 still fails.
Cap30 fails876.44 on joints; fourth speed1.6 fails873.40 on rootposition.
Both new feedback unit tests and15 full-tracking acceptance tests pass.

`g1_root_xy_tube.py` adds an optional horizontal-only root barrier alongside
the unchanged original 3D position, joint and orientation barriers. At a frozen
3.9 state it reduces predicted upward COM acceleration6.57 to0.34; actual
closed-loop radius0.18/0.17 with bandwidth10 stops862.04/859.88 on joints.
Rootposition improves to0.16370/0.15730m at those stops, but no11th landing is
verified. Its two tests independently check the barrier derivative and vertical
invariance. The default entry remains the independently verified10-step baseline.

The offline full-root fitter now has an explicit optional Clarabel backend
(`g1_offline_bound_qp.py`), used to investigate a fast1.8-clock fit for which
OSQP did not converge. Runtime remains one DAQP call. Signed multipliers,
mixed equality/one-sided bounds, KKT residuals and infeasibility reporting are
checked by two analytical tests. Exported nonlinear dynamics and closed-loop
validation remain mandatory; an optimizer success alone is not tracking.

Clarification on foot feedback: the generic attractor norm cap is15, but the
effective amp-full profile already replaces horizontal components with cap80
and vertical with cap40 aftersource1.4. The new late norm cap60 replaces those
existing components on the selected foot; it is not simply raising15 to60.
Saved `controller_options` confirms the effective settings. Prefix histories
remain exact:163 shared fields/1903 updates before816 for linear60, and165
fields/1947 updates before840 for the late XY option (`late_feedback_xy_prefix.json`).

Gentler XY bandwidth5 from816 still fails joints859.88. Starting it at840
instead reaches877.88, with10 landings, joint0.20359, position0.19806m and
orientation0.33512. Cap100 on this variant rejects the QP after874.28; upper
posture weight32 stops864.20 on joints. Neither becomes the default.

Fast1.8 offline fits exposed two interpolation issues. The optimizer previously
bounded a smoothed foot correction while export linearly interpolates the
coefficient samples into raw frames. Optional `--bound-exported-foot-pose`
adds raw-frame constraints as well. Foot rotation drops from0.20373 to0.12432rad
and the candidate passes unchanged geometry preflight. Exact dynamics still
show motor ratios3.01 in816..884 and3.50 in885..1032, so this first candidate
was not run. A smaller relinearized correction (`export_refine1`, with foot
curvature penalty1) reduces those ratios to1.00091/1.16665 and CoP residuals
to0.15289/0.09078m; two actual tracking comparisons are started.

`g1_exported_rotation_maps.py` optionally differentiates the actual normalized
quaternion coefficient spline, instead of applying a continuous world rotation
after interpolation. A test perturbs exported raw poses and independently
reconstructs q/v/a with `BlendedSplineReference`; the Jacobians agree. Both
preparation modes and all hashes remain recorded. This optional offline change
does not add runtime solves or change original-source acceptance.

The exact quaternion-derivative first fit alone still has nominal late effort
ratios3.34/3.51 and is not promoted. Smaller relinearization is essential.
Increasing the offline CoP penalty after refine1 worsens the exact tail ratio
to1.53; that candidate is also not run. A new optional source-joint squared-error
penalty explicitly attracts raw exported joint poses toward the original PKL.
With weight1e6, the summed per-frame mean squared error falls1.6901 to1.0111,
then0.7634 after one small relinearization. Its raw jointRMSE peak stays0.14258;
exact late effort ratios are1.07448/1.33558, CoP residuals0.15847/0.14477m.
These remain nominal diagnostics, not physical-feasibility certificates.

Closed-loop refine1 without the source-joint penalty stops872.84 on joints
0.20374, despite orientation0.19394. At this stop waistroll is at0.52rad and
the unconstrained-stop model predicts outward acceleration459.43rad/s².
Enabling the existing observed waist-stop reaction model fromsource816 reduces
joint error but still stops872.84. On floor3_linear60 it reaches878.60 with
joint0.20037, rootposition0.20067 and orientation0.30642. COMZ weight16 instead
stops872.12. Combining the stop model with late XY stops870.68.

Source-joint-penalized refine1 stops865.64 without the stop model and878.60
with it: joint0.20106, position0.19673m, orientation0.32932. All retain10
landings. No11th landing is verified and the default baseline is unchanged.

Frozen stop-model QPs show normalized joint-tube slack138.99 at3.8,
3.80 at3.9, and3264.26 at4.0. Hardening that row is locally feasible at3.8/3.9,
but fails at4.0. Optional `--late-tube-soft-weight-start/--late-tube-soft-weight`
therefore blends the existing slack cost over60ms of source time; it does not
change physical rows or acceptance. Current experiments use weights100/1000
from816, plus an independent existing soft-braking comparison for waistpitch
and rightshoulderroll. Nine soft-tube tests and15 full-tracking tests pass.

Late tube100/1000 initially reject the QP at869.24/868.52. Frozen numerical
settings never certify tube100 at that state; cap1 certifies tube1000, but its
full rerun rejects later869.24. Named braking initially rejects858.44;
preselected cap1 removes this numerical rejection and reaches876.44, failing
rootposition0.20017m. Source-joint-refine1/stop plus tube100 rejects874.28.
These experiments still verify10 landings; no runtime retries are introduced.

A separate reference audit identifies large root angular feedforward produced
by the prepared poses: at878.6 the floor3 root yaw acceleration is134.81rad/s²,
versus original-source7.85. `prepare_root_smooth.py` now accepts a local start
and ramp, with its old defaults unchanged. Gaussian widths8/16 over822..924
preserve both raw foot poses below8e-7 and change COM by at most5.18/5.78mm.
Runtime q/v/a is identical through816; dense native left clearance retains
minimum1.727mm and peak53.04mm. Root yaw acceleration at878.6 becomes27.29/5.55.
See `late_root_smoothing_preflight.json` for hashes and exact prefix checks.
Current closed-loop tests combine these references with the observed stop
model; a third case adds the existing pelvis orientation task atsource836.

Local smoothing8/16 both stop878.60, with jointRMSE0.19584/0.19406 and
rootposition0.20056/0.20026m. Smooth8 plus pelvis3 rejects the QP after875.
The full-tail sigma16 preparation is rejected by the existing geometry check:
worst foot position residual0.288mm at1490, versus its20µm bound. It is not
exported. A lower COM IK weight is being checked with the same acceptance
bounds, to prioritize exact foot preservation in that later region.

Root-position task bug fixed: `source` previously selected the local
`reference_q/v/a`, which already held prepared control poses. It now uses an
independent PCHIP sampler of the original ground-aligned PKL. `control` keeps
the explicit unified control q/v/a. The new `g1_root_position_reference.py`
tests source invariance under unrelated control-pose changes, clock derivative
chain rules and control selection. Two tests plus15 full-tracking tests pass.
Earlier experiments labeled source-root-task therefore did not test the
original source as intended; default runs had this task disabled. New root
weights300/1000/3000 fromsource816 are tested on smooth16/observed-stop.

True original-source root weights300/1000 reject the QP after875/823.70;
weight3000 stops861.32 on joints0.20040, with position0.17335m. No additional
landing results. The full-tail foot-dominant IK variant also fails existing
preflight (foot0.203mm, COM26.04mm), so neither full-tail file is exported.
The current comparison changes the existing supported COM desired-acceleration
lower clamp from−3 to−6/−9m/s², including−6 with vertical metric16. This clamp
is a task target policy, not a motor/contact physical bound. All physical
constraints and independent original-source acceptance remain unchanged.

COM clamp−6/−9 both reject the QP at875;−6/metric16 stops862.04 on joints.
Preselected cap1 on true source-root1000 reaches880.04, with10 landings,
joint0.18429, position0.20066m and orientation0.32523. Cap1 baseline smooth16
still stops878.60; smooth8/pelvis3 reaches876.44 but fails joints0.20601.

`late_support_loss.png` and its JSON show the measured failure sequence:
right normal load739N/count9 at815,110N/count4 at864.2,73N/count3 at871.4,
then no ground contact at878.6. Upward root velocity peaks near1.52m/s around857;
root angular speed grows sharply after support loss. This does not certify
which objective caused the loss. Frozen swing-foot orientation reweighting
has little effect at3.9, so no new runtime option was added for that hypothesis.

The drift begins before the ten-step prefix: at815, measured COMy−0.8900m,
vy−0.8529m/s versus original source−0.7969m/−0.7049m/s. An instantaneous LIPM
capture-point diagnostic differs by about12cm laterally; this ignores angular
momentum, impacts and future contacts and is not an infeasibility certificate
(`lateral_capture_point_audit.json`). Current experiments move the corrected
source-root task earlier, to660 after step8, with XYZ1000 or XY1000/300.
Optional XY axes remove only the vertical soft-task cost; an independent
natural-form invariance test confirms this, and all16 full-tracking tests pass.


Early true source-root feedback does not improve the verified prefix: XY100
from660 reaches9/815.96; XY1000 reaches8/743.48 (tube100:743.96). Stronger
XYZ/XY300 trials reject the QP. These remain experiments, not default changes.

Whole-turn full-root dynamics fitting now supports an explicit raw-source
joint squared-error penalty and derivatives of the actual exported quaternion
spline. `turn_sourcejoint1m` preserves q/v/a through660; geometry passes and
step10 nominal native peak clearance is27.58mm. Exact nominal effort ratios
in660..815/816..884/885..1032 are1.134/1.253/1.615. Runtime reaches8/777.08,
with jointRMSE0.10916 and orientation0.35080. Its bounded refinement lowers
those effort ratios to1.010/0.820/1.009 and CoP residuals to87.59/46.88/47.99mm.
These are soft offline constraints, not a dynamics feasibility certificate.
Refine1 runtime reaches8/794.84 on jointRMSE0.20175; step9 loaded hold42ms
falls short of50ms. Pelvis metric1/3 from660 reduces final orientation to
0.179/0.117rad but stops790.52/787.64 on joints, with landing errors82/92mm.

Extended right-stance anchoring through884 (fade908), local smooth16 root,
20mm extra COM drop and fixed root rotation passes bounded IK. Prefix q/v/a
through816 is unchanged, raw-source joint peak0.17243. Runtime alone rejects
QP876.44; adding source-root1000 reaches10/880.04 with joint0.19309,
position0.20017m and orientation0.32408rad. A40mm drop and a longer hold900
fail geometry preflight and are not exported. Current trials test the first
whole-turn fit with pelvis1/3, plus support-orientation metric4 on the valid
extended-anchor/root1000 case. Default verified10-step entry is unchanged.


First whole-turn fit with pelvis1/3 still reaches8/794.84 or787.16; no9.
Extended-anchor support-orientation4 reaches10/880.04 on orientation0.35225.
Support1 reaches10/880.76, position0.19801m, joint0.19464, orientation0.35265;
no loaded left landing11. Support4 plus pelvis1/3 fails joints876.44/869.24.

`audit_support_motion.py` includes native sphere radii and reports source-contact
foot motion. Peak speeds at contact onsets include impact transitions; they
are not continuous-support no-slip certificates. The extended anchor's right
reference minimum penetrates7.27mm near829. The whole-turn refined reference
penetrates19.34mm near835. Fixed-root-attitude, fixed-COM clearance correction
in `prepare_stance10_clearance.py` passes bounded IK: maximum lift7.27mm from822
or8.30mm from810, raw-source joint peak0.17390, foot residual below0.5µm.
The respective q/v/a prefixes remain exact through816/804. Smooth onset leaves
minimum penetration4.34/2.86mm. Runtime822/root1000 rejectsQP880.04;
822/support1 stops880.76 orientation;810/support1 rejectsQP879.32. No promotion.

A COM diagnostic on anchor/support1 finds desired vertical acceleration17.32
at836; predicted13.27 and observed13.28m/s² agree closely. At842.7 the desired
value is9.08. Existing minimum-support-acceleration is a lower clamp and does
not suppress these upward targets. A separate default-disabled maximum-support
acceleration policy now caps only the soft vertical target, with C2 onset.
It retains horizontal/downward targets, the metric, physical constraints,
source validation and single QP. Two focused tests and16 profile tests pass.
Trials cap0/3 from816 or cap3 from802; current default remains verified10.


COMcap0/3 from816 reaches10/866.36 or878.60; cap3 from802 reaches863.48.
Cap3 retains279N right support at878.60, but orientation0.35129 exceeds the
unchanged bound. Pelvis0.3/1 from816 or1 from836 then fails joints near864.
The dominant added errors in these trials include left knee/hip, not only arms:
cap3 knee extends from0.949 at842 to−0.032 at866 while prepared knee remains
0.796. Root XY lag leaves the swing foot behind the target as the leg extends.

The three subsequent `arms*_metric2` cases are NOT valid late-only arm
comparisons: overriding global-posture-feedback-start to4.08 removed the
profile's existing all-joint feedback from0.70. All diverge before the intended
activation, failing7/584.6. Resolved `controller_options` confirms the baseline
already uses all-joint feedback cap150; the nominal joint-feedback-limit30
is overwritten by that existing policy. Do not interpret those trials as
isolating arm weighting and do not promote them. No default was changed.

Measured baseline root Y error grows from+3mm at500 to−87mm at650, before
previous source660 corrections. New experiments start original-source XY
feedback at500 with a0.5 source-second ramp and weights30/100/300, retaining
all original profile feedback. This changes earlier walking intentionally.


Early XY30/100/300 trials reach9/832.93,829.46,823.70, failing joints. Step10
native unloaded peak18.72/14.83/15.64mm does not meet20mm. Compared at650/750,
Y tracking improves somewhat while forward X lag increases. New optional Y-only
root-task axes preserve X/Z natural-form costs at zero; an invariance test and
all17 full-tracking tests pass. Current trials use Y100/300/1000 from500.

`late_swing_reach_audit.json` uses two bounded IK seeds at recorded roots in
COMcap3. At877.88, best local foot position residual125.85mm; shifting root
forward/left/down by50mm each reduces it to48.93mm. These offline hypothetical
shifts are diagnostic only, never simulator resets, and local IK is not a
global infeasibility proof. At866 the measured left knee is near extension.
This supports working on root drift and height before increasing swing gains.


Y100 from500 reaches10/880.04 before a freeflight QP failure; last recorded
jointRMSE0.17168, root error[-0.14383,-0.06946,0.04756]m. Y300 reaches8/858.44,
but step9 loaded hold only16ms. Y1000 reaches9/848.37, with step10 peak flight
18.92mm and a QP failure during dual contact. All12 frozen numerical settings
remain invalid for that failure, so no runtime retry is introduced. Y1000
root Y error is−34mm at650,0mm at750,+10mm at815, but this alone does not imply
successful tracking. Current trials pair Y100 with the COMcap3 policy and
Y1000 with each of the two audited whole-turn source-joint fits.
`com_upper_cap_prefix_verification.json` confirms166 history fields match
through815.96/1903 updates before the earlier cap3 experiment activates.


Y100+COMcap3 reaches10/884.36: joints0.20308, position0.17057m,
orientation0.14401. Step11 remains unloaded and both feet are airborne at the
stop. Y1000 plus first/refined whole-turn references reaches8/791.48 or871.40;
no new verified landing. Furthest source progress is not a landing count.

COM vertical feedback retains the original4m/s² saturation because the
existing enhanced COM feedback selects XY. Testing existing XYZ mode from640
with gain1.25/cap20 reaches10/880.04 on baseline but regresses Y100 branches
to9/834.14, with or without the late uppercap. These remain unpromoted.
`late_vertical_target_audit.json` records large reference vertical accelerations
(up to26.63, down to−22.21m/s² in the compared references) and the asymmetric
lower clamp. This is a reference/control consistency issue, not proof that a
single task alone causes loss. Current trials retain Y100 and use uppercap0/1
from816, including a variant with the lower clamp changed from−3 to−6.


Y100 uppercap0/1 reaches10/885.08, joints0.20069/0.20242, position0.17114m,
orientation0.18262/0.17018. Lower clamp−6 plus uppercap0 rejectsQP819.32.
No11. First shoulder4/16 runs were inactive: shoulder start uses control time,
and4.08 exceeds their4.018 stop. The histories are identical, not evidence
that active shoulder weighting is ineffective. First late-tube100 used3.806
SOURCE seconds, which changed walking before10 and reaches9/837.31. Corrected
trials use shoulder controlstart3.806/ramp0.03, and tube source-start4.08.

Prepared, not yet run: existing causal landing-pause11 clock ending at884,
starting802/816/840. Full-source completion remains required. Nominal RNEA
screening over816..883 shows these clocks reduce negative-normal-force sample
counts28 to6/9/22, but still have infeasible acceleration samples. See
`pause11_nominal_force_audit.json`; no feasibility claim or runtime MPC.


Correctly timed shoulder4/16 stops atQP884.36/878.60; late-tube100 from816
stops atQP879.32, all10 landings. Frozen shoulder4 cap3/prox1e-4 validates,
so a new run preselects cap3 after3.806; no extra runtime attempt is allowed.
The recent summary audit passes all saved completed trials' single-attempt,
no-secondary-QP, exact applied torque and original fidelity invariants.

Pause11 from802/816 rejectsQP at883.78/883.97; from840 stops at884 on joints.
Step11 remains unloaded in all cases. The816 case retains368N right load at
control4.168, but the left foot stays high. Paused progress is not full tracking.

The optional COM-speed policy caps the soft vertical target at
max(-6, bandwidth*(upward_speed_limit-measured_COM_vz)), blended from the
configured source time. It permits positive braking during descent, unlike a
constant zero upward-acceleration cap. It changes no QP physical bounds or
source acceptance and adds no optimizer. Default disabled; focused tests now3,
profile tests17. Trials use speed limits0.2/0.4m/s, bandwidth10 from816 on
Y100 without the fixed uppercap. No default reference/entry promotion.


COM-speed0.2/0.4 reaches10/885.80, joints0.20048/0.20264, position0.17190/
0.17160m, orientation0.20534/0.18840. Neither has a loaded11. Preselected
shoulder4/cap3 instead rejectsQP839.96. The fixed-state numerical success did
not predict its full closed-loop trajectory.

An optional late root-cost transition now convexly blends the SAME root
position task's axis metric; pose targets, QP count and constraints remain
unchanged. Disabled by default. Natural-form convex-combination test passes;
full-tracking tests now18. Starting at816/ramp20frames on COM-speed0.2,
Y300/XY300/XY1000 stops10/885.80,885.80,885.08 on joints. No promotion.

At885.8 the COM-speed0.2 waist pitch first reaches−0.520000rad with stationary
velocity; earlier at885.08 it is−0.50378 at−10.79rad/s. An observed-stop mode
cannot fix fidelity already exceeded at first impact. Current trials use the
existing soft braking rows for waist pitch/right shoulder roll (weight1000)
or existing zero-angular-momentum feedback (damping10, rate cap30, metric0.1/1)
from816. No additional WBC or MPC was added.


Zero-momentum metric0.1/1 reaches10/885.80 or886.52 (latter joint0.20019,
position0.17562m, orientation0.16865); named soft braking1000 rejectsQP882.20.
No11. `late_loaded_contact_audit.json` shows only ONE loaded right sole point
at3.8/3.85/3.9/3.94 despite7/7/8/9 reported contacts. Most other points are
separated by several millimetres to2cm; counts alone do not establish a support
polygon. No contact forces or rotational constraints were invented. Current
trials activate the existing soft flattening target from816 with gain3,
feedback cap100, explicitly promoted support orientation (tilt-only weights4/16,
or full orientation16). All original physical contact geometry remains measured.


Flatten tilt4/16/full16 stops10/877.16,875.00,877.16 on joint fidelity; no11.
On Y100/COM-speed0.2, fourth speeds1.2/1.4/1.6 reach10/881.72,891.16,891.32.
Clock1.4 still carries309N right load with root angular velocity about
[3.04,−0.64,1.69]rad/s at its stop, compared with airborne1.6/1.8 branches.
Its dominant joint errors are right shoulder roll−0.748, waist pitch−0.550,
waist roll+0.438rad. Left foot link remains126.7mm high; no loaded landing11.
Clock1.2 retains441N right load but also fails joints. Slower is not automatically
better; all are full-source attempts under the unchanged original-source clock audit.

`fourth_speed_nominal_force_audit.json` reports negative-normal-force samples
in816..883:1/7/22/28 for1.2/1.4/1.6/1.8. This is only a nominal reference
screen, not whole-body feasibility. Current clock1.4 trials increase right
shoulder roll weight4/16 aftercontrol3.806; a third also models the measured
stationary waist-pitch stop. The stop is first observed near890.04 with
jointRMSE0.1975, before that branch's final threshold crossing. No extra torque,
force, solve, or simulator state reset is introduced.


Clock1.4 right-shoulder scale4 fails QP883.32; scale16 reaches10/894.52
(joint0.20113, position0.18558m, orientation0.25296rad at stop). Adding
observed waist-pitch stop changes almost nothing. Pelvis replacement1/4 rejects
QP883.32/882.20, extendedanchor rejects886.68. LateXY300 stops892.84joints;
XY1000 from780 rejects882.76; waist16 fromoriginal740 regresses869.32.

Frozen failures from pelvis1/pelvis4/anchor all certify under the SAME original
QP with50000 internal iterations. Added default-off `--qp-late-iteration-start`
(control seconds) and `--qp-late-iterations`; no retry or relaxed certificate.
Closed-loop50k from4.02: pelvis1/pelvis4 still reject near883.88/883.32; anchor
reaches10/895.64 then joint0.200324, rootposition0.187044, orientation0.263994.
`clock14_iteration_budget_prefix_audit.json` verifies167 historyfields/2010
updates exactly unchanged throughcontrol4.018, one native attempt, zero addedPD,
and exact appliedQP torque. Hard-frame QPwalltimes0.35..0.44s preclude a500Hz
real-time claim; this is offline diagnostic progress only. SingleQPtests6 and
profiletests18 pass. Default entry remains the independently verified10baseline.

On anchor+shoulder16, clocks1.0/1.2 stop10/877.00joints or883.64QP.
Pelvis4+cap10fromcontrol4.02+50k reaches892.28joints. Still no11.
Atanchor895.64 the largest errors are waistpitch−0.5555, waistroll+0.4768,
rightknee−0.3495,leftknee+0.3415. Leftfootlink is190.1mm aboveground.
At880 the waistpitch task opposes the measured downward motion but the QP
acceleration remains negative: weighted task conflict persists before the stop.
Added default-off late waist metric replacement (source seconds) to test16/64
from816 without changing the already-verified prefix or desired accelerations.


Latewaist16/64 from816 regresses to10/862.04,857.56; withoutanchor16 reaches
864.84. Late sourceXYZ1000/3000 onanchor stops892.84/869.32joints; without
anchor1000 rejects883.88. No11. Frozen swing-angular weight0.01..0.3 changes
leftfoot vertical acceleration atcontrol4.0 only−14.26→−14.57m/s², far from
requested−68; no runtime angular-weight feature was added on this evidence.

`prepare_late_waist_source.py` restores half/full of source waist angles from
822/rise882/hold940/fade1000 while preserving COM and both feet. Fixed-root
fits fail preflight at906..907; allowing the existing bounded root-attitude IK
passes with additionalrotation0.03307/0.07102rad, footerrors<0.7µm, COM<0.3µm,
rawjointpeak0.16656/0.16118. q/v/a prefix through816 is exact. Closedloop on
anchorclock14+sh16+50k reaches10/895.64 or892.28 (joint0.2040/0.2043), no11.
Namedwaistroll+pitch softbraking1000 from816 stops880.52joints.

Current contact friction is mu0.6. Atanchorcontrol4.0 rightwrench is
[107.743,107.743,253.953,0,0,0], at circular friction limit. Existing optional
16-ray inner cone was tested (no mu increase, no invented support moments).
Anchor from816 rejects883.88; from780 stops883.32orientation0.35512; noanchor
from816 rejects885.00. Tests independently maximize tangential capacity in65
directions using LP and verify every direction stays inside the original
Coulomb circle; a point contact has zero independent moment capacity.2pass.
Helper source saved separately as clock14_inscribed_patch_wrench_source.py/json
and added to future trial archives. No default promotion or acceptance changes.

Next experiments weight existing leg posture4/16/64 from816. The optional
question about allowing multiple internal priority QPs within ONE WBC is pending;
all executed trials still retain the strict one-native-attempt restriction.


User explicitly reaffirmed: strictly one QP solve per control cycle; no internal
multi-stage QPs. This remains the runtime constraint; no MPC, addedPD or force.
Legposture4/16/64 from816 onanchorclock14 stops10/892.28joints,883.32QP,
887.80joints. Flatstance candidate preserves leftfoot+COM+rightfootXY with
additionalrootrotation0.01740rad andrightfootheightchange11.09mm. Preflight
feet<0.6µm,COM<0.3µm,rawjointpeak0.17292; runtimeqva816exact. Closedloop
base/supporttilt1/supporttilt4 stops10/895.64,883.88QP,893.40joints. No11.
AdditionalCOMdrop40 with free bounded root rotation fails geometric preflight
andrawjoint0.2089; no candidate exported or runtime trial launched.

The best anchor branch jointtube slack rises to1074 at890.04 and1246 at895.08.
Current tests raise only this existing soft cost to100/1000/10000 fromsource816;
previous10 remains active earlier. Source PKL SHA256 reconfirmed unchanged.


Anchorjointtube100/1000/10000 from816 stops10/824.63QP,897.32joints,886.12QP.
At897.32 leftfootlink279.8mmhigh; no11. Caps15/30/45 onleftlinearfeedback
stop884.44joints,862.60QP,883.32QP. Zero-H damping10/ratecap30/metrics1,3,10
stop894.52,895.64,891.72joints. Currentbestframes897.32still10/29, notfulltracking.

`clock14_active_constraints.json`: atanchorcontrol4.0, point-contact friction
andallthree auxiliarytrackingrowsbind; noeffort/acceleration bound active.
`clock14_auxiliary_barrier_sensitivity.json` removes auxiliaryrows OFFLINE only.
Removingpositionbarrier changes waistrollacc+79.76→−34.15, pitch−30.73→+22.99,
butrootlinearacc becomes[−16.83,−13.94,−22.09]. This identifies a conflict,
not a safe runtime fix. Existing latepositionrates4/20 stop885.56/897.32joints;
alltube rate20 stops895.08joints. Original error radii unchanged throughout.
Native eta/primal/dual numerical-tolerance probes all stillfail original KKT
checks; no runtime tolerance changes were made. `clock14_summary_audit.log`
completed across savedruns throughflatref: singleattempt,noextraQP/noPD,
exactQPtorque and original acceptance invariants pass.

Earlier `six_source_rooty1000_500` has9landings, rooterrorpeak0.08688m;
step10nativeflight18.915mm misses20mm while best loadederror28.14mm passes.
Step9landing775.16/control3.634; originalfailure849.083/control3.918.
New `prepare_rooty_clearance10.py` keepsroot,leftleg,upperbodyexact andadds
rightfoot5/10mm over762..802. BothIKpasses(<0.5µm feet), rawjointpeak0.14258,
qva prefix756exact. A late5mm798..826 version failsIK; noexport/no runtime.
Current batch tests passedlift5/lift10 andunmodifiedcontrol onearlyY1000,
allwithclock1.4,COMspeed.2from816,50kbudgetfromcontrol3.806.


StrongY1000from500 +clock1.4+COMspeed.2+50kfromcontrol3.806: control still9/
850.84joints andstep10flight18.915mm; lift5 reaches9/860.92 withflight19.653mm;
**lift10 reaches10/917.48** (joint0.20242,pos0.17280,ori0.31313). Step10landing
814.04/control3.796, sourceerror34.09mm, load810.63N. Step11nativeflight51.71mm
isverified, butno loadedlanding. This is progress, still not11orfulltracking.
At916.92 rooterror[−.16237,+.04144,−.04106]m, omega[.729,1.905,−1.935];
leftfootlink67.45mmhigh andrightload239.89N. Dominanterrorsnowrightknee−.666,
righthippitch+.489,rightanklepitch−.435,leftanklepitch+.321,leftshoulderpitch+.281.
The reference beginsrightleg's next swing beforeleftlanded11. Currentcausal
pause11 tests start840/end884,start840/end900,start860/end900,resume.3;
clockresumesonlyafterindependent11landingcertificate. A pausedphase isnot
completion. StillstrictoneQP,noMPC/noextraPD/force,defaultentryunchanged.


**Observed11 strict single-QP landings:** `rooty1000_lift10_pause11_860_900`.
Pause start840/end884 rejectsQP875.574; start840/end900 rejects898.390.
Start860/end900 reaches11: leftlandingat source900/control4.344, error0.038171m,
load578.039N. Rightfoot12flight0.050087mverified, butno12landing. Stop4.474/
908.9686: joint0.200839, rootposition0.152622m, orientation0.286462rad.
RootXerrorchanges−.04876 atpaused900/control4.258 to+.12090at4.474; rightknee
error−.515,rightanklepitch+.404, shoulders−.36/−.34, waistroll−.356.
`eleven_rooty1000_invariants.json` verifiesoneDAQPattempt/update,noextraQP/PD,
exactappliedtorqueandoriginalfidelitythresholds. Independentreplicationpending.
Nextbatch repeatsidentically andcomparesresume.1/.15(sourceclockseconds),
which canonlytakeeffectafter11landinghasbeenindependentlycertified.
No defaultentrypromotionyet; 11ispartial, full29/2000stillnotachieved.


11-step result reproduced exactly across167 historyfields (`eleven_verify_rooty1000_pause`).
`eleven_resume10` and`eleven_resume15` retain11 andadvance939.84/936.52.
Resume.1 originaljoint0.202020,position0.165744,orientation0.289946; step12
flight58.69mmverified,no12load. Firstchangedcontroltime4.348; all167prefix
fields identical beforethe11certificate. At939.84 rooterror[+.08018,+.14346,
+.02147]m, leftload104.08N. Largesterrorsrightknee−.541,rightanklepitch+.420,
leftshoulderpitch−.398,righthippitch+.375,waistroll−.329. Current trials
rootY3000/XY1000 orlegposture16 activateonlyafter source900 onresume.1.
Flatleftlanding11 alternativesxy100/xy50 failIK/rawpreflight andwerenotexported.
Default10entry remains available; currentbestexperimental chain is11, notfull.

### 第 12 步：晚期上身权重与跨髋碰撞审计（2026-11-24）

继续严格每个控制周期一次原生 DAQP 求解。`eleven_leg16_lateupper64` 将既有上身姿态权重从 16 平滑过渡至 64（原始源时间 4.5 s 之后），保留先前从 3.7 s 开始的权重配置；11 次合格落地、最远源帧 952.72/2000，仍未完成第 12 步。原始关节 RMSE 上限触发停止。`eleven_latest_audit.json` 验证前 11 步全部历史字段前缀一致、单次原生求解、实际施加力矩逐项等于 QP 力矩、原始验收阈值不变。晚期权重 256 和 COM 上升速度软限制退场均未改善。

直接覆盖 `--upper-posture-metric-start` 会撤掉早期已验证配置：`eleven_leg16_upper4/16/64` 均只得到 8 步，不能当作保留前缀的晚期实验。新增默认关闭的 `--late-upper-metric-*` 用于仅替换后段权重。`--com-upward-speed-end` 默认无限，未启用时保留原行为。

`step12_hip_collision.json` 对照原生接触报告与独立凸网格距离：`eleven_leg16_lateupper64` 在控制时间 4.552 s 发生左右 `hip_yaw_link` 接触。4.55 s 实际间隙约 0.33 mm，而原始源姿态间隙约 133.6 mm，控制参考约 138.4 mm。该对原先未纳入精确网格碰撞约束。新增默认关闭的 `--hip-yaw-barrier-start`，启用后将这一对的距离加速度屏障追加到原有 QP，不引入第二次优化、附加力矩或外力。使用现有状态的中心差分 Jacobian 导数。28 项物理任务测试通过，包括保持已有任务不变、追加单个约束、屏障方向以及重复调用不累积行。

等待位置 894 帧的实验仍为 11 步/951.2 帧；898、902 帧在第 11 步之前 QP 验证失败。完整 tracking 仍未完成，默认 10 步入口保持不变，11 步独立入口仍可复现。

第 12 步继续验证：跨髋屏障带宽 8/12/20 均保留 11 步；带宽20达到948.8帧且没有再次报告跨髋接触。晚期支撑载荷下限200/300/400 N分别退至945.44/941.52/938.72帧，未采用。脚端权重64/256退步，晚期腿部权重64仍为948.8帧。新增默认关闭的第五次平滑时钟过渡，复用已有解析导数函数；4项时钟测试通过。从控制时间4.42s开始，在0.12s内变速：0.8/1.0倍分别941.92/944.72帧，1.8倍推进至954.16帧，仍仅11步。`eleven_hip_leg64_speed18.strict_audit.json`确认前11步历史前缀完全一致，单次原生QP与实际力矩一致性通过。

后续第12步实验：`eleven_speed24_tube1000` 将原有辅助关节跟踪软约束的权重在源时间4.5s之后提升至1000；11步，963.04帧，关节RMSE峰值0.19121，根位置峰值0.17848m，最终根姿态0.35267rad超过原始0.35rad上限而停止。与原11步版本到4.344s的历史前缀完全一致。没有放宽独立PKL验收。

`step12_scaling_replay.json` 对失败状态进行离线单次DAQP重放：缩放cap2可使部分高权重QP通过相同KKT验证，姿态半径0.32的失败状态仍不通过。新增默认关闭的 `--qp-final-equilibration-start/cap` 在一次求解之前选择缩放，未增加重试。闭环cap2的tube1000仍963.04帧，tube10000达到960.16帧，pelvis4更早失败；未宣称数值调整已解决tracking。3项DAQP缩放测试通过。

`prepare_swing12_offset.py` 的固定根/左腿/上身离线IK中，从906帧开始的侧移候选触及右踝roll上限、未通过预检，因此没有导出或运行。将紧支撑平滑修正移至918..966帧，右脚最多外移20mm、下移20mm，通过预检（位置残差0.26µm，最大原始关节RMSE0.14258rad，900帧以前q/v/a逐项不变）；其保留原参考既有的后段脚底穿地，没有宣称完整几何可行。闭环 `eleven_tube1000_swing12` 为11步/962.08帧，未超过原配置。

注意源时间停在4.5s时尚在等待第11步认证：直接使用 `>=4.5` 的姿态半径切换和物理制动，会影响等待阶段。对应 `*.strict_audit.json` 如实记录前缀差异；它们不是严格保留前11步的实验，也没有推广。平滑权重在边界的blend为0，新增跨髋屏障使用严格大于，数值缩放使用控制时间4.348s，已分别审计。

`step11_support_transfer_geometry.json` 记录第11步交接：4.28s仍由右脚承重，4.30s左脚承重743N、右脚已无载荷，4.344s左脚只剩一个承重接触点。已有保留支撑任务weight1/10/100三组全轨迹结果相同，均11步/963.04帧；进一步查明该任务要求原始计划为双支撑，而900帧的原始计划已是单支撑；实际门控后的控制计划仍为双支撑。因此原任务没有启用。已改为读取门控后的控制计划，待闭环验证。继续检查更早的横向根任务和支撑脚压平。

### 第12步时钟、接触与力矩变化（2026-11-24）

新增默认关闭的 `--extra-contact-pause START END LANDINGS`，可串联既有落地暂停；只读取已验证落地数量，使用解析链式导数，不做预测优化。3项测试覆盖保持源时间单调、逐个落地证书释放和导数。第12步名义落地952帧、下一次抬脚955帧，当前2.4倍速度下仅6.25ms，短于50ms落地保持要求。单独在920/930/940帧开始减速、停于952帧的三组仍为11步，右脚未承重、关节误差先超限。相关18项完整tracking和6项单次QP测试通过。

`step12_acceleration_response.json` 对照逐周期实测速度差分与QP预测，主要偏差集中在支撑脚踝。启用已有球形接触滚动修正（控制4.296或4.348s开始）未改善11步/963.04帧结果。

原先保留支撑任务已改用 `controller_contact_plan`。weight100时4.30s右脚实际仍有86.46N，但仍为11步/958.24帧；把目标提高到250N在第11步认证前QP失败，未采用。更早加强根Y到10000（源4.1s）可使第11步提前至控制4.268s，但后续关节误差在953.01帧超限；其tube3000变体958.01帧，仍11步。

力矩连续性实验前修复两处兼容：晚期tracking tube权重直接写入底层tube solver，避免与连续性代价的weight混淆；连续性代价允许QP尾部的被动机械限位反力变量，仅惩罚47:76的电机力矩，保留物理约束与被动反力。9项连续性测试通过。零权重 `eleven_continuity0` 与 `eleven_speed24_tube1000` 的168个完整history字段逐项一致（排除墙钟求解耗时），见 `eleven_continuity_zero_full_audit.json`。

注意 `--torque-continuity-start` 是**源时间**。这次设置4.348s对应源869.6帧、控制约4.006s启用，实际改善的是第11步交接，并非只改变第11步之后。weight100为11步/965.28帧；weight1000为11步/985.17帧，第11步提前到控制4.248s，根位置峰值0.16334m、姿态0.30509rad，最终关节RMSE0.20155rad超限。第12步右脚尚无承重落地证据，不能计作12步。后续分别验证额外第12步暂停、提高连续性权重、以及统一参考逐渐回到原始源姿态。

离线 `prepare_return_source.py` 从906到966帧把统一控制参考平滑恢复至原始源姿态。900帧以前q/v/a精确不变；逐帧源关节偏差不增加、关节限位误差仅浮点舍入。此区间旧准备参考右脚原生几何穿地约15mm，而原始源仅约0.17mm；恢复候选约0.21mm。该候选不是完整动力学可行性证书，必须通过闭环原始验收，原PKL未修改。

Further single-QP step-12 diagnostics (2026-11-24): `eleven_cont1000_speed10`
reaches source 1015.89 but remains 11/29; increasing the optional right-foot
feedback cap to 150 reaches 1017.89, still with no loaded step-12 landing.
Slowing and pausing at source 952 fails the original orientation limit
(0.350233 rad); whole-body return-to-source with final equilibration cap 2
still rejects the native QP at source 936. No acceptance thresholds changed.
The first three balance trials preserve all audited history fields through
control 4.0 and execute exactly one native QP per update with exact applied
QP torque. More source frames do not imply another qualified step.

`prepare_swing12_source.py` creates a bounded offline right-leg IK candidate,
blending the right foot toward the original world pose over source 906–930,
then restoring the parent suffix over 1010–1040. Root, left leg and upper body
remain exact; sampled q/v/a through frame 900 are identical. Maximum position
and rotation residuals are 0.518 micrometres and 0.301 microradians, joint-limit
violation is float32 rounding (2.98e-8 rad). Minimum native right-foot clearance
improves from -15.040 mm to -0.173 mm. This is a geometric preflight, not a
closed-loop certificate; three runtime variants are under evaluation.

An optional `--torque-continuity-end` source-time transition can reduce the
continuity cost after transfer, using `--torque-continuity-end-ramp` (default
0.06 source seconds) and `--torque-continuity-end-weight` (default zero).
The default end is infinity, preserving existing behavior. It changes the
objective of the same single QP, never filters or adds applied torque.

The `speed10_foot150` option experiment also replaces the inherited left-foot
feedback override from source 4.08, so it is not a prefix-preserving late-only
ablation. Its strict audit records those differences; it is not promoted.

The right-foot source-pose candidate passes geometric preflight but all three
closed-loop variants remain 11/29 (source 988.05, 1017.89 and paused 952).
Continuity decay 1000→100 rejects a native QP at source 915.41; decay to zero
rejects at 938.37, while combining decay with the right-foot pose candidate
reaches 1014.29 and fails the original joint RMSE bound. None is promoted.

Frozen-state sensitivity (`continuity_objective_sensitivity.json`) shows little
COM-Y improvement from fourfold COM/foot weights. The independent offline LP
bound (`lateral_feasibility.json`) at control 4.5 gives minimum COM-Y acceleration
-13.1265 m/s² versus requested -18.1245 m/s² under the recorded constraints.
At 4.3/4.4 the scalar request is feasible only with other motions unconstrained
by task fidelity; extreme LP solutions have huge angular accelerations and
are not viable tracking commands. These LPs are diagnostics only and never
run inside the controller. Removed-constraint diagnostic variants are not
runtime settings. Work shifts to reducing lateral root-reference offset
before the step-11 transfer while preserving both world foot poses.

Direct history inspection of `eleven_cont1000_speed10` localizes the reversal:
at control 4.0, actual/source/prepared root Y is -0.86884/-0.87720/-0.82665 m
and actual Y velocity is -0.56093 m/s; by 4.2, Y velocity is +0.11647 m/s
while the prepared root remains about 53.5 mm to positive Y of the source.
This motivates a coherent reference ablation rather than further gain sweeps.
`prepare_transfer11_rooty.py` removes 25/50/100% of that lateral offset with a
source 840–900 quintic blend and fades the edit over 960–990. Bounded two-leg
IK preserves both world foot poses; max position residual <0.7 micrometres,
root X/Z/orientation and upper joints are exact, q/v/a prefix through834 exact.
The original unwindowed half-offset candidate failed preflight near straight
right-leg frame1008 (0.746 mm residual) and was not exported or simulated.

Root-Y-only quarter/full runs regress to 10/29: step-11 best valid loaded
hold is respectively 48 ms and 26 ms, below the unchanged 50 ms requirement.
The half run aborted at a real software edge case: a quintic margin blend
rounded above one at a paused source endpoint. The caller now clamps this
mathematically unit-interval blend before allocation; no acceptance check or
physical constraint is altered. The original half configuration is rerun.

Prepared left-foot Y is already roughly -32 mm relative to original source
at frames880–950. Actual tracking therefore runs close to the 40 mm landing
boundary (also ~-22 mm X), explaining the fragile step-11 certificate. New
bounded-IK combinations add +10 mm X/+20 mm Y to this left-foot target over
840–880, alongside half/full root-Y correction, and fade edits960–990.
These offset candidates pass the same submicrometre IK and joint-limit
preflight; original-source landing acceptance remains untouched.

The margin-blend fix resolves the half-root-Y runtime exception: its unchanged
configuration now completes to source961.18, with11 landings and a joint-RMSE
stop. Root-Y plus left-foot XY correction reduces step-11 landing errors to
18.762/19.586 mm (half/full), but half exceeds joint RMSE immediately afterward;
full reaches950.09 with about111 mm root-X lag. An earlier fifth speed transition
at control4.24 gives full/speed1.0 source966.16; full/speed0.8 and half/speed0.8
reject QPs at904/934. None passes step12. All six completed transfer variants
audit as one native QP per update, exact applied torque and unchanged source
acceptance, with history prefix throughcontrol3.85 preserved.

Late foot metric64/128 and pelvis replacement1 on the continuity1000/speed1.0
branch remain11/29 (source1017.49/1017.09/1019.09). Native single-QP and exact
applied torque checks pass, with the original tolerance values unchanged.
The 1019.09 maximum is source-clock progress, not a twelfth landing.

At control4.248 the paused COM reference velocity is zero while measured COM
velocity is approximately [0.95,0.18,0.16] m/s. During resume, reference X
acceleration reaches24.84 m/s² at4.29 and upward velocity0.64 m/s at4.33.
Current tests lengthen resume to0.2/0.3 s (previous0.1) or correct prepared
root Z aftersource888. `prepare_transfer11_rootz.py` keeps both foot poses and
returns height toward originalsource over888–912, fading960–990. Max root-Z
change18.84 mm, maxIK residual0.51 micrometres, q/v/a prefix882 exact. The
version starting840 failed bounded IK near879 by4.94 mm and was not exported.

Resume0.2/0.3 and late root-Z correction remain11/29 (source937.17 QP rejection,
973.49 joint RMSE stop,999.63 joint RMSE stop). Coherent source return over840–900/930
also remains11 (959.68 joint error/956.28 orientation error). Root-Z plus
step12 pause remains at952 and fails joint fidelity. These are not promotions.

A task-selection gap is now isolated: on the continuity1000 baseline at
control4.4, `support_orientation_0` has active local pitch but its composed
metric is zero. `fabric_secondary_scale=0`, `landing_orientation_priority=False`
and `support_orientation_priority_start=100` omit the free support-foot
rotation from the effective objective. Stance feet_pose rows are masked, so
raising ordinary swing-foot weights does not fix this. Offline native-QP
replay adding the free pitch cost reduces pitch acceleration residual from
76.96 to17.28 rad/s² at scale64 and5.19 at256. Physical contact rows and wrench
cones are unchanged. Three new trials activate the existing support tilt
selection smoothly after source900, with/without native sole flattening;
all still compose task costs into one inverse-dynamics QP.

Support tilt64, native-flat64/256 remain11/29 (source1004.29/1016.69/1011.09).
A second gating mismatch was verified in logs: at source957 the actual gated
support plan remains left(1), but raw plan selects right and the left flatten
weight drops from1 to0. The optional flatten gate now uses
`controller_contact_plan`, still requiring measured load and a real free
contact tilt. Its default is disabled. Gate-fixed flat64 remains11/source1014.29;
flat64+pause12 and source-return60+flat64+pause12 both holdsource952 and stop
on original orientation/joint fidelity. No twelfth loaded landing yet.

The source-return60 paused branch gets the right native sole within13 mm of
ground aroundcontrol4.65, but foot-link error remains[-66,+25,+14] mm and load
iszero. At4.55 its task requests right-foot X/Z accelerations[19.82,-7.14]
m/s² while the QP predicts[-2.90,+1.23]. Frozen objective replay with unchanged
constraints (`step12_swing_weight_sensitivity.json`) shows16x/64x/256x foot
weights reduce X/Z residuals to[-5.95,4.29]/[-1.77,1.73]/[-0.47,0.51] m/s²,
all certified in one native attempt. Current trials test those effective
weights through existing late foot-weight options, not another QP stage.

Foot256 rejects QP atsource912; foot1024/4096 track right-foot XY to roughly
[-11,+10]/[-10,+11] mm but stop atsource950/944 on full-body jointRMSE before
load. Waist/shoulder errors dominate. Upper256 with early foot weights rejects
QPs around930. Delaying foot4096 tosource930 with upper256 stays untilcontrol
4.936 atpaused952; right-foot error near4.7 is[-16.8,+6.1,+3.6] mm, loadzero.

Exact collision FK identifies a reference/contact mismatch: atpaused952,
stride6 control spline puts the native sole4.687 mm aboveground, while the
original-source audit spline puts it0.224 mm aboveground. Measured rightsole
converges toabout5 mm clearance. Control native clearance byphase is
954:2.271 mm,956:0.805 mm,958:-0.040 mm,960:-0.257 mm. Current trials move the
extra pause end to958/960 or transition only the control spline tostride2
from930 (20-frame ramp). Source acceptance and foot-load certificate are
unchanged. This explains this branch's hover; it is not a full-tracking claim.

Pause958/960 and late control stride2 remain11/29. At pause960/control4.7,
actual/native sole clearance is1.139 mm, control target -0.257 mm, source
-0.164 mm; right-foot link position is already within3 mm of original. The
control-spline phase mismatch is real, but finite tracking error still leaves
the foot unloaded. `audit_step12_pause_geometry.py` reproduces the geometry.

A default-off bounded touchdown search is added in `g1_contact_search.py`.
`--foot-contact-search-start` is SOURCE seconds; default infinity disables it.
Only a raw-source planned contact with an active free-foot position task,
measured load<20 N, XY goal gap<=40 mm and vertical gap in[-5,+20] mm receives
an extra downward Cartesian acceleration. It is blended over0.05 source s,
bounded to<=10 m/s², and exits when contact masks the task or load reaches20 N.
It changes only the existing single-QP foot request, never applied torque
outside that QP; the source pose, physical constraints and independent
flight/100N/50ms landing criteria remain unchanged. Current trials use0.5/1/2
m/s² aftersource940 on the paused960 branch. Four gate/exit/bounds tests plus
18 full-profile and6 single-QP tests pass. Runtime certification is pending.

### Reproduced twelfth landing, still incomplete

`eleven_pause960_search2` and `twelve_search2_verify` reproduce 12/29
original-source landings, with all 171 compared history fields and applied
servo torque equal. Original source SHA remains
`919eaff9b2521cce1e8b40e263fceb3f3f1873a7bc42500d9932be62f007eca9`.
The optional entry `examples/run_amp_single_tracking_12step.sh` has the same
resolved controller options as that trial (excluding output and snapshots).
Default and eleven-step entries remain unchanged.

The existing online continuous counter requires 20 N bearing for 50 ms;
it records step12 at control4.680/source959.998667, foot error8.103 mm.
The independent stronger audit requires at least100 N at every sample for
50 ms elapsed:4.638–4.688, minimum110.918 N, maximum original-source foot
error8.401 mm, verified unloaded native clearance34.274 mm. These are distinct
certificates; the online threshold was not changed. Through the strict
certificate jointRMSE<=.196433, root error<=.188930 m and root angle<=.327707
rad. `audit_twelfth_landing.py` independently computes these from FK/history.

Both runs stop at4.804/source972.49185 on original jointRMSE=.203768 rad;
full tracking is false. Source coverage is not the same as completed steps:
the older11-step trial reached source1019.09 without a twelfth landing.

Reducing torque continuity from1000 to100/0 after source960 preserves12 but
still stops at972.49. Reducing foot multiplier4096 to16 after960 has exactly
unchanged trajectory through this stop because both feet are then in support.
Waist pitch/roll and shoulder errors rise rapidly during reference-clock
resumption; current trials test waist metric16, arm feedback and pelvis
orientation after source960. Each online update still attempts one native QP.

Step13 frozen-state diagnosis (`step13_auxiliary_tube_diagnosis.json`):
at control4.740 the root-position and root-orientation auxiliary tube rows
are both active. The existing QP predicts waist roll/pitch accelerations
-71.56/-88.16 despite desired +68.01/+82.19. Removing only the position tube
**offline**, with physical constraints retained, changes them to+55.33/+66.68;
it violates that position tube by283.99 normalized units. This isolates the
root-lag/posture conflict; it does not justify reporting a relaxed run as
success. Runtime source acceptance remains unchanged.

Waist16 and pelvis1 activated aftersource960 retain12 but do not advance13.
The arm-feedback trial accidentally replaced the profile's feedback onset
0.7 with4.8, so its7-step result is not a preserved-prefix late intervention
and is excluded from comparison. Sixth clock transition afterCONTROL4.69
(default disabled, existing exact-derivative helper) atspeed.6/1.4 retains12
and stops source971.13/972.25. Disabling only the auxiliary joint tube after
source960 also retains12 but stops968.50. Four clock and18 profile tests pass.
Current trials promote pelvis orientation atsource930/940 or add rootX
tracking at930 to reduce error before the landing rather than after it.

Stronger contact search4/8 m/s² records12 at4.642/4.620 (38/60 ms sooner)
but still stops around source972.3/971.7. The optional final unloading cost
(`--final-unload-start/weight/ramp`, default disabled) affects only the
existing outgoing-contact wrench objective. Atweight2048 aftersource960,
measured left load falls tozero and the controller releases its contact by
4.770; native clearance reaches2.9 mm by4.792, below the20 mm flight criterion.
It still stops on jointRMSE. Current trials pair this genuine unloading with
foot-metric reductions, which previously had no effect while both feet were
constrained.

Earlier fifth clock transition atCONTROL4.26 (after step11) with final speeds
1.0/.8/.6 fails joint fidelity before12, atsource945.08/934.26/939.96. Root
position errors improve to.165/.148/.155 m, but left hip yaw deviates
-.832/-.908/-.921 rad. In the speed1.0 run, loaded left-foot heading is-.957
rad from original, while tilt remains small. The existing support-orientation
selector now accepts `--no-support-orientation-roll-pitch-only` to test all
free angular rows without changing its prior positive/default behavior.

Unloading2048 plus foot multiplier16/64 and unloading10000 plus foot64 all
retain12, stop atsource972.09, and fail the20 mm unloaded-flight criterion.
Full free support orientation on earlier-clock1.0/.8 still fails11-only at
source944.44/934.26. Increasing leg posture64->256 with earlier-clock1.0
reaches paused960 but cannot verify12 before joint error exceeds.2.

Next diagnostics track the problem backward: atstep11/control4.230 actual
COM=[5.207,-.905,.596] m versus target[5.281,-.930,.665], velocity
[.868,+.159,.019] m/s. By4.680, actualCOM=[5.382,-.884,.670], target
[5.522,-1.012,.682], velocity[.039,-.005,-.080]. The paused transfer loses
forward motion and accumulates lateral lag. Current trials enable the same
bounded contact-search request already atsource860 for step11, or strengthen
rootXY tracking to10000 aftersource900. No source acceptance changes.

### Optional observed near-touching support experiment

`audit_touching_geometry.py` replays the saved actual PhysX ground manifold.
Atcontrol4.400 the left foot has138.57 N on one corner; the other three
native sole points have zero force but only9/11/19 micrometres separation.
The loaded-only selector collapses a flat sole to one point. A10 micrometre
selector gives a line, while50/100 micrometres retain the measured four-point
polygon. By4.450 the foot has tilted and all selectors correctly return a
point. This is a model diagnostic, not evidence that zero-force points were
already carrying load.

Default-off `--touching-support-start` and `--touching-support-tolerance`
(default50 micrometres, maximum100) permit load redistribution onto these
reported near-touching manifold points only when that foot already has at
least20 N of measured bearing load. No point is inferred from a planned pose
or mesh; separated points are excluded, the original unilateral/friction
polygon cone is retained, and no airborne foot gains support. Original
source/contact/fidelity audit stays unchanged. The default selector is still
loaded-only. Seven observed-sole tests (including two new coverage tests),
six patch-manager tests, six wrench-cone tests and18 profile tests pass.
Current online trials enable10/50/100 micrometres aftersource900.

Near-touching tolerances10/50/100 micrometres all fail11-only atsource950.73
on joint error, despite root error .161 m. Reducing the later foot multiplier
to256 (50 micrometres) restores12: online4.892, independent100N/50ms certificate
4.844–4.894, minimum124.30 N, maximumfooterror21.35 mm, flight34.43 mm.
It then fails atsource960.459; this is worse than the reproduced12 baseline,
so neither the launcher's default nor the original loaded-only model changes.
Foot1024 remains11; foot256/search8 fails root angle atpaused960.

Removing the second contact pause with search8 stops11/source955.12;
pause952/search8 stops11/952; right-foot shortened20 mm stops11/954.42.
Position tube tightened to.18 m atsource900 stops11/952.15; orientation tube
.30 rad at900 keeps12 but failsjoint971.13 (root angle .29562 rad); both tighter
tubes stop11/952.66. These auxiliary guards are distinct from the unchanged
original .2 m/.35 rad acceptance.

`prepare_step11_foot_bias.py` creates offline bounded left-leg IK candidates
from the verified12 reference, with offsets[-20,+20] or[-30,0] mm. Root,
right leg and upper body remain identical; q/v/a prefix through834 is exact,
position IK residual<.51 micrometres, joint-limit residual<3e-8 rad, candidate
source jointRMSE<=.142581 rad. Actual source landing acceptance remains40 mm.
Current trials test both candidates and position tube.18 with foot256/search8.
`summarize_twelve_followups.py` writes a compact completed-trial JSON index.

Both step11 foot-bias candidates stay11-only, failing joint fidelity atsource
955.93/951.30. Position guard.18 + foot256/search8 retains jointRMSE.17862 and
rootposition.17850 until4.940 but loses root angle (.35007), with best loaded
step12 error60.09 mm. Adding orientationguard.30 and foot512 improves the
best loaded error to52.13 mm but failsjoint atpaused960; foot1024/2048 fail
joint earlier at956.43/954.61. None qualifies12.

Offline state/error-dependent posture-weight sensitivity was evaluated in
`step13_adaptive_posture_sensitivity.json` and
`step13_energy_posture_sensitivity.json`; the tested weights worsen the joint
error acceleration projection. **No adaptive-weight runtime feature was
implemented.** Current trials instead use existing stance-only posture-cost
scaling (.1/.01 aftersource900, or .1 after960 on the unloading/foot64 branch).
All original fidelity and native contact acceptance remain unchanged.

Stance-cost.1 after900 fails11/source946.88; .01 fails the sole QP at927.30.
After960 on unloading2048/foot64, stance-cost.1 keeps12 but stops972.09.
Momentum task enabled after816 (damping4, total-rate cap30) at scales1/10/100
stops11/source941.89,10/900.00,10/899.57 respectively. Audited first10 prefix
throughCONTROL3.800 remains exact and every update/failed attempt is single-QP.

`prepare_step12_early_ground.py` shapes only the right leg to lower the
native touchdown height during930–946 or924–940, then returns to the parent
by960. Both keep reference native flight>35 mm, limits<3e-8 rad and exact
q/v/a prefix924/918. Dense stride6 spline native minimum is-.433/-.929 mm
(parent -.433 mm), recorded in `step12_early_ground_spline_audit.json`.
Search2 on either candidate stays11;930–946/search4 verifies12 strictly at
4.634 (50 ms, min101.55 N, maxfooterror16.00 mm, flight33.99 mm), then fails
joint atsource967.26. Earlier touchdown alone does not restore step13.

`prepare_waist_balance_bias.py` makes smooth waist/optional shoulder biases
with fixed root and legs, default906–954 and fade1000–1060; optional earlier
840–900. All candidates pass raw-source reference jointRMSE.142581 and joint
limits. Waist(+.15,+.15), (+.2,+.3), (0,+.3) shifts referenceCOM at960 by
[7.8,-9.8,-1.1], [16.2,-13.3,-3.6], [17.6,-2.6,-3.2] mm, but each fails11-only.
Combined waist/shoulder-.3, early combined, and shoulder-only also stay11,
stopping source942.05/957.57/953.37. These remain optional offline candidates.

`step12_acceleration_prediction_audit.json` compares commanded QP acceleration
with next-tick measured velocity differences on unchanged-contact-mode samples.
Waist acceleration RMS closely agrees (e.g. best12/control4.65–4.8 pitch
76.25 predicted vs76.00 observed rad/s²). Thus the waist divergence is largely
commanded behavior, not evidence for an unmodelled bias-torque correction.

Ending the optional COM upward-speed ceiling after900 loses12; ending after
930/960 retains12 with the same source972.49 stop. Current experiment adds
`--late-fabric-priority-start/mode` (default disabled): it switches only the
existing algebraic Fabric composition to `paper-tracking` or
`contact-tracking`, then feeds its acceleration/metric to the same one native
inverse-dynamics QP. No priority QP stages are added. History records the
effective composition mode. Four composition tests and18 profile tests pass.
Trials test paper/contact aftersource900 and contact after960.

### Continued strict single-QP experiments

Late algebraic Fabric contact/paper composition aftersource900 stops11 at
923.95 (QP validation)/954.61 (joint fidelity). Contact composition after960
keeps12 but stops968.11. All actual native attempt counts remain1, exact QP
torque is applied, and no extra PD appears. The contact960 branch preserves
171 history fields through the original12 certificate atcontrol4.688 exactly.

Ending the auxiliary position tube at960 keeps12 but fails original root
position atsource961.37 (.200665m), even though jointRMSE falls to.10912.
Ending at930 loses12 at948.76 on rootposition. Reducing only position tube
bandwidth to4 from930 loses12 at956.97 on joint fidelity. Thus merely removing
this guard transfers the failure to original root acceptance, not a solution.

Default-off `--final-tube-soft-weight-start/weight/ramp` provides a subsequent
smooth change of the existing joint tube slack penalty in the same single QP.
`step13_joint_tube_strong_sensitivity.json` checks weights10–10000 times larger
offline at4.740; all retain physical feasibility, but waist acceleration alone
still worsens, so online whole-body acceptance must decide the outcome.

Default-off `--late-com-metric-start/scale/ramp` changes the existing COM cost
smoothly after the original multiplier. It neither adds a force nor a solver.
`step11_root_com_sensitivity.json` uses the archived `eleven_resume10` state
atcontrol3.900 (not the exact best12 state after its source834 reference change).
In this frozen diagnostic,10x COM cost raises
feasible forwardCOM acceleration from1.00 to4.15m/s² and reduces jointaccnorm
372 to343rad/s². RootXY10000 instead raises jointaccnorm to785. These are frozen
QP diagnostics only; queued full-source trials test total COM scales16/40
starting source816 or860. Profile18 and single-QP6 tests pass.

Joint slack cost10000 fromsource940 keeps12, online4.706 and strict100N/50ms
certificate4.666–4.716 (minimum101.72N, footerror6.21mm, flight34.27mm), but
stopsjoint at972.80 with no13flight. Cost10000 after960 stops972.49;100000 after
940 prevents12touchdown and failsorientation atpaused960. All three audit
single native attempts, exact appliedQP torque, and unchanged source thresholds.
Original best12 remains the repeated baseline; the .31frame extension is not
an additional verified step or a fulltracking result.

COM totalcost40 from816 fails10/source882.80;16 from816 and40 from860 keep11
but failjoint at952.51/949.11, with step12 nativeflight only17.83/12.27mm.
Optional `--late-com-metric-axes xy` uses positive-semidefinite congruence
scaling to change only horizontal cost. XY40 from816 still fails10/882.33;
XY16 from816 andXY40 from860 keep11 and restore step12flight43.51/58.64mm,
but failjoint before touchdown at930.44/935.22. Main errors involve hips,
waist and shoulders. All six preserve first10 throughcontrol3.800 exactly,
use one native attempt/update, and apply only the certified QP torque.
Current followups advance existing leg64/upper256 costs on the two XY branches
and test the same earlier posture activation alone. Fullprofile18 passes.

`step10_touching_geometry.json` replays the right-foot manifold at3.800–4.200.
Near-touching tolerances10/50/100micrometres do not change any sampled support
selection: line at3.8, point3.85–3.95, line4.0–4.2. No basis was found for
using that optional contact model to fix this earlier transfer.
`step12_horizontal_com_sensitivity.json` instead examines the exact best12
snapshot4.300: even10x horizontalCOM cost leaves forward acceleration negative;
100x permits positive acceleration only with negative vertical acceleration.
These are frozen diagnostics, not dynamic feasibility certificates.

Advancing leg64/upper256 to816 onXY16 keeps11/954.52; advancing to860 onXY40
keeps11/paused960 but step12flight falls to6.80mm. Posture-only activation at816
loses11 and stopsjoint atpaused900. All preserve first10 throughcontrol3.800
and pass the strict single-QP/appliedtorque audit.

Best12 history shows measured anticipatory unloading already activates at
control4.650/source959.956 after real right-foot bearing evidence. The previous
strong final cost starting atsource960 activates later and only gradually.
Current experiments prepare final outgoing-wrench costs128/2048/10000 from940;
they still activate only during existing dynamically gated measured transfer.
No force is applied outside QP, no support row is invented, and original
landing/source fidelity gates stay unchanged.

Early unloading128/2048/10000 all retain12 but stopjoint at972.09/970.09/970.09.
The2048 branch frees left-foot contact already aroundcontrol4.650 and reaches
nativeleftflight9.38mm (original required20mm). Strict12 certificate remains
4.638–4.688, minimum110.92N, maximumfooterror8.16mm, flight34.27mm. This is still
not13. Queued followups use .04/.06s recovery after12's causal landing pause,
or reduce only left Cartesian cost to16 before release. New default-both
`--foot-metric-final-side` supports left/right selection using PSD congruence;
right task cost is retained for left-only selection. Original defaults are
unchanged; fullprofile18 passes after this optional selector.

Slowing fourth clock transition from1.4 to1.0 retains11 but failsjoint at906.23:
rooterror.04197m, COMvx1.136m/s, dominant errors rightknee-.617/anklepitch-.661rad.
Right-foot measured unloading starts around4.200 and contact releases4.280.
Clock1.2 and1.2+COMxy16 retain11 but failjoint957.09/952.93. All three have exact
first10 prefix throughcontrol3.800 and pass single-QP/torque/acceptance audits.
The slower1.0 branch may offer better incoming root/momentum state, but its
worse overall tracking is not promoted as the baseline. Next queued checks
advance leg64 tosource880, unloadcost128 to880, or both.

Early-unload branch resume.04/.06 retains12 but failsjoint at969.29/970.49,
with native13flight7.03/7.02mm. Left-only Cartesian16 retains12 but stops970.89
without unloaded13flight. All preserve the earlier11 prefix throughcontrol3.850,
pass actualsingleQP/appliedtorque audits, and keep original acceptance. No branch
is promoted over the independently repeated12 baseline. Slowclock1.0 right-leg
transfer followups are running next.
