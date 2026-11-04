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
