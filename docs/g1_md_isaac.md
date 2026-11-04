# g1.md 在 Isaac Sim 中的实现

入口：`examples/test_g1.py --g1-md`；完整 PKL 配置启动：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_g1_md.sh
```

显示窗口可追加 `--show-isaac-window`。保留旧入口作为对照；新脚本明确启用新控制路径。

## 实际数据流

1. 复用现有 PKL 检查、目标机器人 FK、统一参考和实测接触管理器。四步终点仍为 129、155、176、198 帧；不修改 PKL，不新增 ZMP 或简化动力学模型。
2. `dynamics-aware` 任务包含 COM、躯干、源 pelvis 姿态、足端、手部位置、关节姿态与避碰。手部姿态可用 `--hand-tracking pose`。COM 与 pelvis 不是同一个物理量，不能以 pelvis 高度冒充 COM。
3. 所有任务（包括接触姿态、关节软限位附加任务）先以自然形式拉回并求和：`H = εI + Σ AᵀWA`，`f = Σ AᵀW(bias−desired)`，`a_fabric = −solve(H,f)`。任务 bias 包含运动学加速度偏置。实现位于 `solver/fabric_mit.py`。
4. 以合成加速度及其完整度量构成一个根策略，针对包含 MIT 反馈的整机预测加速度求接触动力学可行解。此模式不再把各任务作为独立 QP 目标或使用任务层级。仍保留浮动基座动力学、接触、摩擦、电机力矩、关节速度约束及已启用的碰撞约束。软件加速度/制动限幅仍默认关闭。
5. 从初始实测关节状态启动持续半隐式积分：`v_des += dt*a_joint`，`q_des += dt*v_des`。不在每个 tick 重置到实测状态，不以 PKL 关节角直接覆盖积分命令。浮动基座参与求解但不生成基座执行器命令。
6. `tau_cmd = tau_ff + Kp*(q_des−q) + Kd*(v_des−v)`；QP 通过 `integrated_servo_model` 计入本次积分后的 PD 和总力矩预算。PhysX 内部驱动增益为零，最终通过 `set_joint_efforts` 执行；两次控制更新之间保持目标并以新实测状态计算 PD。
7. 控制 1 ms，Isaac PhysX/PD 0.5 ms。增益沿用按整机有效惯量限制的已有计算，不对所有关节直接套用文档的 40–100 Nm/rad。腕部与腿部执行器能力不同。

## 对原方案必须作出的修正

解析 Fabric 合成只能生成一个加速度意图，不能保证它满足欠驱动动力学、单向接触、摩擦和电机能力。固定任意加速度后仅求接触力也可能无解，因此保留允许修正加速度的约束动力学求解。此实现没有声称完全消除 QP，也不承诺“绝对不会崩溃”。无效解不会继续作为有效力矩发送。

不使用 MuJoCo，不新增该依赖。1 kHz 是仿真控制时钟设置，尚不构成实时性能或稳定性证明。连续积分也不自动保证无漂移。

## 验证范围

单元测试覆盖模式切换、保留速度限制、持续积分 PD 与求解器仿射力矩模型一致，以及带 bias 的自然形式拉回求和。Isaac 短运行只验证执行链路，不能证明四步成功；源参考仍需经过卸载、摆动、实际着地与落脚误差检查。

2026-10-28 实测：`test_fabric_mit` 与 `test_joint_control` 共 24 项通过。最终代码在 Isaac Sim 完成 0.5 秒、1000 个物理步、500 个控制更新，数值链路和站立参考检查通过，记录为 `acc_data/g1_md_smoke.npz`。短运行止于预热第 0 帧，四步尚未执行；程序因完整源参考尚未完成而按既有验收规则退出码为 1，不能报告整个 tracking 测试通过。


## 四步修复与诊断

原 g1.md 四步运行在 14.525 s、约第 113 帧 PREPARE_STEP 超时，尚未完成首步。最后测得左脚 103.89 N、右脚 228.85 N，左脚载荷比例 31.22%，COM 误差 0.10776 m。最后 1 秒平均左右载荷为 74.22/254.60 N。预测关节速度约束违例为零。基准摘要：`acc_data/walk1_g1_md_four.failure_summary.json`。

已修复：`compose_task_policy` 同时保存合成加速度与 H，作为单一根 TaskRequest 交给物理求解，使用全 35 维预测加速度（包括 MIT 反馈响应）。叶子目标不重复加入。初始化与每次在线更新均使用该路径。合成代价差与原叶子加正则代价差的等价性、MIT 反馈模型和源关键帧测试共 28 项通过。

后续实施顺序：先实测修复后的首步卸载，再决定是否启用/改造整机接触载荷转移参考；不得用放宽接触门槛替代卸载。若仍不足，应联合检查 COM/动量反馈、真实力矩预算和双脚锚点下的可行整机参考，保留四个源落脚位置，明确记录必要的中间姿态/步时调整。最后逐步验收抬脚、摆脚无穿地、原位置落脚以及四步后站稳。修改度量不构成稳定性证明。


## 正常四步流程的载荷转移接入

`reference/mocap_transition.py` 现在由 `--g1-md --mocap-pipeline` 自动启用。PREPARE_STEP 中源参考停稳后，从当前测得 q/v 和上次 WBC 预测加速度建立 0.5 秒 C2 接入；用整机 FK/质量模型和实际双脚姿态生成 2 秒载荷转移参考。卸载目标的权重按同一个转移进度平滑增加（默认软权重 1e-4）。原实测接触/载荷门槛保留，并要求整机转移参考的 COM、速度、腿姿态条件满足；不强制切脚。

进入单支撑时，参考从实测 q/v/上次预测 a 接回正在推进的 PKL，偏移在不超过 0.25 秒且不超过估计剩余摆动时间一半的窗口内消失。不是重置机器人，不修改 PKL，也不移动四个原始落脚目标。过早进入落地阶段会报错，不能在偏移仍有效时将着地算作原目标跟踪成功。每个新 takeoff 事件重建独立的左右载荷转移。记录 `history_mocap_transition_phase` 与 `history_mocap_transition_unload_weight`。

上述桥接仅保证参考连续；尚不保证途中足底间隙、碰撞、完整动力学或长期积分稳定性。必须用实际四步运行验收。目标仍为原始 129/155/176/198 帧位置。

数值修复：前一次仅传递度量的运行在约 10.9 秒因浮点矩阵不对称退出。现在所有拉回/合成及根任务度量均保留 float64，并显式取对称部分；不增加物理约束容差，不改变任务权重。task_objective 允许与累加精度一致的根任务，残差在任务自身精度中计算。回归覆盖病态 float32 输入、混合精度完整 WBC、C2 接入/退出、左右下一步事件、过早落地拒绝；包含既有动力学测试的 51 项通过。真实 Isaac 验证另行记录，不能以这些单元测试宣称四步已成功。


### 持续积分漂移修复

载荷转移接入后的实测记录 `walk1_g1_md_transition_four.reference_failure.npz` 在 10.865 s 丢失双脚支撑，首步未放行。8 s 时左踝滚转命令相对实测偏差约 -1.46 rad，随后左踝俯仰到达 20 Nm 的 QP 预算；全过程最大积分关节偏差 1.8413 rad，末 COM 误差 0.7421 m。诊断报告和三联图由 `examples/analyze_integrated_tracking.py` 生成。

现在 g1.md 路径复用 `synchronize_command_acceleration`：关节命令同步修正为 `clip(20*(q_measured-q_des)+8*(v_measured-v_des), -30, 30)`，连续积分保留、不重置。此项限制的是同步修正幅值，不是重新启用实际加速度/关节制动硬约束。修正通过 QP 的 `command_acc_correction` 从物理名义加速度中扣除；合成根任务仍跟踪包含 PD 响应的实际预测加速度，速度/力矩/接触约束继续作用于实际预测量。含这一映射的 MIT、关节控制和动力学回归 43 项通过。整机效果需要后续 Isaac 记录，不能由代数测试推断。

状态机修正：`LANDING` 也可以只表示接近落地时减速。桥接只能在实际回到双支撑 (`mode == 3`) 时认定提前触地，不能仅按阶段名字判断。相关 5 项状态机测试通过。

### 2026-10-30：实际静止门槛实测与参考动力学审计

本次运行：`G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python bash examples/run_g1_md.sh --duration 25 --output-path acc_data/walk1_g1_md_settled_four.npz`。原始四步落脚目标未改动，本轮没有改变控制器参数。实际静止门槛要求连续 0.2 秒 COM 速度不超过 0.04 m/s、关节速度不超过 0.2 rad/s、每脚法向载荷至少 20 N。8.526 s 进入 WAIT_FOR_MEASURED_REST，9.888 s 开始 TRANSFER，最后记录为 11.151 s、第 112.50006 帧。终止原因 `ReferenceManager lost support: expected 3, observed 2`；接触管理器有确认延迟，最后原始记录的两脚法向载荷均为零。末 COM 误差 0.73317 m、最低 COM 高度 0.22853 m。`analyze_bounded_steps.py` 实测完成 **0/4**，没有进入首步摆动。

9.5 s 和 9.887 s 左踝俯仰的 QP 总力矩已到 20 Nm 预算。10.25 s COM 期望加速度约 [-0.913, 2.473, 0.861] m/s²，QP 预测约 [0.286, -0.286, -0.253] m/s²；增加等待未解决问题。不能仅凭力矩饱和断言执行器是唯一原因。

新增并实际运行 `examples/audit_recorded_reference_dynamics.py`，用整机惯量、反射电机惯量、真实电机上限的 0.8 倍预算及完整矩形足底力锥，检查参考 q/v/a 是否允许双脚接触力分配。完整足底相对在线观测支撑片是乐观假设。复现：

```bash
PYTHONPATH=src:examples:.runtime/qp_solvers OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/audit_recorded_reference_dynamics.py \
  acc_data/walk1_g1_md_settled_four.reference_failure.npz \
  --times 9.5 9.887 9.888 10 10.1 10.25 10.5 \
  --output acc_data/walk1_g1_md_settled_four.reference_dynamics.json
```

9.5/9.887 s 的静止源参考可分配接触力；9.888、10、10.1、10.25、10.5 s 五个转移参考采样均无可行分配。接入初始支撑脚最大线速度 0.02266 m/s、角速度 0.13011 rad/s；10 s 最大线加速度 0.16203 m/s²、角加速度 0.91248 rad/s²，与固定双支撑假设不一致。10.5 s 接触运动学残差已接近零，力分配仍不可行，因此单纯消除接入段脚运动也不足够。

这些是当前模型与预算下精确执行所记录参考的必要条件检查，不是对整条 PKL 不可行的证明，也不意味着在线允许修正加速度的 QP 无解。没有检查未采样区间或认证碰撞/稳定性。后续应调整中间整机姿态与转移轨迹，使固定支撑、浮动基座动力学和力矩条件同时可行，再做在线跟踪；保留四个原始落脚目标。

产物：`acc_data/walk1_g1_md_settled_four.steps_report.json`、`acc_data/walk1_g1_md_settled_four.reference_failure.integrator.json`、同名 `.png`、`acc_data/walk1_g1_md_settled_four.reference_dynamics.json`。本轮完成在线运行及离线分析，未宣称四步通过，未放宽物理约束或增加任务权重。

### 2026-10-30：固定实际锚点的源姿态转移

`MocapContactTransition` 的双支撑入口改为：保持停止源参考的 COM、躯干姿态和上身关节，用 `contact_pose_reference` 将腿部拟合到本次实际双脚锚点，再生成整机转移。去掉入口的关节空间 measured-to-reference 偏移桥接，避免它引入支撑脚运动。没有重置真实机器人或积分命令，也未修改原始四个落脚位置。进入单支撑后的有限时间 source join 仍保留。这一修改取代上文旧的入口 C2 桥接描述；源姿态投影可能改变入口腿部参考，不宣称入口全关节 C2 连续。

离线比较 `acc_data/walk1_transfer_start_comparison.json`：固定实际脚姿态、源姿态起点、COM 高度下降 0.02 m 的 2 秒转移，在间隔 0.2 秒的 11 个采样点有可行力分配；使用实测姿态作起点的早期采样不可行。此结论仅适用于检查的记录和采样，不是在线全路径认证。

回归 `test_mocap_transition test_reference_initial_stance test_fabric_mit` 19 项通过。实际运行 `run_g1_md.sh --duration 25 --output-path acc_data/walk1_g1_md_source_anchor_four.npz`，9.888 s 开始转移，最后记录 11.159 s、源帧 112.50006；`ReferenceManager lost support: expected 3, observed 1`，实测 **0/4**。末质心误差 0.74702 m，最低 COM 高度 0.22732 m。没有稳定性改善的结论。

对本轮记录的 9.888、10、10.1、10.25、10.5 s 重新运行 `audit_recorded_reference_dynamics.py`，五个采样点全部有可行接触力分配，最大支撑脚线加速度残差约 2.03e-6 m/s²；修复了之前同时间点的精确参考力分配不可行及显著接触运动学冲突。注意可行性 LP 的目标为零，返回解的峰值力矩比为 1 不意味着所有可行解都必须饱和，也不证明存在足够纠偏余量。

实际转移起点 COM 误差已为 0.02218 m，左踝俯仰达到 20 Nm QP 预算。10.25 s 质心请求加速度 [-1.1633,2.8397,0.8099] m/s²，预测 [0.2721,-0.2888,-0.2655] m/s²。参考姿态上的可行性不等于实际偏离状态能够跟踪；下一步应检查实测状态下可实现的质心纠偏、任务冲突和实际接触片模型，不应继续以增大权重或放宽硬约束代替诊断。

产物前缀 `acc_data/walk1_g1_md_source_anchor_four`：`.steps_report.json`、`.reference_dynamics.json`、`.reference_failure.npz`、`.reference_failure.integrator.json` 和 `.reference_failure.integrator.png`。本轮只验证一次新控制配置；完成四步仍未通过。

### 2026-11-01：实际状态的质心可达性与可选分层控制

新增 `examples/analyze_com_reachability.py`：读取在线 QP 快照的全部等式、不等式及变量上下界，离线求解精确质心目标的可行性，以及仅质心目标的最小最大坐标误差。由动力学等式恢复实际使用的掩码接触雅可比，包含 MIT 仿射响应与命令同步修正；先与快照保存的预测质心加速度交叉校验。没有删除接触、力矩、碰撞或速度约束，也不修改在线机器人。空等式集合测试问题已修复，可达与不可达测试通过。

已完成的 `walk1_g1_md_com_reachability` 运行仍在第一步前丢失支撑。其 `qp_9500.npz` 和 `qp_10250.npz` 均显示：保存的全部硬约束下，质心请求加速度可以精确达到（LP 解约束残差分别约 1.27e-8、1.94e-8）。9.5 s 在线最大质心加速度误差约 0.5001 m/s²；10.25 s 约 3.1284 m/s²。仿射重放误差均小于 7e-8。诊断报告为对应的 `.reachability.json`。

这纠正了仅凭左踝饱和推断质心纠偏不可达的解释：至少这两个瞬时模型中，存在满足全部硬约束的质心目标解，当前加权目标选择了其他折中。可达性不保证未来接触、姿态或稳定性。

新增可选 `--g1-com-priority`（需 `--g1-md`）：使用独立 `priority_requests` 将原始 COM 叶子用于一级求解；最终目标仍只有原来的合成根策略，不重复加入 COM 叶子。二级保留一级达到的 COM 输出，沿用现有 0.02 m/s² 优先级锁定容差。此模式明确改变原来的纯加权合成策略，未默认启用。初始化和每个控制 tick 都应用同一规则，所有物理约束保留。

`test_task_priority test_fabric_mit test_com_reachability` 共 23 项通过，含冲突根任务、独立优先叶子和不可达目标受真实力矩上限约束的回归。在线运行命令：`G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python bash examples/run_g1_md.sh --g1-com-priority --duration 25 --output-path acc_data/walk1_g1_md_com_priority_four.npz`。结果另记，测试通过不等于完成四步。
