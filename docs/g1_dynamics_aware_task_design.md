# G1 Dynamics-aware Fabric 任务设计

本设计依据 `dynamics_aware_fabric_tasks.md`，并对应当前 `G1PoseFabric`、
`examples/test_g1.py` 与 `G1DynamicsAccelerationQP`。本文保留原始设计规格；
首版实现、实际 API、验证结果与尚未覆盖的范围见
[运行与验证说明](g1_dynamics_aware_usage.md)。短时初始姿态保持已通过仿真，动作跟踪仍未通过。

## 1. 架构与统一约定

```text
Motion reference + measured robot/contact state
  -> contact state reconciliation + reference generation
  -> Dynamics-aware Fabric: task requests and natural forms
  -> WBC: constrained acceleration / torque / contact wrench
  -> existing 500 Hz explicit joint servo -> robot
```

- 配置 `q: [B,36] = [p_world(3), quaternion_xyzw(4), joints(29)]`。
- 广义速度 `v: [B,35] = [linear_world(3), angular_world(3), joint_velocity(29)]`。
- 求解量为 `a = dv/dt`，不是对 36 维配置逐元素二阶求导。
- 世界系表达任务速度、角动量与接触 wrench；接触摩擦/CoP 约束转换到足底局部系。
- Fabric 沿用 50 Hz；参考速度、加速度随 motion speed 分别按一次、二次缩放。
- 位置误差统一为 `current - reference`，控制项为负反馈。
- Fabric metric 使用 `W`；物理质量矩阵使用 `M_rb`，二者不可混用。

每个任务输出统一的加速度层意图：

\[
 A_i(q)a+b_i(q,v)\simeq u_i^{des},\qquad W_i\succeq0.
\]

建议新增数据结构 `TaskRequest(name, A, bias, desired, metric, active, priority)`，
其中 `A:[B,k,35]`、`bias/desired:[B,k]`、`metric:[B,k,k]`、
`active:[B,k]`。标记 inactive 的行必须同时从目标和 metric 中消除；
目标张量需检查形状、有限值、设备和 dtype。

普通位置任务有 `A=J`、`bias=Jdot*v`。其自然形式为：

\[
 H_i=A_i^TW_iA_i,\qquad
 f_i=A_i^TW_i(b_i-u_i^{des}).
\]

兼容路径仍可计算 `a_fabric = -solve(sum(H_i)+epsilon*I, sum(f_i))`，
但保留任务意图给 WBC，避免合成为一个关节加速度后丢失任务权重。
既有根空间阻尼应作为显式低权重正则项记录，避免额外引入不可解释的高优先级目标。

## 2. 任务划分

| 任务 | 维度 | 激活条件 | 目标 | 代码落点 |
|---|---:|---|---|---|
| CoM | 3 | 全程；腾空时接受动力学限制 | 质心位置、速度、加速度 | 复用 `g1_center_of_mass.py` / `center_of_mass.py` |
| Contact | 每个支撑脚 6 | 接触状态确认有效 | 固定落地锚点，满足接触与摩擦约束 | 接触状态机、projector、`AccelerationQP.py` |
| Swing foot | 每脚 6 | 该脚非支撑 | 落足轨迹与足部姿态 | 复用 feet task，增加逐脚 mask |
| Centroidal angular momentum | 3 | 全程 | 角动量及其变化率 | 新增 centroidal model 与 momentum policy |
| Body orientation | 3 | 全程 | pelvis 或 torso 姿态 | 首版复用 pelvis；真实 torso 使用 link Jacobian |
| Joint posture | 29 | 有参考时 | 冗余姿态、参考关节速度 | 复用 `joint_posture.py` |
| Hand | 每手 6 | 操作/动作参考启用时 | 手的位置与姿态 | 复用 hand task，增加逐手 mask |
| Collision | 每个有效距离对 1 | 安全距离附近 | 分离加速度 | G1 collision geometry + distance map + repulsion |

### CoM

\[
 u_c=\ddot c_{ref}-K_c(c-c_{ref})-D_c(J_cv-\dot c_{ref}).
\]

保留现有质量加权 CoM 与加速度限幅。站立可采用支撑区域内参考；行走由轨迹及
接触阶段生成平滑参考，不将每个时刻的 CoM 强制投影到静态支撑多边形。
腾空时质心加速度由重力决定，WBC 必须允许质心任务出现残差。

### Contact 与 Swing foot

维护统一 `ContactState(active:[B,2], anchors:[B,2,7], mode:[B])`，
mode 沿用 `0/1/2/3 = flight/left/right/double`。
同一控制周期 Fabric 和 WBC 必须消费同一个已协调的接触状态。

接触确认由现有计划状态与测量力协调逻辑负责。仅在 inactive -> active 时记录当前
足部位姿为锚点。持续支撑时不可用运动参考刷新锚点。

支撑约束采用世界系足部 twist Jacobian：

\[
 J_ca+\dot J_cv=-K_{contact}e_{anchor}-D_{contact}J_cv.
\]

姿态锚点误差也表达在世界系；这里的稳定项是物理角加速度请求。
位姿完全稳定时右侧为零。Fabric 与 QP 共享同一 bias/稳定项，禁止重复添加。
若接触实际断开，状态机应撤销约束，不能通过继续锁脚掩盖接触丢失。

摆动脚使用位置 PD + 姿态控制与参考前馈。每脚 metric 乘以 `1-active_foot`，
支撑脚的整块六维跟踪 metric 置零；支撑恢复由 WBC 接触约束承担。
接触切换的硬约束使用离散状态，摆脚软权重可以平滑恢复。

当前 `anchor_support_feet()` 能替换支撑脚参考，但并未使该脚退出 soft tracking；
这不等于完整的 stance/swing 分工。现有无 motion 分支在 projector 后才取得接触状态，
应把状态检测前移，避免 Fabric 使用上一周期的接触而 WBC 使用本周期接触。

### Centroidal angular momentum

使用关于整机 CoM、世界系表达的角动量：

\[
 L=A_L(q)v,\qquad \dot L=A_La+\dot A_Lv,
\]
\[
 u_L=\dot L_{ref}-K_L(L-L_{ref}).
\]

因此此任务的 `A=A_L`、`bias=Adot_L*v`、`desired=u_L`。
站立默认 `L_ref=0`、`Ldot_ref=0`；动态动作支持非零参考，避免把自然摆臂一律抑制。
腾空且无外部力矩时，不能要求控制器改变整机总角动量。

由同一动力学模型的每个 link 构造：

\[
 A_L=\sum_i\left[I_i^WJ_{\omega i}
 +m_i[c_i-c]_\times J_{v_i}\right].
\]

`J_vi` 是 link **质心**线速度 Jacobian，`I_i^W` 是关于该 link 质心、已转换到
世界系的惯量。计算必须包括 link 自转项与平动贡献；不能用 `base omega` 代替整机角动量。
现有 CoM 实现可共享 link 质心、质量和 Jacobian 计算，但需要补上惯量信息。
优先让 `A_L` 与 WBC 质量矩阵来自同一模型；若继续使用 URDF 与 Isaac 两套模型，
必须检查总质量、惯量、关节次序和浮动基坐标转换的一致性。

`Adot_L*v` 首版可用配置流形上的中心差分：

\[
 b_L\approx\frac{A_L(q\oplus\epsilon v)-A_L(q\oplus(-\epsilon v))}{2\epsilon}v.
\]

使用现有 quaternion 积分约定，固定 v，仅扰动 q。通过 epsilon 扫描检查数值误差，
不能直接复用控制周期作为差分步长。避免使用跨周期差分引入接触切换伪脉冲。

`L` 不属于普通 `x=phi(q)` 位置任务。现有 `BaseMap`/pullback 假设 `xd=Jv`，
因此新增 `evaluate_velocity_task(q,v)` 入口输出 TaskRequest，直接在自然形式层合并。
不要伪造 `forward_position()` 返回零位置和 `A_L`，这会把非完整速度任务伪装成位置映射。

### Body orientation

首版选 pelvis orientation，名称明确为 body/pelvis orientation；G1 pelvis 与 torso
之间存在腰关节，若目标是胸部朝向，必须使用 torso link 的 FK 与角速度 Jacobian。
CoM 承担全身平移平衡，旧 pelvis translation 默认降为可选低权重参考，避免二者争抢。

统一世界系误差与物理角加速度请求：

\[
 e_R=\operatorname{Log}(RR_{ref}^T)^\vee,\qquad
 u_R=\alpha_{ref}-K_Re_R-D_R(\omega-\omega_{ref}).
\]

本任务使用 `A=J_omega`、`bias=Jdot_omega*v`，残差单位是物理角加速度。
原始说明中的 `Log(R^T R_ref)` 配 `-K*e` 在小角度下反馈方向相反，且该误差在机体系；
不能直接与世界系角速度混用。

现有 `PelvisPoseTaskMap` 返回的是 log 误差 Jacobian，包含 `J_l(e)^(-1)`。
若保留该误差坐标实现，应完整转换参考速度与加速度（包含移动参考和 Jacobian 导数项），
不能直接将世界系 omega/alpha 当作误差导数。新 TaskRequest 接口建议使用上述物理 twist
表述；feet/hand 姿态块采用相同约定。

### Posture 与 Hand

Posture 使用 `A=[0_(29x6), I_29]`、`bias=0`：

\[
 u_q=\ddot q_{j,ref}-K_q(q_j-q_{j,ref})-D_q(v_j-v_{j,ref}).
\]

无可靠二阶参考时设前馈为零。关节限位不能仅靠 posture 保证；位置/速度预测边界
应继续交给下层约束。Hand 保留六维跟踪，采用独立激活权重与相同姿态约定。
手若用于承重，需显式注册为接触并扩展 WBC 接触模型，不能仍按普通 hand attractor 处理。

### Collision

先为 G1 配置 link-local 球/胶囊近似、半径和需要检查的非相邻 link 对，再连接环境距离查询。
距离 d 为几何表面间隙，正值表示分离；`J_d=dd/dq` 的速度输入为 35 维切空间。
固定球对有 `J_d=n^T(J_a-J_b)`；动态环境还需计入障碍物速度、加速度。

采用在 activation distance 外为零的平滑权重，在近距离内生成正向分离加速度，
可使用截断的反距离势与接近速度阻尼：

\[
 u_d=\operatorname{clip}\left(
 k\left[\frac{1}{\max(d-d_{safe},\epsilon)^3}
 -\frac{1}{(d_{activate}-d_{safe})^3}\right]_+
 -k_v\min(\dot d,0),0,a_{max}\right).
\]

阻尼项也乘以平滑激活函数。处理重合球中心、穿透、距离梯度不连续及 metric 上界，避免 NaN。
现有 `body_sphere_3d_repulsion.py` 可作为复用候选，但 G1 尚需注册几何和浮动基 task map。
软排斥不能保证零碰撞；若最小间隙是硬要求，可在 WBC 加入
`J_d*a+b_d+2*zeta*omega*dot(d)+omega^2*(d-d_safe)>=0`，并定义不可行恢复策略。
因此“collision 低优先级”仅指普通软排斥偏好，不能解释成安全间隙保证。

## 3. WBC 与优先级

WBC 决策量是 `[a(35), tau(29), lambda(active contacts)]`，物理约束为：

\[
 M_{rb}a+h=S^T\tau+J_c^T\lambda,
\]
\[
 J_ca+b_c=u_c^{contact},\quad
 \lambda\in\mathcal F,\quad \tau_{min}\le\tau\le\tau_{max}.
\]

沿用现有 QP 对显式反馈、积分命令修正和总力矩边界的处理。物理加速度与命令同步修正
继续区分，不把纯同步修正计入物理 feedforward。
平足接触 wrench 包含力和力矩，所以角动量平衡还必须包括接触力矩：

\[
 \dot L=\sum_i\big((r_i-c)\times F_i+\tau_i^{contact}\big).
\]

推荐首版使用任务目标直接组成加权 QP：

\[
 \min\;\frac12\sum_i\|A_i a+b_i-u_i^{des}\|_{W_i}^2
 +\frac{\rho_a}{2}\|a\|^2+\frac{\rho_\tau}{2}\|\tau\|^2
 +\frac{\rho_\lambda}{2}\|\lambda-\lambda_{ref}\|^2.
\]

实现时增加 `H_aa += A.T W A`、`g_a += A.T W (bias-desired)`。
任务残差按各自容许误差/单位归一化后再设置优先组，不能仅比较 COM 与角动量原始 metric 数值。
分组为 high(CoM/momentum/body orientation)、medium(swing foot)、low(posture/hand)，
collision 根据距离激活。该设计是软优先级，高组内部也可能互相冲突。

若需要严格优先级，增加分层 QP：先在硬约束下求 high，再保留其最优任务输出求 medium，
最后求 low。正则项只用于选取解，不应冻结多余自由度；不可用单纯放大权重宣称实现严格层级。

原有 projector 可以保留为基线/诊断路径；直接任务 WBC 路径由同一 QP 处理接触，无需先对
全部任务投影再重复添加目标。两种路径由显式配置选择，避免双重计权。

## 4. 拟新增接口与文件改造

以下均为拟定接口，不是当前可直接调用的 API：

```python
fabric.set_contact_state(contact_state)
fabric.set_centroidal_momentum_target(L_ref, Ldot_ref=None)
fabric.set_body_orientation_target(quaternion_xyzw, omega_ref=None, alpha_ref=None)
requests = fabric.compute_task_requests(q, v, model_state)
wbc.solve(task_requests=requests, contact_state=contact_state, ...)
```

| 文件 | 改造 |
|---|---|
| 新增 `src/fabrics_sim/fabrics/task_request.py` | 请求结构、验证、自然形式组装 |
| 新增 `src/fabrics_sim/taskmaps/g1_centroidal_momentum.py` | link 动力学数据、A_L 和 bias；明确为速度映射 |
| 新增 `src/fabrics_sim/fabric_terms/centroidal_momentum.py` | L/Ldot 参考与阻尼策略 |
| `src/fabrics_sim/fabrics/g1_pose_fabric.py` | 统一 requests、接触状态、任务激活及兼容 acceleration API |
| `src/fabrics_sim/fabric_terms/g1_end_effector_pose_attractor.py` | 逐脚/逐手权重；物理姿态任务接口 |
| `src/fabrics_sim/fabric_params/g1_params.yaml` | momentum 参数、body orientation、任务归一化与激活配置 |
| `src/fabrics_sim/solver/AccelerationQP.py` | 接受 TaskRequest 目标并报告各任务残差 |
| `examples/test_g1.py` | 接触协调前移、统一模型状态、任务参考和诊断接线 |
| 新增 G1 collision 几何配置和 distance map | 配对过滤、几何参数、环境查询与可选安全约束 |

不直接覆盖已有参数调优结果。新增 dynamics-aware profile，并保留当前 profile 作为基线。
首个 profile 使用 pelvis orientation；真实 torso 是明确的可选目标，避免两个姿态任务重复约束。

## 5. 实施顺序与验收

1. 统一 TaskRequest 和坐标约定，将现有 CoM/posture/foot/hand 的输出接入，验证自然形式等价。
2. 前移接触协调，完成逐脚 stance/swing mask 和锚点生命周期。
3. 实现 centroidal model、momentum policy、body orientation，并接入任务级 WBC。
4. 配置 G1 collision 几何与距离任务，完成 self/environment 两类场景。
5. 在单一 QP 路径下比较基线与 dynamics-aware profile；有严格层级需求再实现 HQP。

必要单元/数值检查：

- 单刚体 `L=I_world*omega`；纯整体平移不产生关于 CoM 的角动量；整体平移配置不改变 A_L。
- 多 link 动量与逐 link 求和一致；`A_L*a+b_L` 与轨迹差分的 Ldot 一致。
- 正负姿态扰动均产生恢复方向；静态/移动参考、近零和接近 pi 的旋转保持有限。
- 批量混合 flight/left/right/double；mask 正确，锚点只在新接触刷新。
- task metric 半正定、自然形式正负号正确；关闭任务后没有残留 force。
- collision 距离正方向、接近阻尼、穿透与重合边界；远距离零激活。
- 原有 QP dynamics/contact/friction/total-effort 检查继续通过。

闭环场景：静站、单脚站立、左右支撑切换、摆臂、外部推扰及自碰撞接近。
记录 CoM 误差、L/Ldot、姿态误差、摆脚误差、支撑滑移、最小几何间隙、各任务残差、
QP 可行率、力矩饱和和控制耗时。分别消融 momentum、contact mask、collision，
并与相同参考及力矩边界下的基线比较；数值可行不能替代动态平衡和跟踪验收。
