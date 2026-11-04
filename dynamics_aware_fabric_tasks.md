# Dynamics-aware Fabric：Humanoid Tasks 及其设计原因

普通 Geometric Fabric 的 task 主要解决：

- Goal attraction
- Collision avoidance
- Posture regulation

但是对于 humanoid locomotion，核心问题变成：

> 如何在动态接触条件下保持平衡、实现步态切换，并同时维持自然的全身运动。

因此，Dynamics-aware Fabric 需要加入与人形动力学直接相关的任务，例如：

\[
\mathcal{F}
=
\{
\mathcal{F}_{CoM},
\mathcal{F}_{Contact},
\mathcal{F}_{Foot},
\mathcal{F}_{Momentum},
\mathcal{F}_{Torso},
\mathcal{F}_{Posture},
\mathcal{F}_{Hand},
\mathcal{F}_{Collision}
\}
\]

---

## 1. CoM Task

### 作用

解决机器人的动态平衡问题。

质心动力学满足：

\[
m\ddot p_{CoM}
=
\sum_i F_i + mg
\]

如果没有 CoM task，Fabric 即使生成了自然的腿部和手臂动作，也可能使质心偏离稳定区域。

因此加入：

\[
\mathcal F_{CoM}
\]

目标：

\[
p_{CoM}\rightarrow p_{CoM}^{ref}
\]

可定义：

\[
a_{CoM}^{des}
=
K_p(p_{CoM}^{ref}-p_{CoM})
+
K_d(\dot p_{CoM}^{ref}-\dot p_{CoM})
\]

### 为什么需要

- 站立平衡
- 行走过程中的质心迁移
- 外部扰动后的恢复
- 为下层 WBC 提供明确的 centroidal motion intention

---

## 2. Contact Task

### 作用

保证支撑脚满足接触约束。

支撑脚速度约束：

\[
J_c\dot q=0
\]

加速度约束：

\[
J_c\ddot q+\dot J_c\dot q=0
\]

### 为什么需要

Humanoid 是 floating-base system，腿与地面的接触决定了 floating base 是否可控。

如果没有 contact awareness，Fabric 可能同时要求：

- 支撑脚保持地面接触；
- 又通过其他 task 让支撑脚移动。

从而产生冲突。

因此 contact task 的本质是：

> 告诉系统当前哪些自由度由于接触不能自由运动。

---

## 3. Swing Foot Task

### 作用

控制摆动脚跟踪下一步落足位置。

例如：

\[
p_{foot}^{des}
\]

摆脚 task 可以写成：

\[
a_{foot}
=
K_p(p_f^{des}-p_f)
+
K_d(\dot p_f^{des}-\dot p_f)
\]

### 为什么需要

行走过程中，支撑脚和摆动脚承担完全不同的角色：

- Stance foot：满足 contact constraint
- Swing foot：需要跟踪落足轨迹

Motion Generator 可以提供 reference，但 Fabric 可以根据实时状态做 reactive correction，例如：

- 落足误差
- 地形变化
- 外部扰动
- 避障需求

---

## 4. Centroidal Momentum Task

### 作用

控制整机角动量：

\[
L
\]

Centroidal angular momentum 满足：

\[
\dot L
=
\sum_i (r_i-p_{CoM})\times F_i
\]

可以定义：

\[
\dot L^{des}
=
-K_L(L-L^{ref})
\]

### 为什么需要

CoM 只描述平动平衡，不能描述：

- 躯干 pitch/roll
- 摆腿产生的旋转动量
- 手臂和腿之间的角动量耦合

例如高速行走或跑步时，腿部摆动可能产生较大的 sagittal angular momentum。

此时手臂、腰部和躯干需要产生补偿。

因此：

\[
\boxed{
CoM\ control \neq Whole\text{-}body\ balance
}
\]

Momentum task 是补足 CoM task 的关键。

---

## 5. Torso Orientation Task

### 作用

控制躯干或 pelvis 的姿态。

设：

\[
R_b
\]

为当前 orientation，目标为：

\[
R_b^{ref}
\]

orientation error 可定义为：

\[
e_R
=
\mathrm{Log}(R_b^TR_b^{ref})
\]

得到期望角加速度：

\[
\alpha_R^{des}
=
-K_R e_R
-K_\omega \omega
\]

### 为什么需要

仅仅稳定 CoM 不代表身体姿态稳定。

例如：

\[
p_{CoM}
\]

可能仍然处于合理区域，但 torso 已经产生较大 pitch 或 roll。

Torso orientation task 用于：

- 保持身体直立
- 控制前倾
- 抑制 roll
- 配合动态步态

---

## 6. Whole-body Posture Task

### 作用

处理 humanoid 的运动冗余。

目标：

\[
q\rightarrow q_{nominal}
\]

例如：

\[
a_q
=
-K_q(q-q^{ref})
-K_d\dot q
\]

### 为什么需要

同一个 CoM、foot pose 和 torso pose 可以对应很多组 joint configuration。

因此需要 posture task：

- 避免奇异位形
- 避免膝盖过伸
- 避免不自然的关节组合
- 保持类人的 whole-body pose

它主要解决：

\[
\boxed{
Task\ redundancy
}
\]

---

## 7. Hand Task

### 作用

支持 humanoid manipulation 和 whole-body interaction。

例如：

\[
x_{hand}\rightarrow x_{hand}^{ref}
\]

可以定义：

\[
a_{hand}
=
K_p(x_{hand}^{ref}-x_{hand})
+
K_d(\dot x_{hand}^{ref}-\dot x_{hand})
\]

### 为什么需要

如果机器人不仅需要 locomotion，还需要：

- 搬运
- 抓取
- 推门
- 与环境交互

就需要保留 end-effector Fabric。

通常其优先级低于：

- Contact
- CoM
- Momentum
- Torso

---

## 8. Collision Avoidance Task

### 作用

保留原始 Geometric Fabrics 最重要的优势之一。

定义距离：

\[
d(q)
\]

构造 barrier / repulsive fabric，例如：

\[
\phi(d)
=
\frac{1}{(d-d_{safe})^2}
\]

用于：

- self-collision avoidance
- environment collision avoidance
- 保持 safety margin

### 为什么需要

Humanoid 自由度多，全身动作时容易产生：

- 手碰身体
- 双腿干涉
- 手臂撞击环境
- 摆腿撞障碍物

因此 collision Fabric 很适合继续保留。

---

# 任务总结

| Task | 解决的问题 | Humanoid 中的作用 |
|---|---|---|
| CoM | 质心稳定 | Dynamic balance |
| Contact | 支撑脚不能自由运动 | Contact consistency |
| Swing Foot | 摆腿和落足 | Locomotion |
| Momentum | 全身角动量 | Dynamic stability |
| Torso | 躯干姿态 | Body stabilization |
| Posture | 冗余自由度 | Natural configuration |
| Hand | 操作与交互 | Humanoid manipulation |
| Collision | 自碰撞和环境碰撞 | Safety |

---

# 哪些才是真正的 Dynamics-aware 部分

如果做论文，不建议把所有 task 都包装成新的 Dynamics-aware Fabric。

真正体现 humanoid dynamics awareness 的核心是：

\[
\boxed{
CoM\ Fabric
+
Contact\ Awareness
+
Centroidal\ Momentum\ Fabric
}
\]

因为这三个 task 将原本偏运动学的：

\[
\text{Geometric Fabric}
\]

扩展成：

\[
\boxed{
\text{Humanoid dynamics-aware motion generator}
}
\]

而：

- Swing foot
- Posture
- Hand
- Collision

更适合作为传统 Fabric task 在 humanoid 上的延伸。

---

# 推荐的优先级

## Hard constraints / WBC level

1. Robot dynamics

\[
M(q)\ddot q+h(q)
=
S^T\tau+J_c^T\lambda
\]

2. Contact constraint

\[
J_c\ddot q+\dot J_c\dot q=0
\]

3. Friction constraints

\[
\lambda\in\mathcal{F}
\]

---

## High-priority Fabric tasks

- CoM
- Centroidal Momentum
- Torso Orientation

---

## Medium-priority tasks

- Swing Foot
- Foot Orientation

---

## Low-priority tasks

- Whole-body Posture
- Hand
- Collision Avoidance

---

# 最终职责划分

Dynamics-aware Fabric 负责：

\[
\boxed{
\text{生成 desired task-space motion intention}
}
\]

例如：

\[
\{
\ddot p_{CoM}^{des},
\dot L^{des},
\ddot x_{foot}^{des},
\alpha_{torso}^{des},
\ddot x_{hand}^{des}
\}
\]

而下层 Whole-Body Controller 负责求解：

\[
\boxed{
\ddot q,\tau,\lambda
}
\]

使 Fabric 给出的运动意图尽可能满足，同时保证：

- floating-base dynamics
- contact consistency
- friction feasibility
- actuator limits

因此完整架构为：

\[
\boxed{
Motion\ Generator
\rightarrow
Dynamics\text{-}aware\ Fabric
\rightarrow
WBC
\rightarrow
Robot
}
\]
