# G1 Dynamics-aware Fabric：实现与验证

## 使用

从项目根目录、在 `humanoid_fabric` 环境中执行：

```bash
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --duration 10
```

运动跟踪在原有命令上增加 `--task-profile dynamics-aware`：

```bash
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --motion-pkl /path/to/motion.pkl --diagnose-com
```

`--task-profile legacy` 保留旧链路，也是默认值，方便复现已有控制器实验。
`--disable-momentum-task` 和 `--disable-collision-task` 可分别关闭新增策略做消融。
这些选项不关闭物理接触、自碰撞仿真或 QP 约束。

静态环境障碍物使用 JSON 数组 `[[x,y,z,radius], ...]`，单位米、世界坐标：

```bash
PYTHONPATH=src python examples/test_g1.py --task-profile dynamics-aware --collision-obstacles /path/to/spheres.json
```

## 实际控制行为

- **CoM**：现有位置、速度和加速度参考；补齐仅提供加速度参考时被丢弃的问题。
  新 profile 的运动参考使用有界二阶更新，避免对一阶滤波位置二次差分产生起步加速度尖峰；历史记录保存目标速度和加速度。
- **Contact / Swing foot**：先协调接触状态，再供 Fabric 和 WBC 共同使用。支撑脚六维跟踪目标退出软目标，摆脚继续跟踪。锚点仍仅在新接触时更新。
- **Momentum**：由全部 link 的质心、质量、世界系惯量和 Jacobian 计算 `L=A_L v`；中心差分计算 `Adot_L v`。Motion 参考生成 `L_ref` 和 `Ldot_ref`；静态参考默认为零。
- **Body orientation**：默认 pelvis 三维姿态任务；全身平移由 CoM 承担。可显式选择 torso link。世界系物理角加速度与姿态误差使用一致的负反馈方向，包含接近 pi 的旋转处理。
- **Posture / Hand**：保留低权重参考跟踪，posture 支持加速度前馈；手支持独立激活。
- **Collision**：八个局部球体、八组自碰撞配对及可配置静态环境球体。近距离平滑激活、接近速度阻尼和有限加速度排斥。

七组软任务加上 WBC 接触硬约束，共覆盖设计中的八类职责。
`TaskRequest` 统一表示 `A a + bias ≈ desired`，使用半正定 metric 和逐行激活 mask。
角动量通过独立速度任务求值，未伪装为位置 task map。

WBC 直接优化任务残差，替代旧的单一关节加速度跟踪目标。存在显式 PD 时，全部任务作用于
`predicted_acc`；命令同步修正继续留在命令/总力矩约束中，但不会被当成物理加速度前馈。
projector 保留用于已有接触 bias、锚点稳定和诊断；其投影加速度不再是新 profile 的 QP 目标。
新 profile 的足部 `Jdot*v` 使用当前状态中心差分，与任务运动学共用快照；旧 profile 保留历史估计。
`--task-track-nominal` 可做名义加速度目标消融，报告残差仍在预测物理加速度上计算。
`--limit-predicted-acceleration` 可启用预测加速度边界。两者默认关闭，均未在 PKL 实验中解决失稳。
脚底几何修正现在默认开启（`--no-use-sole-wrench-geometry` 仅用于旧配置对照）：
使用保守半宽 .025 m、后跟 .05 m，并把 ankle wrench 平移到下方 .035 m 的脚底平面。
控制循环同时按实测关节剩余行程和速度生成预测加速度上下界，保留独立的命令积分边界。
制动预算为 5 rad/s²、关节余量 .01 rad；这依赖当前接触模型，不能保证外部冲击下不撞限位。
计划单支撑时，摆脚若承受超过总载荷 25% 且至少 20 N，立即恢复双支撑约束。

启动新 profile 时比较 URDF 与 Isaac 的归一化质量矩阵，差异超过 5% 时明确报错，
避免在模型、关节排序或浮动基坐标不一致的情况下执行角动量控制。
URDF 导入也已补上惯性坐标系到 link 坐标系的惯量旋转。

## Python API 与配置

```python
fabric = G1PoseFabric(batch_size=1, device="cpu", task_profile="dynamics-aware")
fabric.set_contact_state(active)  # bool [B,2]，left/right
fabric.set_centroidal_momentum_target(L_ref, Ldot_ref)  # [B,3]
fabric.set_body_orientation_target(quat_xyzw, omega_ref, alpha_ref, link="torso_link")
fabric.set_hand_task_active(active)  # bool [B,2]
fabric.set_collision_obstacles(spheres)  # [B,N,4]，静态 xyz/radius
requests = fabric.compute_task_requests(q, v)
# 在原有 solve 参数上增加 task_requests=requests。
```

模型状态使用 float32、配置 `[B,36]` 和速度 `[B,35]`；Warp 应先调用 `wp.init()`。
`compute_acceleration()` 提供自然形式合成的无约束参考；实际发送给机器人的命令仍需经过 WBC。

新参数位于 `src/fabrics_sim/fabric_params/g1_dynamics_aware.yaml`；CoM、脚、手和 posture
继续复用 `g1_params.yaml` 的参数。momentum/body 权重仅表示软偏好，尚未实现严格分层 HQP。
现有基线参数未被新 profile 覆盖。

在 `test_g1.py` 的结果 NPZ 中，`task_profile` 标记所选方案，`task_residual_*` 保存最后一次 QP
在预测物理加速度上的任务残差；运动历史额外记录逐周期残差、角动量和碰撞代理距离。
inactive 行残差为零表示该行未参与目标，并不代表该脚或手实际误差为零。

## 已完成的验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/validate_dynamics_aware.py --output /tmp/g1_tasks.json
```

本次使用 `/home/sky/anaconda3/envs/humanoid_fabric/bin/python`：

- **57 项测试通过**：包括自然形式/仿射 QP 梯度、PD 与命令修正、旋转惯量导入、无惯性定义的传感器坐标系、整机角动量、平移不变性、流形差分、接触与手任务 mask、运动参考前馈、障碍物、模型不匹配及 COM/link 原点坐标转换；新增足部加速度差分核验与名义目标消融语义测试。
- **真实 G1 URDF 四模式 QP 验证通过**：flight / left / right / double，均收敛且通过约束验证。
- 修正传感器虚增质量后，独立重算的最大动力学残差约 `3.7e-8`，最大接触加速度残差约 `3.5e-8`；力矩边界无违反。
- 腾空状态 CoM 实现加速度约为 `[0,0,-9.81]`，确认软目标不能覆盖物理约束。

完整数值记录：[g1_dynamics_validation.json](g1_dynamics_validation.json)。
该验证使用真实质量、惯量、Jacobian 和零速度时的重力项；QP 使用 float64，运动学来自
float32 Warp。验证脚本的宽加速度/力矩边界用于检查数值链路，不是硬件限值验收。
报告中的任务耗时是一次批量四实例的 CPU 测量，不是完整 50 Hz 控制器性能结论。

## 尚未验证或未覆盖

最初的 GPU 访问失败是沙箱限制；沙箱外已确认 RTX 4090 D 可用，并完成真实 Isaac rollout。
使用 `B2A_131_02_stageii.pkl`，初始姿态保持 2 秒及修改后回归均通过。
最初动作实验在 1.38 秒停止；后续当前接触偏置加承重门控实验在 2.24 秒停止，
但实际仅推进约 0.65 秒动作，尚未实现稳定动作跟踪。
完整实验与本次模型修正见 [PKL 跟踪实验](g1_pkl_tracking_result.md)。

球体是局部代理，未覆盖完整机器人 mesh；环境接口当前仅支持静态球体。
避碰仍是软目标，不保证安全间隙或零碰撞。新 profile 暂不支持 CUDA graph capture，
已运行 GPU 控制，但未完成实时性能验收。严格 HQP、承重手接触、动态障碍物与硬间隙约束仍属后续扩展。
