# 肩 roll 加速度链路及权重对照（2026-11-12）

**后续已完成 8 组真实闭环试验，见 [闭环报告](closed_loop/REPORT.md)。**
GPU 在获准的沙箱外环境可用；此前沙箱内探测失败不代表主机驱动不可用。
参数启动时钟也已修正为仿真时间。以下保留最初离线阶段的证据与限制。

已完成实测链路诊断和 40 组同状态两级 QP 对照。当前 NVIDIA 驱动不可用，
本轮新闭环仿真次数为 0；以下改善指加速度任务残差，不能等同于关节角误差降低、
通过更多落脚或完整 PKL tracking。

## 数据与方法

- 数据：`../g1_2_tracking/step10_diagnosis/baseline_probe_v2.npz` 及其四对 QP/response 快照。
  记录的 2151 个 `history_q` 样本与视频使用的 `../g1_full_tracking_20261106/baseline.npz`
  完全相同，最大位姿差为 0。
- 实测加速度由同一条指令之后的 `[t_i,t_{i+1}]` 速度差计算，与第 i 行 QP 预测对齐。
  四个 response 快照均与下一条 history 速度一致；下发力矩与快照总力矩最大差为 0。
- 姿态任务的关节 Jacobian 为单位选择矩阵；历史请求通过 `predicted - task_residual`
  还原，并在全部快照与实际保存的 `desired - bias` 核对。
- 每个快照测试基线，以及左肩、右肩、双肩分别 ×2/×4/×8，共 4 × 10 = 40 组。
  用度量矩阵的合同变换缩放权重，保留原状态、参考和全部约束。
- 先重算第一优先级，再根据新的输出重建第二级任务锁的目标；沿用原锁容差。
  没有把原基线锁目标固定后仅修改第二级权重。
- 独立 Clarabel 复算的基线肩加速度与原记录最大差约 3.54e-6 rad/s²。
  40 组两级求解全部通过原单位约束校验，最大残差约 9.30e-10。

## 定位结果

| 时间 | 左肩请求 / QP / 实测（rad/s²） | 右肩请求 / QP / 实测（rad/s²） |
| --- | --- | --- |
| 3.480 s | −26.99 / +45.18 / +45.47 | +68.25 / +19.87 / +20.46 |
| 3.580 s | −66.48 / −55.35 / −53.65 | +67.61 / +48.06 / +52.00 |
| 3.650 s | −29.91 / +1.70 / +1.31 | +58.32 / −94.68 / −95.25 |
| 3.750 s | −63.25 / −18.94 / −17.49 | +17.01 / +3.86 / +5.13 |

3.48 s 左肩、3.65 s 右肩的实际响应跟随了 QP 给出的反向加速度。
3.41–3.48 s 左/右肩“QP−请求”RMS 为 77.77/188.08 rad/s²，
“实测−QP”仅 2.70/3.81 rad/s²；3.48–3.58 s 两者分别为
87.00/108.19 与 4.78/8.21 rad/s²。因此第 9–10 步之间的主要加速度缺口
在任务请求到 QP 输出这一层。

这不能扩展成“整个过程响应模型都准确”：3.58–3.65 s 左肩的
“实测−QP”RMS 为 45.73 rad/s²，大于同期“QP−请求”的 32.90。
有限差分包含冲击、模型误差和离散误差；接触标签不变也不保证没有冲击。
3.3–3.48 s 肩部亦出现过力矩达到预算，不能根据单个落脚时刻推断整段均未饱和。

全部四个快照中，在保留第一阶段原约束、要求双肩精确达到请求加速度时，
独立 HiGHS 可行性检查均找到可行解，原单位残差最高 2.78e-10。
这支持任务折中是可改进因素，但不保证同时保留原来的平衡、脚和身体任务输出。

## 双肩权重 ×4 的结果

| 时间 | 肩加速度请求残差 RMS：基线 → ×4（rad/s²） | 降低 | COM 残差 RMS 变化（m/s²） | 足端平移残差 RMS 变化（m/s²） |
| --- | --- | --- | --- | --- |
| 3.480 s | 61.43 → 20.35 | 66.9% | +0.1234 | +0.2359 |
| 3.580 s | 15.91 → 11.87 | 25.4% | +0.0001 | +0.0148 |
| 3.650 s | 110.47 → 94.39 | 14.6% | −0.1345 | +0.3609 |
| 3.750 s | 32.68 → 11.68 | 64.3% | +0.0714 | +0.1383 |

×4 在 3.48 s 将左肩 QP 加速度从 +45.18 改为 −6.40 rad/s²，恢复了请求方向。
但在 3.65 s，右肩仍是 −74.57 rad/s²，而请求为 +58.32；×8 仍为 −55.36。
提高权重会牺牲部分其他任务，不足以单独修复所有快照。
因此 ×4 仅是值得闭环验证的候选，不是已经选定的新基线。

## 可复现入口

离线复算与绘图：

```bash
PYTHONPATH=.runtime/trajectory_solvers MPLCONFIGDIR=/tmp/g1_shoulder_matplotlib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/analyze_g1_shoulder_roll.py
```

`examples/test_g1.py` 已加入默认关闭的实验参数：
`--shoulder-roll-weight-scale`（默认 1）、`--shoulder-roll-weight-start`（默认 inf）、
`--shoulder-roll-weight-ramp`（默认 0.1 s）、`--shoulder-roll-weight-side`（默认 both）。
权重按五次平滑曲线增加，只修改姿态任务度量；参考、硬约束和原始 PKL 验收不变。
实验开始时间 3.3 s 和 ramp 0.1 s 是待验证选择，不是本轮得到的最优值。

恢复 Isaac 所需 GPU 环境后，可从第 0 帧运行双肩 ×4 候选：

```bash
G1_PYTHON=/home/sky/anaconda3/envs/humanoid_fabric/bin/python \
  bash examples/run_amp_full_motion.sh \
  --output-path acc_data/g1_shoulder_roll_20261112/closed_loop_both4.npz \
  --shoulder-roll-weight-scale 4 --shoulder-roll-weight-start 3.3 \
  --shoulder-roll-weight-ramp 0.1 --shoulder-roll-weight-side both \
  --diagnose-qp-time 3.48 --diagnose-qp-time 3.58 \
  --diagnose-qp-time 3.65 --diagnose-qp-time 3.75
```

闭环对照应分别跑 scale=1/2/4/8 并使用不同输出路径。必须比较实际肩部角度误差、
全身关节 RMSE、根位姿/质心速度、碰撞和通过的落脚数，不能只根据本报告的加速度残差选优。

新增诊断/权重测试 7 项及既有完整跟踪回归 15 项，共 **22 项通过**。
另通过编译及 `git diff --check`。本轮没有改动默认权重，没有进行新物理试验。

## 产物

- [完整 JSON 证据、输入 SHA256 及环境检查](shoulder_roll_audit.json)
- [加速度链路图](acceleration_chain.png)
- [权重对照图](weight_comparison.png)
- [逐时刻双肩加速度数据](acceleration_chain.npz)
- [全部 40 组第二级解](counterfactual_solutions.npz)
- [22 项测试日志](tests.log)
