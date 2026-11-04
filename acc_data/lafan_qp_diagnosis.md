# LAFAN QP diagnosis and startup fixes — 2026-09-18

## Corrected diagnosis

Earlier ablation grouped all G rows after 24 as effort constraints. That was incorrect: these problems have 152 inequality rows. Rows 0:24 are wrench constraints, 24:82 total effort, and 82:152 predicted acceleration constraints. Removing rows 24 onward removed both effort and predicted acceleration constraints; attribution to actuator effort alone was unsupported.

For `lafan_tasks_updated.failure.npz`, independent SciPy HiGHS feasibility checks show:

- All constraints: infeasible.
- Remove only effort rows 24:82: still infeasible.
- Remove only predicted acceleration rows 82:152: feasible.

Initial joint positions are within actual Isaac limits. Both frame 0 and frame 1 fail at 0.02 s with the old initialization; neither diagnostic trial reported penetrating self contacts. The original first-frame trajectory increment is large but playback has not yet started at the initial failure.

## Changes

1. Place the lowest initial modeled sole corner on z=0 by translating the full reference trajectory once (walk1: -0.0214638 m). The original sole mesh had approximately 21/28 mm clearance, inconsistent with unconditional support initialization. Uses the same sole rectangle/depth as the wrench model; original PKL unchanged.
2. Measured joint braking bounds now recover gradually from an externally disturbed velocity outside the conservative stopping envelope. Next-step predicted joint position remains bounded inside the actual limits with margin; effort, dynamics and friction validation are unchanged.
3. Smooth playback speed over one second after warmup and compute neighboring reference samples from the same clock. No frames are removed.
4. QP result/error includes bound_violation, which was previously checked but omitted from the error message.

## Evidence and limits

- Unit checks: 15 contact/reference, 10 joint control, 14 acceleration-QP, 18 physical task tests passed (57 total across relevant suites).
- Ground alignment alone still fails at 0.02 s, with reduced velocity norm (3.08 vs 7.46); it is not a complete fix.
- Ground alignment + braking recovery: 0.1 s static startup passes. Without playback ramp, a longer trial fails at 1.14 s after motion starts.
- With all three changes, `lafan_ramp_3s.npz` passes the existing motion-segment validation: peak COM error 0.007794 m, peak servo RMSE 0.019539 rad, no sustained oscillation, no torque saturation. Final COM error 0.005460 m.
- A 20 s requested trial stops at 5.22 s: `lafan_fixed_20s.failure.npz`. It has NOT passed 20 s or the full 261 s clip. At failure commanded contact mode is double support, foot loads about 50.8/306.1 N, generalized velocity norm 31.12, with large QP residuals. The remaining contact/trajectory/servo behavior needs diagnosis; no specific remaining root cause is established.

The original 0.02 s failure is resolved in these tests. This is not proof of full-motion stability or hardware safety.

## Reference-vs-command audit

Using the 20 s trial's saved history and the same G1 rev_1_0 FK model:

| Simulation time | Actual vs original PKL joint RMSE (rad) | Actual vs integrated command RMSE (rad) |
| --- | --- | --- |
| 2.0 | 0.12477 | 0.00832 |
| 3.0 | 0.07639 | 0.00628 |
| 4.0 | 0.21644 | 0.02024 |
| 5.0 | 0.11846 | 0.00755 |
| 5.2 | 0.21031 | 0.14390 |

All 261 saved planned/commanded states from t=0 through 5.20 are double support. At t=5.20 (source frame 111), reference ankle heights are 0.0316/0.0613 m. The height-only planner compares each ankle to its initial reference height plus 0.04 m and still marks both feet as stance. Reference/live foot position differences are then 0.1096/0.0369 m. At failure t=5.22 left hip yaw velocity is -28.82 rad/s.

These observations support an inconsistent reference/contact plan and increasing divergence from the original motion. They do not isolate every cause of the later physical instability. Existing `motion_segment` pass criteria primarily certify tracking of the integrated joint command and modified COM target, not fidelity to the original PKL. Claims of PKL tracking success based on the earlier 3 s pass were too strong.
