# Experimental ReferenceManager

Run from the repository root:

```bash
python examples/test_g1.py --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-manager --direct-wbc --duration 35 --output-path acc_data/lafan_manager_trial.npz
```

This is an opt-in, one-robot, flat-ground/non-flight prototype. The default controller path is unchanged unless `--reference-manager` is supplied. It requires the dynamics-aware profile and a positive playback ramp; dedicated fixed-pose/support-cycle experiments cannot be combined with it.

## Behavior

- Offline candidate contact plan uses lowest modeled sole-corner height and ankle translational speed, plus a 60 ms confirmation interval. Thresholds (25 mm / 0.20 m/s) are initial experiment values. Frames where both feet fail the candidate criteria are flagged as ambiguous; the lower foot is retained for this non-flight prototype. These labels are not verified ground truth.
- One scheduler owns playback time and commanded contacts. Existing force detection is an observation only. Existing contact-switch logic is bypassed for this path.
- Before an anticipated single-support event, decelerate and move the COM target toward the measured support anchor. Release requires observed support with >=80% total load, >=100 N total load, confirmed for three controller samples, plus a capture point inside the support rectangle and COM speed <=0.06 m/s. These are quasi-static experimental gates, not a general dynamic-walking criterion.
- During `PREPARE_STEP`, the mocap unloading diagnostic can request zero outgoing normal force as a soft objective. The full rigid-body WBC determines the remaining wrench components while enforcing physical constraints.
- Latch the pending event while slowing, so a changing braking-distance threshold cannot cause phase chatter.
- A swing foot must first clear the floor by 10 mm. Touchdown requires sole clearance <=2 mm and >=5 N measured load for three samples. The subsequent `SETTLE` phase requires both feet to carry >=20 N for 0.3 seconds before the next transfer. If touchdown is unconfirmed, brake ahead of the next source event.
- At takeoff, start a two-second wall-time trajectory from the measured swing-foot pose and twist. Adapt the source step relative to the measured support foot, with forward/lateral displacement capped at 0.16/0.18 m in support-heading coordinates. A quintic trajectory supplies position, velocity and acceleration, with 0.06 m clearance; orientation is interpolated on SO(3). On flat ground the nominal final foot is level with ankle Z = 0.035 m. A bounded additional 0.04 m downward reference search over 1.5 seconds addresses incomplete touchdown. These are target adjustments, not physical state resets or a reachability guarantee.
- The managed posture blends toward knees bent by at least 0.5 rad, lowers the COM target by 0.06 m, and keeps hand gestures relative to the adjusted COM. Raw PKL fidelity is logged separately because this deliberately adapts the gait.
- Managed contacts include vertical and rotational stabilization and coupled rectangular-foot yaw/friction constraints. The latter reject combinations of forces and moments that independent yaw limits can incorrectly admit; see [Caron et al., 2015](https://arxiv.org/abs/1501.04719).
- Phase waiting has a six-second timeout; lost support or event overshoot aborts and records a reference failure snapshot. There is no general recovery controller in this prototype.
- Grounding is a one-time global Z offset; live robot states are never reset. With `--direct-wbc`, the actuator receives QP torque plus measured-velocity damping at 500 Hz. Position integration is only a one-step diagnostic prediction, not an actuator command. Physical joint-position/braking bounds still apply; the integrated-position command-speed bound belongs to the other control path.

## Outputs

- Visual and headless runs both advance exactly one 2 ms physics step per torque
  update. The window renders separately at 50 Hz; it does not call the
  multi-physics-step `World.step(render=True)` path. Each update checks elapsed
  physics time, and completed files record `physics_elapsed_seconds`,
  `physics_step_seconds`, and `rendering_step_seconds`. A 35 s visual compact
  regression matches the headless state and torque histories exactly, with
  seven alternating settled touchdowns. This does not validate source-stride
  tracking.
- `.contact_plan.npz`: candidate modes (1 left / 2 right / 3 double), height, speed, ambiguous flags, FPS.
- Final or `.failure.npz`: actual, reference, integrated commands, phase, rate, waiting time and contact histories.
- `.reference_failure.npz`: reference-state-machine failure with the history up to the failure.
- `history_pkl_joint_rmse` / `history_pkl_root_position_error`: fidelity at the current retimed, ground-aligned source frame; distinct from internal servo error.
- `peak_pkl_joint_rmse` and root error are reported separately. The existing controller pass criteria do not certify PKL fidelity or complete gait success.
- `history_managed_feet_position` records the foot target after landing overrides; compare with `history_feet`, source-frame progress and measured foot loads.
- `reference_status` distinguishes `waiting_for_contact`, `advancing`, `initializing`, and `source_finished`. Reaching the source endpoint is a playback statement, not a physical tracking certificate.

## Scope still missing

No validated contact annotations, heel/toe contact model, full kinematic/dynamic reachability solver, or general emergency recovery. COM transfer still uses a bounded reference generator and support heuristic. A longer run without QP failure is not success if motion stalls or reference errors grow. Inspect progress and errors together:

```bash
python examples/analyze_reference_walk.py acc_data/lafan_manager_trial.npz \
  --output acc_data/lafan_manager_trial.audit.json
```

The audit requires a completed, controller-valid run, at least four alternating touchdowns that finish load settling, >=0.2 m forward travel, and <=0.02 m stance-foot XY slip after initialization. Even a passing finite segment does not certify the entire PKL.


## Full-source tracking experiments

### Experimental toe-off support

`--toe-transfer-load-fraction` optionally sets the incoming-load threshold for
rolling (default 0.8, accepted range 0.5 to 1). It does not change the manager's
0.8 takeoff gate or measured toe-line geometry/force confirmation. A 0.65 trial
with pelvis tracking rolls the old foot at 6.15 s, but still stalls before the
second swing. It is not a validated complete gait configuration.

`--toe-off` requires `--dynamic-reference` and enables contact diagnostics.
During double support, the outgoing foot can change from a rigid sole to a toe
line after 60 ms of incoming-foot load >=80% of total (total >=100 N), with both
toe corners actually loaded. The loaded points must span the modeled 5 cm line;
a single loaded corner does not qualify. This confirmation persists across the
one-tick `SETTLE` to `TRACK` transition.

The toe midpoint position, roll and yaw remain constrained. Pitch is released,
and its dual contact moment is fixed to zero. A bounded soft pitch task lifts
the heel toward 0.25 rad over one second. The Jacobian is shifted to the toe
point, with the lever arm and centrifugal bias included. The wrench cone uses
a zero-length, 5 cm wide line, expressed in a fixed horizontal toe frame.
Dynamics, effort, friction, joint limits and measured support checks remain.
The next swing must also have a feasible COM preview before the release gate
can accumulate confirmation; a feasible capture point alone is insufficient.

This is an opt-in rigid-line experiment, not an impact/compliance model, a
general heel/toe contact estimator, or a validated full-clip controller.
Existing compact walking and the default full-sole model are unchanged.

For this option, `history_contact_wrench` contains the QP coordinates: world
ankle wrenches for full soles and fixed-frame toe wrenches for an active toe
line. Use `history_contact_wrench_world_ankle` for consistently expressed
wrenches. `history_toe_foot`, `history_toe_anchor`, `history_toe_frame`, and
`history_contact_row_mask` identify the representation; `-1` means no toe line.
`history_preview_release_ready` records the next-step planning gate.

The 50 Hz trial enters toe support at 6.16 s and starts its second swing at
7.36 s, then fails a joint acceleration bound at 7.64 s. Maximum measured toe
anchor position error during that transfer is 3.04 mm. A normal-force-cap ramp
was tested and removed after an earlier failure; it is not part of this option.
A 100 Hz trial also fails earlier. Neither experiment validates full tracking.
The unchanged compact path passes its separate 35 s physical regression.

### Source trajectory and completion

The successful compact-step segment does not follow the original global path.
`--reference-footsteps source` uses source FK foot poses and COM on a shared
retimed clock, without stride clipping. It retains source sole orientation and
smoothly corrects terminal sole clearance by at most 4 cm. A quintic offset joins
the initial COM target to the previous reference, and the swing-foot position to
its measured pose/twist, then fades to zero at the original source endpoint.
The offset matches position, velocity and acceleration at both ends; this does
not smooth every interpolation knot in the source or prove dynamic feasibility.
Source placement is experimental and has not passed full-clip tracking.

Source candidate contacts use the speed of a near-ground sole corner, rather
than ankle speed: a rolling foot can have a stationary corner and a moving ankle.
On this clip the ambiguous count changes from 514 to 28. This is improved
kinematic inference, not measured ground truth, and the controller still uses
rigid whole-foot support constraints.

The former reduced-model preview and landing-stop planners have been removed. Full-body source references and the contact-constrained WBC are retained. Historical experiment descriptions below are not current CLI instructions.

### Optional task experiments (not validated full tracking)

- `--early-touchdown` separates physical contact activation from load settling.
  After swing clearance, a sole with all modeled corners within 2 mm of the
  floor and at least 5 N measured load can enter double support immediately.
  `SETTLE` still requires both feet to carry at least 20 N for the configured
  duration; this does not authorize the next takeoff. Partial-foot contacts
  retain the existing confirmation path. In a 500 Hz diagnostic, the previous
  60 ms delay left an already loaded flat sole modeled as a free swing foot.
  The new 50 Hz trial enters double support earlier but still fails at 6.10 s
  near the waist pitch limit. This option remains off by default.
- `--flat-foot-spawn` prepares a nearby initial pose with both soles horizontal,
  keeping initial foot XY/yaw and root XY/orientation. It solves for root height
  and leg joints within URDF limits, rejects changes over 5 cm root height or
  0.35 rad per joint, and leaves the source clip and upper-body pose unchanged.
  This is a one-time spawn operation, never a live-state reset. It requires a
  positive warmup and does not establish full-body static/dynamic feasibility.
- `--joint-limit-time-constant .2` adds a measured velocity bound proportional
  to remaining joint travel, ramped in after warmup. Existing physical bounds
  stay in force. It requests earlier braking than the 5 rad/s² recovery fallback
  when a measured velocity has left the stopping envelope. The original-stride
  trial still failed at 6.32 s, so this option is not a validated fix.
- `--source-pelvis-weight 1` adds a soft pelvis orientation task alongside the
  existing source torso task, using the same retimed reference and derivatives.
  It prevents satisfying that task solely by compensating pelvis rotation at
  the waist. Default 0 retains the existing task set.
- `--posture-metric-scale` exposes the existing managed joint posture metric
  multiplier. It changes a soft objective, not joint limits or the PKL.
- `--balance-priority` performs two QP stages with the same physical constraints:
  first optimize the COM task; then lock its achieved acceleration while solving
  the complete tracking objective. Infeasible COM requests remain soft in stage
  one. Stage-one feasibility and the output lock are separately checked and
  recorded as `priority_stage_constraint_error` / `priority_lock_error`.
  Numerical locks do not certify optimality of the first solve or real-world
  balance. These options are disabled by default.
- `--transfer-balance-priority` applies that COM priority only during
  `PREPARE_STEP` after a dynamic swing and load settling, leaving first startup
  and swing objectives unchanged. It excludes `--balance-priority` and requires
  dynamic reference. `history_balance_priority_active` identifies active
  updates; priority errors are zero when inactive. The 500 Hz ADMM trial starts
  the second swing at 6.534 s but fails right-knee bounds at 7.060 s; full
  tracking has not passed.
- `--swing-height-priority` uses only the COM z row as the first-stage objective
  during dynamic single support. Horizontal COM coordinates remain in the
  ordinary weighted solve with the swing-foot tasks. It can be combined with
  transfer priority, is disabled by default, and requires physical validation.
- `--qp-solver osqp` selects an optional CPU backend (requires `osqp`/`scipy`).
  Physical checks remain mandatory; explicit infeasibility is rejected.
  Native OSQP has not resolved the current physical walking failures.

See `acc_data/lafan_full_tracking_followup.json` for failed physical trials.
The 500 Hz inverse-dynamics refresh experiment in that report was removed after
it failed; there is no `--refresh-inverse-dynamics` option in the current code.

### Contact diagnostics

QP failure files also contain `physical_acc_lower` / `physical_acc_upper`,
which bound measured-state predicted acceleration, separately from command
acceleration bounds. The failure log names the worst violating joint together
with position, velocity, predicted acceleration and bounds when applicable.

`--diagnose-contacts` enables actual PhysX foot-ground contact buffers, saved as
`history_contact_{normal_force,point,normal,distance,count,start}`. Buffer counts
and start indices must be used; unused entries are not contacts. This flag only
records observations. It does not change the active constraint model.

```bash
python examples/analyze_reference_contacts.py acc_data/lafan_source_bridge_12s.failure.npz \
  --output acc_data/lafan_source_bridge_12s.contacts.json
```

The measured source trial previously jumped its COM target by 0.1100 m at first
takeoff (5.66 s). The bridge removes that discontinuity, but this trial still
fails at 6.64 s. Contact points show heel-only support during swing; the QP still
assumes a rigid sole. Full-source tracking therefore remains unvalidated.
`--swing-metric-scale` is an explicit weight-ablation option; increasing a soft
weight does not establish task feasibility.
The weight-10 dynamic trial lands once, but after the double-support polygon fix
it stops at 7.58 s when the old support is lost before the next release gate.
The half-speed trial fails a joint acceleration bound at 1.62 s, before swing.
Neither trial passes full-source tracking. See `examples/G1_TRACKING.md` for
artifacts and the separately passing compact-step regression.

`--until-motion-end` treats `--duration` as a maximum simulation budget. Completion
requires reaching the final source frame and 0.5 seconds of double support with
both foot loads >=20 N and COM speed <=0.06 m/s. Exhausting the time budget is a
failure even if the controller stayed upright. A `.tracking.json` report also
requires coverage from frame 0 and compares actual poses directly with the
original, ground-aligned, retimed PKL. Initial acceptance limits are peak joint
RMSE <=0.2 rad, root position error <=0.2 m, root orientation error <=0.35 rad.
No per-frame root alignment or path rescaling is applied to this audit.

Audit an existing recording independently:

```bash
python examples/analyze_motion_tracking.py acc_data/lafan_walk_phase_load_35s.npz \
  --source-frames 7839 --output acc_data/lafan_walk_phase_load_35s.tracking.json
```

That 35-second compact run passes its walking audit but fails full-source coverage
and PKL fidelity. The entire 7839-frame LAFAN source covers about 261 seconds and
156 m of accumulated root-path length, including turns and lateral/backward steps.
