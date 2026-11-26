"""Audit measured shoulder accelerations and replay two-stage QPs at fixed states.

Run with the humanoid_fabric Python environment and
PYTHONPATH=.runtime/trajectory_solvers. No simulation or controller changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


SHOULDERS = ("left_shoulder_roll_joint", "right_shoulder_roll_joint")


def forward_acceleration(times, velocities):
    """Row i predicts the interval [t_i, t_i+1], never the previous interval."""
    dt = np.diff(times)
    if len(times) != len(velocities) or not np.all(dt > 0):
        raise ValueError("Velocity samples must have strictly increasing times")
    return np.diff(velocities, axis=0) / dt[:, None]


def metric_delta(metric, active, indices, multiplier):
    """Congruence preserves PSD and gives selected diagonal weights the factor."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Weight multiplier must be finite and positive")
    factors = np.ones(len(active))
    factors[indices] = np.sqrt(multiplier)
    masked = metric * active[:, None] * active[None, :]
    return factors[:, None] * masked * factors[None, :] - masked


def solve_qp(H, h, E, e, G, b, lo, hi):
    """Independent Clarabel solve, validated in original unscaled units."""
    import clarabel
    C = sparse.csc_matrix(np.vstack((E, G, np.eye(len(h)))))
    lower = np.r_[e, np.full(len(G), -np.inf), lo]
    upper = np.r_[e, b, hi]
    row_scale = np.maximum(np.sqrt(C.multiply(C).sum(axis=1)).A.ravel(), 1e-8)
    Cs = sparse.diags(1 / row_scale) @ C
    ls, us = lower / row_scale, upper / row_scale
    eq = np.isfinite(ls) & np.isfinite(us) & (ls == us)
    up, down = np.isfinite(us) & ~eq, np.isfinite(ls) & ~eq
    A = sparse.vstack((Cs.tocsr()[eq], Cs.tocsr()[up], -Cs.tocsr()[down]), format="csc")
    rhs = np.r_[us[eq], us[up], -ls[down]]
    scale = max(float(np.diag(H).max()), 1.)
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.max_threads, settings.max_iter = False, 1, 200
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    result = clarabel.DefaultSolver(
        sparse.triu(sparse.csc_matrix((H + H.T) / (2 * scale)), format="csc"), h / scale,
        A, rhs, [clarabel.ZeroConeT(int(eq.sum())),
                 clarabel.NonnegativeConeT(int(up.sum() + down.sum()))], settings).solve()
    x = np.asarray(result.x)
    residual = float(max(0., np.max(lower - C @ x), np.max(C @ x - upper)))
    status = dict(status=str(result.status), constraint_residual=residual,
                  accepted=str(result.status) == "Solved" and residual < 1e-5)
    return x, status


def get_problem(d, prefix):
    return [d[prefix + key][0].astype(float) for key in
            ("H", "h", "A_eq", "b_eq", "G", "b", "lb", "ub")]


def solve_two_stage(first, final, delta_H, delta_h):
    """Recompute priority outputs, then rebuild only their output-lock targets."""
    H, h, E, e, G, b, lo, hi = first
    FH, fh, FE, fe, FG, fb, flo, fhi = final
    nbase = len(G)
    extra = len(FG) - nbase
    if extra <= 0 or extra % 2:
        raise ValueError("Expected one priority stage with paired inequality locks")
    for a, other in [(E, FE), (e, fe), (G, FG[:nbase]), (b, fb[:nbase]),
                     (lo, flo), (hi, fhi)]:
        np.testing.assert_array_equal(a, other)
    nlocks = extra // 2
    locks = FG[nbase:nbase + nlocks]
    np.testing.assert_array_equal(locks, -FG[nbase + nlocks:])
    tolerance = (fb[nbase:nbase + nlocks] + fb[nbase + nlocks:]) / 2
    if np.any(tolerance <= 0):
        raise ValueError("Expected positive stored priority tolerances")
    first_x, first_status = solve_qp(H + delta_H, h + delta_h, E, e, G, b, lo, hi)
    if not first_status["accepted"]:
        return None, dict(priority=first_status, accepted=False)
    achieved = locks @ first_x
    new_b = np.r_[b, achieved + tolerance, -achieved + tolerance]
    x, final_status = solve_qp(FH + delta_H, fh + delta_h,
                             FE, fe, FG, new_b, flo, fhi)
    return x, dict(priority=first_status, final=final_status,
                   accepted=final_status["accepted"],
                   lock_error=float(np.max(np.abs(locks @ x - achieved))),
                   lock_tolerance_max=float(tolerance.max()))


def task_metrics(d, x, offset, joints):
    acceleration = x[:35] + offset
    values = dict(shoulder_acceleration=acceleration[6 + joints].tolist(),
                  shoulder_torque=x[47 + joints].tolist(),
                  root_acceleration=acceleration[:6].tolist())
    for task in ("joint_posture", "center_of_mass", "feet_pose", "body_orientation",
                 "source_pelvis_orientation", "hand_pose"):
        A, bias, desired, active = (d[f"task_{task}_{k}"][0] for k in
                                    ("A", "bias", "desired", "active"))
        residual = (A @ acceleration + bias - desired) * active
        values[task + "_residual_rms"] = float(np.sqrt(np.mean(residual ** 2)))
        if task == "joint_posture":
            values["shoulder_residual"] = residual[joints].tolist()
            values["shoulder_residual_rms"] = float(np.sqrt(np.mean(residual[joints] ** 2)))
        if task == "center_of_mass":
            values["com_acceleration"] = (A @ acceleration + bias).tolist()
            values["com_residual"] = residual.tolist()
        if task == "feet_pose":
            values["feet_position_residual_rms"] = float(np.sqrt(np.mean(residual[[0,1,2,6,7,8]] ** 2)))
            values["feet_orientation_residual_rms"] = float(np.sqrt(np.mean(residual[[3,4,5,9,10,11]] ** 2)))
    return values


def snapshot_audit(path, joints, times, history_q, history_v, history_requested):
    with np.load(path, allow_pickle=False) as source:
        d = {key: source[key] for key in source.files if not key.startswith(("history_", "servo_"))}
    options = json.loads(str(d["controller_options"]))
    if not options["direct_wbc"] or "priority_level_1_H" in d:
        raise ValueError("This replay requires a direct-WBC single-priority-stage snapshot")
    offset = -d["correction"][0].astype(float)
    np.testing.assert_allclose(d["problem_solution"][0, :35] + offset,
                               d["predicted_acc"][0], atol=1e-5)
    np.testing.assert_allclose(d["problem_solution"][0, 47:], d["total_torque"][0], atol=1e-5)
    np.testing.assert_array_equal(d["task_joint_posture_A"][0], np.eye(35)[6:])
    assert d["task_joint_posture_active"][0, joints].all()
    i = int(np.argmin(abs(times - float(d["time"])) ))
    np.testing.assert_allclose(times[i], d["time"], atol=1e-9)
    np.testing.assert_array_equal(history_q[i], d["q"][0])
    np.testing.assert_array_equal(history_v[i], d["qd"][0])
    desired = d["task_joint_posture_desired"][0] - d["task_joint_posture_bias"][0]
    np.testing.assert_allclose(history_requested[i, joints], desired[joints], atol=2e-4)
    response_path = path.with_name(path.name.replace(".qp_", ".response_"))
    with np.load(response_path, allow_pickle=False) as r:
        observed = (r["qd"][0].astype(float) - d["qd"][0]) / float(r["dt"])
        torque_error = float(np.max(abs(r["applied_torque"] - d["total_torque"])))
        np.testing.assert_allclose(r["qd"][0], history_v[i + 1], atol=1e-6)
        np.testing.assert_allclose(float(r["dt"]), times[i + 1] - times[i], atol=1e-9)
    first, final = get_problem(d, "priority_level_0_"), get_problem(d, "problem_")
    A = d["task_joint_posture_A"][0].astype(float)
    J = np.c_[A, np.zeros((29, len(first[1]) - 35))]
    residual_offset = A @ offset - desired
    M, active = d["task_joint_posture_metric"][0], d["task_joint_posture_active"][0]
    stored_replay, status = solve_qp(*final)
    if not status["accepted"]:
        raise RuntimeError(f"Cannot reproduce saved final problem: {path}: {status}")
    row = dict(snapshot=str(path.resolve()), time_s=float(d["time"]),
               requested=desired[joints].tolist(), predicted=d["predicted_acc"][0, 6+joints].tolist(),
               observed=observed[6+joints].tolist(), applied_torque_max_error=torque_error,
               saved_final_replay=status,
               saved_final_replay_shoulder_difference=(stored_replay[6+joints]+offset[6+joints]
                                                       - d["predicted_acc"][0, 6+joints]).tolist(),
               cases=[])
    solution_rows = []
    cases = [("baseline", joints, 1.)] + [(side, indices, weight)
             for side, indices in (("left", joints[:1]), ("right", joints[1:]), ("both", joints))
             for weight in (2., 4., 8.)]
    for side, indices, weight in cases:
        delta = metric_delta(M, active, indices, weight)
        x, solve_status = solve_two_stage(first, final, J.T @ delta @ J,
                                          J.T @ delta @ residual_offset)
        case = dict(side=side, multiplier=weight, solver=solve_status)
        if solve_status["accepted"]:
            case.update(task_metrics(d, x, offset, joints))
            solution_rows.append(x)
        else:
            solution_rows.append(np.full(len(first[1]), np.nan))
        row["cases"].append(case)
    baseline = row["cases"][0]
    if not baseline["solver"]["accepted"]:
        raise RuntimeError(f"Cannot reproduce hierarchy at {path}")
    for case in row["cases"]:
        if case["solver"]["accepted"]:
            case["shoulder_residual_ratio_vs_replayed_baseline"] = (
                case["shoulder_residual_rms"] / baseline["shoulder_residual_rms"])
            case["com_residual_rms_change"] = (case["center_of_mass_residual_rms"]
                                              - baseline["center_of_mass_residual_rms"])
    row["hierarchy_replay_shoulder_difference"] = (
        np.array(baseline["shoulder_acceleration"]) - d["predicted_acc"][0, 6+joints]).tolist()
    if np.max(abs(np.array(row["hierarchy_replay_shoulder_difference"]))) > 1e-3:
        raise RuntimeError(f"Baseline shoulder replay differs by more than 1e-3 rad/s²: {path}")
    # Feasibility of exactly satisfying both shoulder requests, retaining all
    # original physical/auxiliary constraints, before priority output locks.
    H, h, E, e, G, b, lo, hi = first
    lp = linprog(np.zeros(len(h)), A_ub=G, b_ub=b,
                 A_eq=np.vstack((E, J[joints])), b_eq=np.r_[e, -residual_offset[joints]],
                 bounds=np.c_[lo, hi], method="highs")
    row["exact_shoulders_feasibility"] = dict(status=int(lp.status), message=lp.message)
    if lp.success:
        row["exact_shoulders_feasibility"]["constraint_residual"] = float(max(
            0., np.max(G @ lp.x - b), np.max(lo - lp.x), np.max(lp.x - hi),
            np.max(abs(E @ lp.x - e)), np.max(abs(J[joints] @ lp.x + residual_offset[joints]))))
    return row, np.asarray(solution_rows)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path,
                        default=root / "acc_data/g1_2_tracking/step10_diagnosis/baseline_probe_v2.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "acc_data/g1_shoulder_roll_20261112")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.recording, allow_pickle=False) as d:
        times = d["history_time"].astype(float)
        n = len(times)
        get = lambda key: d["history_" + key].reshape(n, -1).astype(float)
        names = d["final_joint_order"].tolist()
        joints = np.array([names.index(name) for name in SHOULDERS])
        q, v, reference = get("q"), get("qd"), get("reference_q")
        control_reference = get("control_reference_q")
        predicted = get("qdd_predicted")[:, 6:]
        requested = predicted - get("task_residual_joint_posture")
        measured = forward_acceleration(times, v)[:, 6:]
        commanded, measured_mode = get("commanded_mode")[:, 0], get("measured_mode")[:, 0]
        torque, limits = get("torque_qp_total"), get("effective_qp_effort_limits")
    intervals = []
    for lo, hi in ((3.3, 3.41), (3.41, 3.48), (3.48, 3.58), (3.58, 3.65), (3.65, 3.8)):
        mask = (times[:-1] >= lo - 1e-9) & (times[1:] <= hi + 1e-9)
        stable = (commanded[:-1] == commanded[1:]) & (measured_mode[:-1] == measured_mode[1:])
        for scope, ix in (("all", mask), ("unchanged_contact_labels", mask & stable)):
            if not ix.any():
                continue
            rms = lambda x: np.sqrt(np.mean(x[ix] ** 2, axis=0)).tolist()
            intervals.append(dict(start_s=lo, end_s=hi, scope=scope, samples=int(ix.sum()),
                position_rmse=rms((q[:-1, 7:] - reference[:-1, 7:])[:, joints]),
                reference_adjustment_rmse=rms((control_reference[:-1, 7:]-reference[:-1, 7:])[:, joints]),
                qp_minus_requested_rmse=rms((predicted[:-1]-requested[:-1])[:, joints]),
                observed_minus_qp_rmse=rms((measured-predicted[:-1])[:, joints]),
                peak_torque_fraction=np.max(abs(torque[:-1][ix][:, joints]) / limits[:-1][ix][:, joints], axis=0).tolist()))
    snapshots, solutions = [], {}
    for path in sorted(args.recording.parent.glob(args.recording.stem + ".qp_*.npz")):
        row, solution = snapshot_audit(path, joints, times, q, v, requested)
        snapshots.append(row)
        solutions[path.stem] = solution
        print(f"{row['time_s']:.3f}s: requested={row['requested']} predicted={row['predicted']} "
              f"observed={row['observed']}", flush=True)
        for case in row["cases"]:
            print(case["side"], case["multiplier"],
                  case.get("shoulder_residual_ratio_vs_replayed_baseline"),
                  case.get("com_residual_rms_change"), flush=True)
    if not snapshots:
        raise ValueError("No saved QP/response pairs found")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        environment = dict(returncode=gpu.returncode, output=(gpu.stdout+gpu.stderr).strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        environment = dict(error=str(error))
    report = dict(recording=str(args.recording.resolve()), shoulders=SHOULDERS,
        scope="Frozen measured-state, two-stage QP replay; no new closed-loop simulation",
        closed_loop_trials_run=0,
        alignment="Velocity difference over [t_i,t_i+1] compared with predicted acceleration at t_i",
        limitations=["Finite differences include contact impacts and discretization.",
                     "Unchanged contact labels do not exclude impacts or contact geometry changes.",
                     "Weight changes are frozen-state counterfactuals, not future tracking predictions.",
                     "Requests reconstructed from predicted acceleration minus saved task residual; validated at all snapshots."],
        environment=environment, intervals=intervals, snapshots=snapshots,
        source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [args.recording, Path(__file__),
                                 *sorted(args.recording.parent.glob(args.recording.stem + ".qp_*.npz")),
                                 *sorted(args.recording.parent.glob(args.recording.stem + ".response_*.npz"))]})
    (out / "shoulder_roll_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / "counterfactual_solutions.npz", **solutions)
    np.savez_compressed(out / "acceleration_chain.npz", time=times[:-1],
                        requested=requested[:-1, joints], predicted=predicted[:-1, joints],
                        measured=measured[:, joints], position_error=(q[:-1, 7:]-reference[:-1, 7:])[:, joints])
    plot_results(out, times, q, reference, requested, predicted, measured, joints, snapshots)
    print(f"Saved {out.resolve()}")


def plot_results(out, times, q, reference, requested, predicted, observed, joints, snapshots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask = (times[:-1] >= 3.3) & (times[:-1] <= 3.8)
    t = times[:-1][mask]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for col, joint in enumerate(joints):
        axes[0, col].set_title(SHOULDERS[col])
        axes[0, col].plot(t, (q[:-1, 7+joint]-reference[:-1, 7+joint])[mask])
        axes[0, col].set_ylabel("Position error (rad)")
        for y, label in ((requested[:-1, joint], "Requested"), (predicted[:-1, joint], "QP predicted"),
                         (observed[:, joint], "Observed next interval")):
            axes[1, col].plot(t, y[mask], label=label, linewidth=1)
        axes[1, col].set_ylabel("Acceleration (rad/s²)")
        axes[1, col].legend(fontsize=8)
        for y, label in ((predicted[:-1, joint]-requested[:-1, joint], "QP - request"),
                         (observed[:, joint]-predicted[:-1, joint], "Observed - QP")):
            axes[2, col].plot(t, y[mask], label=label, linewidth=1)
        axes[2, col].set_ylabel("Acceleration mismatch (rad/s²)")
        axes[2, col].set_xlabel("Simulation time (s)")
        axes[2, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.axvline(3.41, color="gray", linestyle="--", alpha=.5)
            ax.axvline(3.58, color="gray", linestyle="--", alpha=.5)
            ax.grid(alpha=.2)
    fig.suptitle("Measured shoulder-roll acceleration chain | dashed: landing 9 and 10")
    fig.savefig(out / "acceleration_chain.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for row in snapshots:
        cases = [c for c in row["cases"] if c["side"] in ("baseline", "both") and c["solver"]["accepted"]]
        x = [c["multiplier"] for c in cases]
        axes[0].plot(x, [c["shoulder_residual_rms"] for c in cases], "o-", label=f"{row['time_s']:.2f}s")
        axes[1].plot(x, [c["com_residual_rms_change"] for c in cases], "o-", label=f"{row['time_s']:.2f}s")
    axes[0].set_ylabel("Shoulder acceleration residual RMS (rad/s²)")
    axes[1].set_ylabel("Change in COM residual RMS (m/s²)")
    for ax in axes:
        ax.set_xlabel("Both shoulder-roll weights multiplier")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Same-state two-stage QP replay | not closed-loop tracking")
    fig.savefig(out / "weight_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
