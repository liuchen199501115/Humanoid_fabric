"""Run one isolated full-source trial; never overwrite prior evidence."""
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
name, *options = sys.argv[1:]
path = OUT / (name + '.npz')
if path.exists() or path.with_suffix('.failure.npz').exists():
    raise FileExistsError('Refusing to overwrite an existing trajectory or prepared reference: '+str(path))
command = [sys.executable, '-u', '-c', "import runpy; runpy.run_path('examples/test_g1.py',run_name='__main__')", '--tracking-profile',
           'amp-full', '--output-path', str(path),
           '--qp-snapshot-times', '3.6', '3.7', '3.8', '3.9', '4.0', *options]
with path.with_suffix('.command.json').open('x') as stream:
    json.dump(command, stream, indent=2)
sources = ['acc_data/g1_tracking_contacts_20261123/prepare_landing8.py',
           'acc_data/g1_tracking_complete_20261123/prepare_force_com.py',
           'acc_data/g1_tracking_complete_20261123/prepare_cop_com.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_smooth.py',
           'acc_data/g1_tracking_complete_20261123/prepare_swing10_smooth.py',
           'acc_data/g1_tracking_complete_20261123/prepare_support10_early.py',
           'acc_data/g1_tracking_complete_20261123/prepare_com_continue.py',
           'acc_data/g1_tracking_complete_20261123/prepare_early_landing9.py',
           'acc_data/g1_tracking_complete_20261123/prepare_early_landing9_full.py',
           'acc_data/g1_tracking_complete_20261123/prepare_early9_smooth10_full.py',
           'acc_data/g1_tracking_complete_20261123/prepare_upper_momentum.py',
           'acc_data/g1_tracking_complete_20261123/prepare_upper_momentum_b20_v1.py',
           'acc_data/g1_tracking_complete_20261123/prepare_upper_dynamics.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_upper_dynamics.py',
           'acc_data/g1_tracking_complete_20261123/prepare_freeleg_dynamics.py',
           'acc_data/g1_tracking_complete_20261123/scale_freeleg_candidate.py',
           'acc_data/g1_tracking_complete_20261123/prepare_freeleg_waist.py',
           'acc_data/g1_tracking_complete_20261123/prepare_freeleg_clearance.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_upper_dynamics_v1.py',
           'acc_data/g1_tracking_complete_20261123/refine_root_upper_ik.py',
           'acc_data/g1_tracking_complete_20261123/prepare_stop_margin.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_roll.py',
           'acc_data/g1_tracking_complete_20261123/prepare_early_landing10.py',
           'acc_data/g1_tracking_complete_20261123/audit_upper_dynamics.py',
           'acc_data/g1_tracking_complete_20261123/reference_com_ik.py',
           'acc_data/g1_tracking_complete_20261123/audit_reference_momentum.py',
           'acc_data/g1_tracking_complete_20261123/prepare_vertical_com_down.py',
           'acc_data/g1_tracking_complete_20261123/numpy_centroidal_geometry.py',
           'acc_data/g1_tracking_complete_20261123/audit_reference_com.py',
           'acc_data/g1_tracking_complete_20261123/audit_candidate_vertical.py',
           'acc_data/g1_tracking_complete_20261123/run_trial.py',
           'acc_data/g1_tracking_complete_20261123/prepare_advance_swing10_xy.py',
           'acc_data/g1_tracking_complete_20261123/prepare_clearance10.py',
           'acc_data/g1_tracking_complete_20261123/prepare_lower_swing10.py',
           'acc_data/g1_tracking_complete_20261123/prepare_advance_swing9.py',
           'acc_data/g1_tracking_complete_20261123/prepare_advance_swing10.py',
           'acc_data/g1_tracking_complete_20261123/prepare_support9_x.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_increment.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_continue.py',
           'acc_data/g1_tracking_complete_20261123/prepare_support9.py',
           'acc_data/g1_tracking_complete_20261123/prepare_support7.py',
           'acc_data/g1_tracking_complete_20261123/prepare_root_support.py',
           'acc_data/g1_tracking_complete_20261123/prepare_combined_root_support.py',
           'acc_data/amp_walk_20261103/feedback_followup/root_balance_offset.py',
           'acc_data/amp_walk_20261103/feedback_followup/early_resume_clock.py',
           'acc_data/amp_walk_20261103/feedback_followup/landing_offset.py',
           'src/fabrics_sim/reference/convex_collision.py',
           'src/fabrics_sim/solver/collision_barrier.py',
           'examples/test_g1.py', 'src/fabrics_sim/solver/daqp_backend.py',
           'examples/prepare_g1_landing_reference.py',
           'acc_data/amp_walk_20261103/feedback_followup/numpy_leg_kinematics.py',
           'acc_data/amp_walk_20261103/feedback_followup/measured_stance_reference.py',
           'acc_data/amp_walk_20261103/feedback_followup/bounded_leg_ik.py',
           'acc_data/amp_walk_20261103/feedback_followup/landing_pause_clock.py',
           'examples/run_amp_single_tracking.sh',
           'examples/g1_priority_tracking.py',
           'examples/g1_unload_timing.py',
           'examples/g1_joint_braking.py',
           'examples/g1_joint_stop_mode.py',
           'examples/g1_late_clock.py',
           'examples/g1_torque_continuity.py',
           'acc_data/amp_walk_20261103/feedback_followup/smooth_clock.py',
           'src/fabrics_sim/solver/joint_stop.py',
           'src/fabrics_sim/solver/AccelerationQP.py',
           'acc_data/amp_walk_20261103/feedback_followup/unload_qp.py',
           'src/fabrics_sim/fabrics/priority.py',
           'src/fabrics_sim/fabrics/dynamics_aware.py',
           'src/fabrics_sim/fabrics/g1_pose_fabric.py',
           'src/fabrics_sim/solver/active_set_predictor.py',
           'examples/g1_soft_tracking_tube.py', 'examples/g1_soft_joint_braking.py']
path.with_suffix('.sources.json').write_text(json.dumps({s: hashlib.sha256(
    (ROOT / s).read_bytes()).hexdigest() for s in sources}, indent=2))
for source in sources:
    archived = OUT/(name+'.source')/source
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/source, archived)
env = dict(os.environ, PYTHONPATH='src:examples:.runtime/qp_solvers',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
timed_out = False
with path.with_suffix('.log').open('w') as log:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
report = dict(name=name, options=options, returncode=code, timed_out=timed_out)
for suffix in ('steps', 'tracking'):
    p = path.with_suffix('.' + suffix + '.json')
    if p.exists():
        report[suffix] = json.loads(p.read_text())
path.with_suffix('.result.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report.get(k) for k in ('name', 'returncode', 'timed_out')}, indent=2), flush=True)
