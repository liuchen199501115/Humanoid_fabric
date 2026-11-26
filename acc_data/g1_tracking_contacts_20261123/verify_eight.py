"""Check the independent replay against the first eight-landing run."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
with np.load(OUT / 'landing8_y15.npz') as first, np.load(OUT / 'eight_verify.npz') as repeat:
    keys = sorted(k for k in first.files if k.startswith('history_')
                  and k != 'history_qp_wall_seconds')
    assert keys == sorted(k for k in repeat.files if k.startswith('history_')
                          and k != 'history_qp_wall_seconds')
    for key in keys:
        np.testing.assert_array_equal(first[key], repeat[key], err_msg=key)
    assert repeat['self_contact_events'].shape[0] == 0
    assert not repeat['self_collisions_disabled'].item()
    assert (repeat['history_qp_native_solve_count'] == 1).all()
    # The first eight sphere pairs are soft tasks, the remaining pairs are
    # the additional hard wrist/body cover used by the collision barrier.
    mask = repeat['history_time'] >= 1.85
    min_cover = float(repeat['history_collision_distance'][mask, :, 8:].min())
    assert min_cover >= .02
    idx = np.flatnonzero(np.isclose(repeat['history_time'], 3.1, atol=1e-8))[0]
    joint_rmse = float(repeat['history_pkl_joint_rmse'][idx, 0])
    root_error = float(repeat['history_pkl_root_position_error'][idx, 0])
    assert joint_rmse < .2 and root_error < .2

with np.load(OUT / 'prepared_landing8_y15.npz') as prepared:
    for name in ['landing8_y15', 'eight_verify']:
        with np.load(OUT / f'{name}.control_reference.npz') as reference:
            np.testing.assert_array_equal(prepared['control_q'], reference['control_q'])
            assert hashlib.sha256(reference['source_q'].tobytes()).hexdigest() == prepared['source_sha256'].item()
reference_hash = hashlib.sha256((OUT / 'prepared_landing8_y15.npz').read_bytes()).hexdigest()
assert reference_hash == json.loads((OUT / 'reference_inputs.json').read_text())['landing8_y15']['reference_sha256']
steps = json.loads((OUT / 'eight_verify.steps.json').read_text())
assert steps['measured_steps'] == 8 and steps['expected_steps'] == 29
assert steps['continuous'] and steps['landing_goal_policy'] == 'source_pose'
diagnostic = steps['next_step_diagnostics']
for key, value in [('required_clearance_m', .02), ('required_landing_hold_s', .05),
                   ('landing_position_tolerance_m', .04)]:
    assert diagnostic[key] == value
assert steps['pkl_fidelity']['tolerances'] == dict(joint_rmse_rad=.2, root_position_m=.2, root_orientation_rad=.35)
assert not steps['full_pkl_tracking']
report = dict(exact_history_fields=len(keys), excluded_history_fields=['history_qp_wall_seconds'],
              prepared_reference_sha256=reference_hash, reference_and_source_match=True,
              self_collision_enabled=True, self_contact_events=0,
              minimum_hard_cover_distance_after_1_85_s_m=min_cover,
              original_acceptance_limits=True, measured_steps=8, expected_steps=29,
              landing8=steps['landings'][-1], landing8_joint_rmse_rad=joint_rmse,
              landing8_root_position_error_m=root_error, full_tracking_passed=False)
(OUT / 'eight_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
