"""Audit nominal and measured sole clearance in a recorded source-step probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from analyze_contact_transfer import sole_clearances
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--stride', type=int, default=1, help='Audit every N recorded control samples')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('Stride must be positive')
    wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
    wp.init()
    torch.set_grad_enabled(False)
    fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
    model = fabric.dynamics_tasks.model
    indices = [model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
    with np.load(args.recording) as data:
        elapsed = np.asarray(data['history_source_step_elapsed']).ravel()
        ids = np.flatnonzero(elapsed >= 0.)
        if not len(ids):
            raise ValueError('No source swing was executed')
        ids = np.unique(np.r_[ids[::args.stride], ids[-1]])
        mode = int(np.asarray(data['history_commanded_mode']).ravel()[ids[0]])
        if mode not in (1,2):
            raise ValueError('First swing sample must be in single support')
        swing = 2-mode
        nominal, measured = [], []
        for i in ids:
            q = torch.as_tensor(data['history_whole_body_reference_q'][i])
            poses = model.snapshot(q)['transforms'][0,indices].cpu().numpy()
            nominal.append(sole_clearances(poses))
            measured.append(sole_clearances(data['history_feet'][i].reshape(2,7)))
        nominal, measured = np.asarray(nominal), np.asarray(measured)
        if not np.isfinite(nominal).all() or not np.isfinite(measured).all():
            raise ValueError('Nonfinite sole geometry')
        report = dict(scope='Recorded nominal/measured FK geometry; no tracking or contact certificate',
                      recording=str(args.recording), foot_order=['left','right'], swing_foot=['left','right'][swing],
                      stride=args.stride, sample_count=len(ids), elapsed_s=elapsed[ids].tolist(),
                      nominal_min_height_m=nominal.tolist(), measured_min_height_m=measured.tolist(),
                      minimum_nominal_swing_height_m=float(nominal[:,swing].min()),
                      nominal_ground_check_passed=bool((nominal[:,swing]>=-2e-5).all()),
                      numerical_tolerance_m=2e-5)
    output = args.recording.with_suffix('.ground.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(report['elapsed_s'], nominal[:,swing], label='Nominal swing sole')
    ax.plot(report['elapsed_s'], measured[:,swing], label='Measured swing sole')
    ax.axhline(0.,color='black',linestyle=':',label='Ground')
    ax.set(xlabel='Time since release [s]',ylabel='Lowest sole corner [m]')
    ax.legend();ax.grid(True);fig.tight_layout()
    fig.savefig(args.recording.with_suffix('.ground.png'))
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('elapsed_s','nominal_min_height_m','measured_min_height_m')},indent=2))


if __name__ == '__main__':
    main()
