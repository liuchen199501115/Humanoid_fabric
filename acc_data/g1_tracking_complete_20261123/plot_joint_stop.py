"""Plot independently audited source error and the measured waist stop."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
with np.load(OUT / 'ten_verify.npz') as a:
    t = a['history_time']
    j = a['final_joint_order'].tolist().index('waist_roll_joint')
    q = a['history_q'][:, 0]
    v = a['history_qd'][:, 0]
    target = a['history_reference_q'][:, 0]
    error = (Rotation.from_quat(q[:, 3:7]) * Rotation.from_quat(target[:, 3:7]).inv()).magnitude()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, error, label='Root orientation error vs original PKL')
    axes[0].axhline(.35, color='red', ls='--', label='Original acceptance limit')
    axes[0].set_ylabel('rad')
    axes[1].plot(t, q[:, 7+j], label='Measured waist roll')
    axes[1].plot(t, target[:, 7+j], label='Original PKL waist roll')
    axes[1].axhline(a['joint_position_limits'][0, j, 1], color='red', ls='--', label='Physical joint stop')
    axes[1].set_ylabel('rad')
    axes[2].plot(t, a['history_qdd_qp'][:, 0, 6+j], label='QP predicted waist acceleration')
    axes[2].plot(t[:-1], np.diff(v[:, 6+j])/np.diff(t), label='Measured acceleration', alpha=.75)
    axes[2].set_ylabel('rad/s²')
    axes[2].set_ylim(-250, 300)
    axes[2].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.axvline(3.798, color='gray', ls=':', label='Tenth landing / waist stop' if ax is axes[0] else None)
        ax.set_xlim(3.60, t[-1])
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Verified 10/29 tracking: unmodeled waist-stop reaction precedes root-angle failure')
    fig.savefig(OUT / 'joint_stop_diagnosis.png', dpi=170)
