"""Plot measured tracking errors for completed shoulder-weight trials."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

OUT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent / 'closed_loop'
result = json.loads((OUT / 'results.json').read_text())
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
labels = ('Left shoulder roll error (rad)', 'Right shoulder roll error (rad)',
          'Full joint RMSE (rad)', 'Root position error (m)',
          'Root orientation error (rad)', 'COM velocity error norm (m/s)')
for row in result['trials']:
    if 'recording' not in row:
        continue
    with np.load(row['recording'], allow_pickle=False) as d:
        t = d['history_time'].ravel()
        q, r = d['history_q'].reshape(-1,36), d['history_reference_q'].reshape(-1,36)
        names = d['final_joint_order' if 'final_joint_order' in d else 'joint_names'].tolist()
        j = [names.index(s+'_shoulder_roll_joint') for s in ('left','right')]
        error = q[:,7:]-r[:,7:]
        ys = [error[:,j[0]], error[:,j[1]], np.sqrt(np.mean(error**2,axis=1)),
              np.linalg.norm(q[:,:3]-r[:,:3],axis=1),
              (Rotation.from_quat(q[:,3:7])*Rotation.from_quat(r[:,3:7]).inv()).magnitude(),
              np.linalg.norm((d['history_com_velocity']-d['history_com_target_velocity']).reshape(-1,3),axis=1)]
        mask = t >= 3.25
        for ax, y in zip(axes.ravel(),ys):
            ax.plot(t[mask], y[mask], label=row['run'], linewidth=1.2)
for ax, label in zip(axes.ravel(),labels):
    ax.set_ylabel(label)
    ax.set_xlabel('Simulation time (s)')
    ax.grid(alpha=.2)
    ax.axvline(3.3,color='gray',linestyle=':',alpha=.5)
    ax.axvline(3.58,color='gray',linestyle='--',alpha=.5)
for ax, threshold in ((axes[0,2],.2),(axes[1,0],.2),(axes[1,1],.35)):
    ax.axhline(threshold,color='red',linestyle='--',alpha=.5)
axes[0,0].legend(fontsize=8)
fig.suptitle('Fresh closed-loop Isaac trials | original PKL audit | unchanged thresholds')
fig.savefig(OUT/'tracking_comparison.png',dpi=160)
plt.close(fig)
print(OUT/'tracking_comparison.png')
