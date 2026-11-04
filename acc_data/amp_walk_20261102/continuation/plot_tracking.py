from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
fig, axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for filename,label in [('baseline.npz','Before contact-force wiring'),('measured_load.failure.npz','Measured contact-force wiring'),('physics_velocity8.failure.npz','Diagnostic: 32/8 PhysX iterations')]:
    with np.load(p/filename) as d:
        t=d['history_time'].reshape(-1)
        q=d['history_q'].reshape(-1,36)
        ref=d['history_reference_q'].reshape(-1,36)
        axes[0].plot(t,q[:,1]-ref[:,1],label=label)
        axes[1].plot(t,q[:,2]-ref[:,2])
        axes[2].plot(t,d['history_pkl_joint_rmse'].reshape(-1))
axes[0].set_ylabel('Root lateral error (m)')
axes[1].set_ylabel('Root height error (m)')
axes[2].set_ylabel('Joint RMSE (rad)')
axes[2].axhline(.2,color='red',ls='--',label='Original fidelity threshold')
axes[2].legend(loc='upper left')
for ax in axes:
    ax.grid(alpha=.25)
    ax.axvspan(289/200,352/200,color='orange',alpha=.12)
axes[0].legend(loc='upper left')
axes[2].set_xlabel('Source / simulation time (s)')
axes[2].set_xlim(0,2.42)
fig.suptitle('Full AMP tracking remains unsuccessful: 4 / 29 verified landings\nShaded interval: fifth source swing')
fig.tight_layout()
fig.savefig(p/'tracking_comparison.png',dpi=160)
