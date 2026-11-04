from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
files=[('Previous 6-landing candidate',p.parent/'priority_followup/later_tails.failure.npz'),
       ('Vertical-feedback candidate',p/'vertical_feedback_verified.failure.npz')]
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
for label,file in files:
    with np.load(file) as d:
        t=d['history_time'].ravel(); mask=(t>=2.3)&(t<=3.35)
        feet=d['history_feet'].reshape(-1,2,7)
        axes[0].plot(t[mask],feet[mask,0,2],label=label)
        axes[1].plot(t[mask],d['history_pkl_root_position_error'].ravel()[mask],label=label)
        axes[2].plot(t[mask],d['history_ground_foot_forces'].reshape(-1,2,3)[mask,1,2],label=label)
        if label.startswith('Previous'):
            source=d['history_source_feet_reference'].reshape(-1,2,3)
            axes[0].plot(t[mask],source[mask,0,2],'k--',alpha=.65,label='Original source left ankle')
axes[0].set_ylabel('Left ankle height (m)');axes[0].legend(loc='upper right',fontsize=8)
axes[1].set_ylabel('Root position error (m)');axes[1].axhline(.2,color='r',linestyle='--',linewidth=1,label='Fidelity limit');axes[1].legend(fontsize=8)
axes[2].set_ylabel('Right ground force (N)');axes[2].set_xlabel('Simulation time (s)')
for ax in axes:
    ax.axvline(2.906,color='green',linestyle=':',alpha=.7);ax.grid(alpha=.2)
axes[0].set_title('7th landing confirmed at 2.906 s; full tracking still fails')
fig.savefig(p/'seventh_step_comparison.png',dpi=160)
