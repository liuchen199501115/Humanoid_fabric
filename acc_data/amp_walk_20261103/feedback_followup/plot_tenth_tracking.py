from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
names=['nine_anticipate008_xy80','nine_contact_gate_stance','nine_contact_tail5','nine_tail4_lead3','nine_tail4_lead3_arms']
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for name in names:
 path=p/(name+'.npz')
 if not path.exists():path=p/(name+'.failure.npz')
 if not path.exists():continue
 with np.load(path) as d:
  t=d['history_time'];mask=t>=3.8;feet=d['history_feet'].reshape(-1,2,7);source=d['history_source_feet_reference'];error=np.linalg.norm(feet[:,1,:3]-source[:,1],axis=1);load=d['history_ground_foot_forces'][:,1,2];joint=np.sqrt(np.mean((d['history_q'][:,0,7:]-d['history_reference_q'][:,0,7:])**2,axis=1))
  line,=axes[0].plot(t[mask],error[mask],label=name.removeprefix('nine_'));color=line.get_color();axes[1].plot(t[mask],load[mask],color=color);axes[2].plot(t[mask],joint[mask],color=color)
axes[0].axhline(.04,color='black',ls='--',label='Original landing limit: 4 cm');axes[0].set_ylabel('Right foot 3D error (m)');axes[0].set_ylim(0,.18);axes[0].legend(fontsize=8,ncol=2)
axes[1].axhline(20,color='black',ls='--');axes[1].set_ylabel('Measured right load (N)')
axes[2].axhline(.2,color='black',ls='--');axes[2].set_ylabel('Source joint RMSE (rad)');axes[2].set_xlabel('Physical time (s)')
for ax in axes:ax.grid(alpha=.25)
axes[0].set_title('Tenth step diagnostics — original source targets and unchanged acceptance limits')
fig.tight_layout();fig.savefig(p/'tenth_tracking_comparison.png',dpi=160)
