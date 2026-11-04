from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
for path,label in [(p.parent/'impact_followup/crouch_bounded.failure.npz','Previous five-contact experiment'),(p/'landing_tail.npz','New landing-tail / unloading experiment')]:
 d=np.load(path);t=d['history_time'];q=d['history_q'].reshape(-1,36);r=d['history_reference_q'].reshape(-1,36);dot=np.abs((q[:,3:7]*r[:,3:7]).sum(1))/(np.linalg.norm(q[:,3:7],axis=1)*np.linalg.norm(r[:,3:7],axis=1));angle=2*np.arccos(np.clip(dot,0,1))
 ax[0].plot(t,np.linalg.norm(q[:,:3]-r[:,:3],axis=1),label=label)
 ax[1].plot(t,angle)
 ax[2].plot(t,d['history_pkl_joint_rmse'].ravel())
for a,limit,title in zip(ax,[.2,.35,.2],['Root position error (m)','Root orientation error (rad)','Joint RMSE (rad)']):
 a.axhline(limit,color='red',ls='--');a.axvline(2.482,color='black',ls=':',alpha=.6);a.set_ylabel(title);a.grid(alpha=.2)
ax[0].legend();ax[2].set_xlabel('Simulation time (s)');ax[2].set_xlim(0,3.6)
fig.suptitle('Six source-pose landing events; strict PKL tracking still fails\nDotted line: sixth landing at 2.482 s; dashed lines: unchanged fidelity limits')
fig.tight_layout();fig.savefig(p/'tracking_comparison.png',dpi=160)
d=np.load(p/'landing_tail.npz');r=np.load(p/'landing_tail.source_feet.npz');t=d['history_time'];feet=d['history_feet'].reshape(-1,2,7);fig,ax=plt.subplots(figsize=(9,4))
ax.plot(t,r['reference_feet'][:,0,0],label='Original same-frame source ankle x');ax.plot(t,feet[:,0,0],label='Measured ankle x');goal=3.341783285140991;ax.axhline(goal,color='black',ls='--',label='Contact-onset ankle x');ax.axhspan(goal-.04,goal+.04,alpha=.15,color='grey',label='Old fixed +/- 4 cm window');ax.set_xlim(1.7,2.05);ax.set_ylim(3.25,3.43);ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Left ankle x (m)');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(p/'landing_goal_audit.png',dpi=160)
