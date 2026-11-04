from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
for name,label in [('default_verified.failure.npz','Default: four landings'),('crouch_bounded.failure.npz','Pelvis -2.5 cm: five landings')]:
    with np.load(p/name) as d:
        t=d['history_time'].reshape(-1);q=d['history_q'].reshape(-1,36);r=d['history_reference_q'].reshape(-1,36)
        quat=q[:,3:7]/np.linalg.norm(q[:,3:7],axis=1,keepdims=True);ref=r[:,3:7]/np.linalg.norm(r[:,3:7],axis=1,keepdims=True)
        angle=2*np.arccos(np.clip(abs((quat*ref).sum(1)),0,1))
        axes[0].plot(t,q[:,1]-r[:,1],label=label)
        axes[1].plot(t,q[:,2]-r[:,2])
        axes[2].plot(t,angle)
        axes[3].plot(t,d['history_pkl_joint_rmse'].reshape(-1))
for ax in axes:
    ax.axvline(1.864,color='black',ls=':',alpha=.6)
    ax.grid(alpha=.25)
axes[2].axhline(.35,color='red',ls='--',label='Original fidelity limit')
axes[3].axhline(.2,color='red',ls='--')
axes[0].set_ylabel('Root lateral error (m)');axes[1].set_ylabel('Root height error (m)')
axes[2].set_ylabel('Root angle error (rad)');axes[3].set_ylabel('Joint RMSE (rad)')
axes[0].legend();axes[2].legend();axes[3].set_xlabel('Simulation time (s)');axes[3].set_xlim(0,2.42)
fig.suptitle('Five landing events reproduced; stable full-clip tracking still fails\nDotted line: fifth landing confirmation at 1.864 s')
fig.tight_layout();fig.savefig(p/'five_landings_diagnostics.png',dpi=160)
