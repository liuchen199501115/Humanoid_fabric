"""Plot original-source errors and waist margin for completed physical trials."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
runs=['response_zero','stop25_320','stop25_root500','stop25_pose500','stop25_pose_early','stop25_hard_root500']
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
for name in runs:
 path=out/(name+'.npz')
 if not path.exists():path=out/(name+'.failure.npz')
 if not path.exists():continue
 with np.load(path) as d:
  if 'history_q' not in d:continue
  t=d['history_time'].ravel();q=d['history_q'].reshape(-1,36);ref=d['history_reference_q'].reshape(-1,36)
  pos=np.linalg.norm(q[:,:3]-ref[:,:3],axis=-1)
  angle=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(ref[:,3:7]).inv()).magnitude()
  axes[0].plot(t,pos,label=name);axes[1].plot(t,angle);axes[2].plot(t,q[:,20])
axes[0].axhline(.2,color='red',linestyle='--');axes[0].set_ylabel('Root position error (m)')
axes[1].axhline(.35,color='red',linestyle='--');axes[1].set_ylabel('Root orientation error (rad)')
axes[2].axhline(.52,color='red',linestyle='--');axes[2].set_ylabel('Waist roll (rad)');axes[2].set_xlabel('Simulation time (s)')
for ax in axes:ax.grid(alpha=.2);ax.set_xlim(3.5,4.6)
axes[0].legend(fontsize=8,ncol=2)
fig.suptitle('State-based tracking experiments; all runs start at source frame zero')
fig.savefig(out/'tracking_comparison.png',dpi=160)
