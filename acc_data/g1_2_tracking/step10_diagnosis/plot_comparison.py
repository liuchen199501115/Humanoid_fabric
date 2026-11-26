from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
runs=['baseline_probe_v2','com4_before10','state_com_xy4','state_com_xyz4']
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for name in runs:
 path=out/(name+'.npz')
 if not path.exists():continue
 with np.load(path,allow_pickle=False) as d:
  frames=d['history_frame'].ravel();q=d['history_q'].reshape(-1,36);r=d['history_reference_q'].reshape(-1,36)
  position=np.linalg.norm(q[:,:3]-r[:,:3],axis=1)
  velocity=np.linalg.norm((d['history_com_velocity']-d['history_com_target_velocity']).reshape(-1,3)[:,:2],axis=1)
  angle=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(r[:,3:7]).inv()).magnitude()
  for ax,y in zip(axes,[position,velocity,angle]):ax.plot(frames,y,label=name,linewidth=1.3)
axes[0].axhline(.2,color='red',linestyle='--');axes[2].axhline(.35,color='red',linestyle='--')
for ax,label in zip(axes,['Root position error (m)','COM horizontal velocity error (m/s)','Root orientation error (rad)']):
 ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_xlim(700,1100)
 for frame in [802,884,952,1013]:ax.axvline(frame,color='grey',alpha=.2)
axes[0].legend(fontsize=8);axes[-1].set_xlabel('Original source frame (vertical markers: planned landings 10--13)')
fig.suptitle('Step-ten diagnosis: original-source error and measured COM drift')
fig.tight_layout();fig.savefig(out/'tracking_comparison.png',dpi=160);plt.close(fig)
