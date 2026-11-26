"""Read-only physical traces around landing 12 and the attempted landing 13."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
p=Path(__file__).parent
fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
for name,label in [('rolling_contact','12-step baseline'),('thirteenth_support_knee20','Support-knee barrier'),('thirteenth_late_upper_braking200','Late soft braking')]:
    path=p/(name+'.npz')
    if not path.exists():path=p/(name+'.failure.npz')
    if not path.exists():continue
    with np.load(path) as d:
        t=d['history_time'].ravel();q=d['history_q'].reshape(-1,36);ref=d['history_reference_q'].reshape(-1,36);keep=t>=4.12
        angle=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(ref[:,3:7]).inv()).magnitude()
        position=np.linalg.norm(q[:,:3]-ref[:,:3],axis=1)
        forces=d['history_foot_forces'].reshape(-1,2,3)[:,:,2]
        for ax,y in zip(axes,[q[:,20],angle,position,forces[:,1]]):ax.plot(t[keep],y[keep],label=label,linewidth=1.4)
for ax,limit in zip(axes,[.52,.35,.2,None]):
    if limit is not None:ax.axhline(limit,color='red',linestyle='--',linewidth=1,label='Native/audit limit')
    ax.grid(alpha=.25);ax.axvline(4.216,color='gray',linestyle=':',linewidth=1)
for ax,label in zip(axes,['Waist roll (rad)','Raw root angle error (rad)','Raw root position error (m)','Right normal load (N)']):ax.set_ylabel(label)
axes[0].legend(loc='upper left',fontsize=8)
axes[-1].set_xlabel('Simulation time (s); dotted line = baseline landing 12 verification')
fig.suptitle('Step 13 remains unverified: native waist stop and tracking errors')
fig.savefig(p/'step13_failure_trace.png',dpi=160)
print(p/'step13_failure_trace.png')
