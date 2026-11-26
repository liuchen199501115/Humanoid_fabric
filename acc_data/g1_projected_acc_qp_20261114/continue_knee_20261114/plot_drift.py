"""Measured comparison of original baseline and support interventions."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
paths={'baseline':root.parent/'continue_20261114_0338/baseline_traced.failure.npz',
       'knee >= 0.15 (barrier)':root/'knee15.failure.npz',
       'knee >= 0.25 (barrier)':root/'knee25.failure.npz',
       'contact velocity gain 8':root/'contact_damping.failure.npz'}
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
for label,path in paths.items():
 with np.load(path) as a:
  t=a['history_time'];q=a['history_q'][:,0];v=a['history_qd'][:,0]
  error=q[:,:3]-a['history_reference_q'][:,0,:3]
  axes[0].plot(t,np.linalg.norm(error,axis=1),label=label)
  axes[1].plot(t,abs(v[:,6:]).max(axis=1))
  axes[2].plot(t,q[:,10])
axes[0].axhline(.2,color='black',ls='--',lw=1,label='Original root tolerance')
axes[0].legend(fontsize=8,ncol=2)
for ax,label in zip(axes,['Original-source root error (m)','Peak absolute joint speed (rad/s)','Left knee angle (rad)']):
 ax.set_ylabel(label);ax.grid(alpha=.25);ax.set_xlim(.9,1.36)
axes[-1].set_xlabel('Simulation time (s)')
fig.suptitle('Support-knee protection changes knee motion, but does not resolve root drift')
fig.savefig(root/'drift_comparison.png',dpi=160)
