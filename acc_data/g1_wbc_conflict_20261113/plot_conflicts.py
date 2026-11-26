import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'conflicts.json').read_text())
t=np.array([s['time_s'] for s in r['snapshots']])
fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for j,axis in enumerate('xyz'):
    bounds=[s['com_feasibility_without_priority_locks'][axis] for s in r['snapshots']]
    lo=np.array([b['minimum']['acceleration_mps2'] for b in bounds])
    hi=np.array([b['maximum']['acceleration_mps2'] for b in bounds])
    axes[j].vlines(t,lo,hi,color='gray',alpha=.5,linewidth=5,label='Feasible scalar range')
    axes[j].plot(t,[b['requested_mps2'] for b in bounds],'o-',label='Requested')
    axes[j].plot(t,[s['cases'][0]['com_acceleration'][j] for s in r['snapshots']],'s-',label='Baseline QP')
    axes[j].set(xlabel='Simulation time (s)',ylabel=f'COM {axis} acceleration (m/s²)')
    axes[j].grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Same-state WBC constraints | each scalar range optimized separately, not jointly attainable')
fig.savefig(OUT/'com_feasible_ranges.png',dpi=160);plt.close(fig)

labels=[f"{c['group']} ×{c['scale']}" for c in r['candidates']]
values=np.array([[(1-v)*100 for v in c['com_residual_ratios']] for c in r['candidates']])
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
im=ax.imshow(values,cmap='RdYlGn',vmin=-5,vmax=5,aspect='auto')
ax.set_xticks(range(len(t)),[f'{v:.3f}' for v in t]);ax.set_yticks(range(len(labels)),labels)
for i in range(len(labels)):
    for j in range(len(t)):ax.text(j,i,f'{values[i,j]:+.2f}%',ha='center',va='center',fontsize=8)
ax.set(xlabel='Simulation time (s)',title='COM acceleration residual reduction | positive is better | unchanged constraints')
fig.colorbar(im,ax=ax,label='Reduction (%)')
fig.savefig(OUT/'posture_weight_comparison.png',dpi=160);plt.close(fig)

names=['left_hip_pitch_joint','left_hip_roll_joint','right_hip_pitch_joint','right_hip_roll_joint',
       'right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint','waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
values=np.array([[next(x['utilization'] for x in s['cases'][0]['torques'] if x['joint']==name)
                  for s in r['snapshots']] for name in names])
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(len(t)),[f'{v:.3f}' for v in t]);ax.set_yticks(range(len(names)),[n.removesuffix('_joint') for n in names])
for i in range(len(names)):
    for j in range(len(t)):ax.text(j,i,f'{values[i,j]*100:.1f}%',ha='center',va='center',fontsize=8)
ax.set(xlabel='Simulation time (s)',title='Baseline torque / actual QP budget | saturation alone does not prove causality')
fig.colorbar(im,ax=ax,label='Budget utilization')
fig.savefig(OUT/'torque_budget.png',dpi=160);plt.close(fig)
