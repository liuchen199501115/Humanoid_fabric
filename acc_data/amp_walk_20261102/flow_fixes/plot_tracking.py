"""Plot recorded physics against the unchanged, ground-aligned source poses."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from inspect_mocap_reference import urdf_fk
from test_g1_two_tasks import G1_JOINT_NAMES
from analyze_contact_transfer import sole_clearances

parser=argparse.ArgumentParser();parser.add_argument('recording',type=Path);args=parser.parse_args()
d=np.load(args.recording);plan=json.loads(args.recording.with_suffix('.steps_plan.json').read_text());report=json.loads(args.recording.with_suffix('.steps.json').read_text())
t=d['history_time'];q=d['history_q'].reshape(-1,36);source=d['history_reference_q'].reshape(-1,36)
feet=d['history_feet'].reshape(-1,2,7);forces=d['history_ground_foot_forces'].reshape(-1,2,3)[...,2]
poses,_,_=urdf_fk(source,G1_JOINT_NAMES,Path('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'))
clearance=sole_clearances(feet,plan['collision_points'],plan['collision_radii'])
fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
for foot,(side,color) in enumerate([('left','tab:blue'),('right','tab:orange')]):
    axes[0].plot(t,feet[:,foot,0],color=color,label=side+' measured')
    axes[0].plot(t,poses[side+'_ankle_roll_link'][0][:,0],'--',color=color,alpha=.7,label=side+' original PKL')
    axes[1].plot(t,clearance[:,foot]*100,color=color,label=side)
    axes[2].plot(t,forces[:,foot],color=color,label=side)
axes[1].axhline(2.,color='black',ls=':',label='flight criterion: 2 cm')
error=np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=1))
axes[3].plot(t,error,color='tab:red',label='original-PKL joint RMSE')
axes[3].axhline(.2,color='black',ls=':',label='fidelity limit: 0.2 rad')
for ax,label in zip(axes,['Foot world X (m)','Native sole clearance (cm)','Ground normal force (N)','Joint RMSE (rad)']):
    ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(loc='upper left',ncol=2,fontsize=8)
    for landing in report['landings']:ax.axvline(landing['time'],color='gray',alpha=.3)
axes[3].set_xlabel('Simulation time (s)')
fig.suptitle(f"AMP walking: {report['measured_steps']}/4 measured landings; strict tracking passed = {report['steps_tracking_passed']}")
fig.tight_layout();fig.savefig(args.recording.with_suffix('.png'),dpi=160)
