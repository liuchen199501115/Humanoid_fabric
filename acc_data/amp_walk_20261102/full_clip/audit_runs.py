"""Regenerate full-clip failure audits and a compact diagnostic plot."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_motion_tracking import summarize_full_motion
from analyze_contact_transfer import sole_clearances
p=Path('acc_data/amp_walk_20261102/full_clip')
runs=[]
for name in ('full_motion','horizontal_balance4','adaptive_balance','landing_orientation'):
    with np.load(p/(name+'.failure.npz')) as data:
        plan=json.loads((p/(name+'.steps_plan.json')).read_text())
        report=summarize_full_motion(data,plan)
        report.update(stop_reason='QP validation failed',failed_time_s=float(data['control_time']),
                      failed_source_frame=float(data['motion_frame_position']))
        (p/(name+'.tracking.json')).write_text(json.dumps(report,indent=2)+'\n')
        runs.append(dict(name=name,**report))
source=Path('retarget/amp_reference/amp_walk_1ms.pkl')
summary=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    fps=200,source_frames=2000,duration_s=9.995,requested_complete_landings=29,
    final_source_segment='Unfinished swing at frames 1986..1999; no invented terminal landing',
    runner='examples/run_amp_full_motion.sh',full_motion_tracking_passed=False,
    active_profile='Original four-step controller, extended to all complete source arcs with feasible IK lift',
    default_record=str(p/'full_motion.failure.npz'),runs=runs,
    endpoint_smoke_test=json.loads((p/'endpoint_smoke.tracking.json').read_text()),
    endpoint_smoke_scope='Only a 60-frame test of stopping/auditing at a moving endpoint; not full AMP tracking',
    experimental_controller_changes_retained=False)
(p/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
with np.load(p/'full_motion.failure.npz') as d:
    t=d['history_time'].ravel();frame=d['history_frame'].ravel()
    q=d['history_q'].reshape(-1,36);ref=d['history_reference_q'].reshape(-1,36)
    com=d['history_com'].reshape(-1,3);target=d['history_com_target'].reshape(-1,3)
    feet=d['history_feet'].reshape(-1,2,7);forces=d['history_ground_foot_forces'].reshape(-1,2,3)[...,2]
plan=json.loads((p/'full_motion.steps_plan.json').read_text())
steps=json.loads((p/'full_motion.steps.json').read_text())
fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
for i,(side,color) in enumerate((('left','tab:blue'),('right','tab:orange'))):
    goal=plan['landing_positions_world'][4+i]
    axes[0].plot(t,feet[:,i,0],label=side+' foot X',color=color)
    axes[0].axhline(goal[0],ls=':',color=color,label=side+' next landing X')
    axes[2].plot(t,forces[:,i],color=color,label=side+' ground force')
for i,axis in enumerate('xyz'):
    axes[1].plot(t,(com-target)[:,i],label='COM '+axis+' error')
axes[3].plot(t,np.sqrt(np.mean((q[:,7:]-ref[:,7:])**2,axis=-1)),label='Original PKL joint RMSE',color='tab:red')
axes[3].axhline(.2,ls=':',color='black',label='Unchanged 0.2 rad tolerance')
for ax,label in zip(axes,['Foot X (m)','COM error (m)','Ground load (N)','Joint RMSE (rad)']):
    ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(loc='upper left',fontsize=8,ncol=2)
    for landing in steps['landings']:ax.axvline(landing['time'],color='gray',alpha=.25)
    ax.axvline(1.76,color='purple',ls='--',alpha=.6)
axes[-1].set_xlabel('Simulation time (s); dashed purple = source step 5 landing time')
fig.suptitle('Full AMP attempt: 4/29 verified landings; QP failure at 2.410 s / requested 9.995 s')
fig.tight_layout();fig.savefig(p/'full_motion_diagnostics.png',dpi=150)
print('Saved full-clip audits and diagnostic plot; all full-length attempts failed.')
for r in runs: print(r['name'],r['failed_time_s'],r['measured_landings'])
