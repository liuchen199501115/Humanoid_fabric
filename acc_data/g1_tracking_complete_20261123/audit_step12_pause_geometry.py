"""Compare native sole geometry of the actual, audit and smoothed control poses."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
name='eleven_source60_foot4096_late_upper256';a=np.load(OUT/(name+'.npz'));plan=json.loads((OUT/(name+'.steps_plan.json')).read_text());points=np.asarray(plan['collision_points'][1]);radii=np.asarray(plan['collision_radii'][1]);legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),a['final_joint_order'].tolist())
def clearance(q):
 p,R,_=legs.evaluate(q,1);return float(p[2]+np.min((points@R.T)[:,2]-radii))
raw=torch.tensor(np.load(OUT/'prepared_return_source_transfer60.npz')['control_q']);sampler=BlendedSplineReference(MotionInterpolator(raw),raw,stride=6,start_frame=130,ramp_frames=20)
phase=[]
for k in range(948,967):
 q,_,_=sampler.kinematics(float(k),0.,0.);phase.append(dict(source_frame=k,control_native_clearance_m=clearance(q.numpy().reshape(-1))))
measured=[]
for stamp in [4.5,4.55,4.6,4.65,4.7,4.75,4.8,4.9]:
 i=int(np.argmin(abs(a['history_time']-stamp)));r=dict(time_s=float(a['history_time'][i]),source_frame=float(a['history_frame'][i]),normal_load_n=float(a['history_ground_foot_forces'][i,1,2]))
 for label,key in [('actual','history_q'),('source','history_reference_q'),('control','history_control_reference_q')]:r[label+'_native_clearance_m']=clearance(a[key][i,0])
 measured.append(r)
assert measured[-1]['normal_load_n']==0
assert abs(measured[-1]['control_native_clearance_m']-phase[4]['control_native_clearance_m'])<1e-7
report=dict(trial=name,measured=measured,control_phase_geometry=phase,first_sampled_control_ground_frame=next(x['source_frame'] for x in phase if x['control_native_clearance_m']<=0),full_tracking_passed=False)
(OUT/'step12_pause_geometry.json').write_text(json.dumps(report,indent=2)+'\n');print('First sampled native-ground control phase:',report['first_sampled_control_ground_frame'])
