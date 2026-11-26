"""Independent FK/load audit of the first observed twelfth landing."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
name=sys.argv[1] if len(sys.argv)>1 else 'eleven_pause960_search2'
a=np.load(OUT/(name+'.npz'));r=json.loads((OUT/(name+'.result.json')).read_text());event=r['steps']['landings'][11];assert event['step']==12
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),a['final_joint_order'].tolist());plan=json.loads((OUT/(name+'.steps_plan.json')).read_text());points=np.asarray(plan['collision_points'][1]);radii=np.asarray(plan['collision_radii'][1]);times=a['history_time'];window=[];flight=[];certificate=None
for i in np.flatnonzero(times>=4.25):
 q=a['history_q'][i,0];source=a['history_reference_q'][i,0];pos,R,_=legs.evaluate(q,1);goal,_,_=legs.evaluate(source,1);error=float(np.linalg.norm(pos-goal));load=float(a['history_ground_foot_forces'][i,1,2]);clearance=float(pos[2]+np.min((points@R.T)[:,2]-radii));mode=int(a['history_measured_mode'][i].item())
 if load<10 and a['history_ground_foot_forces'][i,0,2]>=100 and mode==1:flight.append(clearance)
 ready=bool(a['history_frame'][i]>=951 and error<=.04 and load>=100 and mode in (2,3))
 if ready:window.append((i,error,load))
 else:window=[]
 if window and times[i]-times[window[0][0]]>=.05-1e-9:
  certificate=i;break
assert certificate is not None and max(flight)>=.02
hold=float(times[certificate]-times[window[0][0]])
# The existing online counter permits 20N continuous bearing. Independently
# demand >=100N at every recorded sample for a full 50ms elapsed interval.
q=a['history_q'][:certificate+1,0];ref=a['history_reference_q'][:certificate+1,0]
joint=np.sqrt(np.mean((q[:,7:]-ref[:,7:])**2,axis=-1));root=np.linalg.norm(q[:,:3]-ref[:,:3],axis=-1)
u=q[:,3:7]/np.linalg.norm(q[:,3:7],axis=-1,keepdims=True);v=ref[:,3:7]/np.linalg.norm(ref[:,3:7],axis=-1,keepdims=True);angle=2*np.arccos(np.clip(abs(np.sum(u*v,axis=-1)),0,1))
assert joint.max()<=.2 and root.max()<=.2 and angle.max()<=.35
assert np.all(a['history_qp_native_solve_count']==1)
assert not a['history_torque_pd_estimate'].any()
original=ROOT/'acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl';sha=hashlib.sha256(original.read_bytes()).hexdigest();assert sha=='919eaff9b2521cce1e8b40e263fceb3f3f1873a7bc42500d9932be62f007eca9'
report=dict(trial=name,source_sha256=sha,step=12,online_landing_time_s=event['time'],strict_100N_50ms_time_s=float(times[certificate]),source_frame=float(a['history_frame'][certificate]),hold_start_s=float(times[window[0][0]]),hold_end_s=float(times[window[-1][0]]),loaded_hold_elapsed_s=hold,minimum_measured_load_n=min(x[2] for x in window),maximum_same_frame_source_foot_error_m=max(x[1] for x in window),peak_unloaded_native_clearance_m=max(flight),peak_joint_rmse_until_certificate_rad=float(joint.max()),peak_root_position_until_certificate_m=float(root.max()),peak_root_orientation_until_certificate_rad=float(angle.max()),one_native_qp_per_control_update=True,extra_pd_torque=False,full_tracking_passed=False)
(OUT/(name+'.twelfth_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
