"""Check later spline geometry and acceleration before closed-loop experiments."""
import sys,json
from pathlib import Path
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from late_tube_smoothing import transition_tube_spline
from numpy_leg_kinematics import LegKinematics
q=torch.from_numpy(np.load(OUT/'prepared_combined_root_y50.npz')['control_q'])
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
names=np.load(OUT/'landing9_load.npz')['final_joint_order'].tolist();model=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)

sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
events=json.loads((OUT/'landing9_load.steps_plan.json').read_text())['events'];records=[]
for i,event in enumerate(events):
 side=1-event['support'];end=event['end_frame'];later=next((e['takeoff_frame'] for e in events[i+1:] if e['support']==event['support']),len(q)-1)
 errors=[]
 for k in range(end,min(later,end+80)+1):
  pose=sampler(float(k)).numpy()[0];errors.append(float(np.linalg.norm(model.evaluate(pose,side)[0]-model.evaluate(source[k],side)[0])))
 records.append(dict(step=i+1,swing=side,end_frame=end,window_end=min(later,end+80),min_reference_foot_error_m=min(errors),max_reference_foot_error_m=max(errors)))
(OUT/'reference_landing_windows.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
