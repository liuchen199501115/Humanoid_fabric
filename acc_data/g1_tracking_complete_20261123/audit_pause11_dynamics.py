"""Nominal force audit for existing causal landing clocks; not a feasibility certificate."""
import json,sys
from pathlib import Path
import numpy as np,torch
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from landing_pause_clock import LandingPauseClock
from g1_late_clock import late_speed_transition
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_centroidal_geometry import CentroidalGeometry
from batched_reference_dynamics import reference_outputs
from g1_armature import unitree_armatures
raw=torch.tensor(np.load(OUT/'prepared_root_smooth_floor3_late11_sigma16.npz')['control_q']);ref=BlendedSplineReference(MotionInterpolator(raw),raw,stride=6,start_frame=130,ramp_frames=20)
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();model=CentroidalGeometry(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
base=lambda t:late_speed_transition(smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2),t,start=3.81,ramp=.12,previous_speed=1.2,speed=1.8)
frames=np.arange(816,884);contacts=np.load(OUT/'reference_com_cache.npz')['contacts'][frames];rows=[]
for start in [None,802,816,840]:
 pause=LandingPauseClock(start/200,4.42,.3,11) if start else None
 def clock(t):
  phase,rate,acc=base(t)
  if pause is None:return phase,rate,acc
  p,r,a=pause.sample(phase);return p,r*rate,a*rate**2+r*acc
 times=np.array([brentq(lambda t:200*clock(t)[0]-k,0,6) for k in frames]);states=[ref.kinematics(float(k),200*clock(t)[1],200*clock(t)[2]) for k,t in zip(frames,times)]
 q,v,a=[np.concatenate([s[j].numpy() for s in states]).astype(float) for j in range(3)]
 out=reference_outputs(model,q,v,a,contacts,np.asarray(unitree_armatures(names)).ravel());az=out[:,2]/model.mass-9.80665
 row=dict(pause_start_frame=start,minimum_com_acceleration=float(az.min()),maximum_com_acceleration=float(az.max()),negative_normal_force_frames=int((out[:,2]<0).sum()),time_to_frame883=float(times[-1]),sampled_frames=[816,883]);rows.append(row);print(row,flush=True)
(OUT/'pause11_nominal_force_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
