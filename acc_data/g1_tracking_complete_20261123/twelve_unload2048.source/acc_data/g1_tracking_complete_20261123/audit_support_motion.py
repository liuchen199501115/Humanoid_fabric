"""Audit nominal support motion; source contact flags do not guarantee no slip."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from g1_late_clock import late_speed_transition
from fabrics_sim.reference.interpolation import MotionInterpolator
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
contacts=np.load(OUT/'reference_com_cache.npz')['contacts']
geometry=json.loads((OUT/'ten_verify.steps_plan.json').read_text())
points=geometry['collision_points'];radii=geometry['collision_radii']
base=lambda t:smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
clock=lambda t:late_speed_transition(base(t),t,start=3.81,ramp=.12,previous_speed=1.2,speed=1.8)
frames=np.arange(660,1033);times=[brentq(lambda t:clock(t)[0]*200-k,0,12) for k in frames]
reports=[]
for name in sys.argv[1:]:
 path=OUT/(name+'.npz');raw=torch.tensor(np.load(path)['control_q'])
 sampler=BlendedSplineReference(MotionInterpolator(raw),raw,stride=6,start_frame=130,ramp_frames=20)
 rows=[]
 for k,t in zip(frames,times):
  q,v,_=[x.numpy().ravel() for x in sampler.kinematics(float(k),200*clock(t)[1],200*clock(t)[2])]
  for s in range(2):
   if not contacts[k,s]:continue
   p,R,J=legs.evaluate(q,s);local=J@v[6+6*s:12+6*s]
   lin=v[:3]+np.cross(v[3:6],p-q[:3])+local[:3];ang=v[3:6]+local[3:]
   offsets=np.asarray(points[s])@R.T;offsets[:,2]-=np.asarray(radii[s]);z=p[2]+offsets[:,2]
   low=z<=z.min()+.003
   point_v=lin+np.cross(ang,offsets)
   rows.append(dict(frame=int(k),foot=s,ankle_speed=float(np.linalg.norm(lin)),angular_speed=float(np.linalg.norm(ang)),lowest_material_point_speed=float(np.linalg.norm(point_v[low],axis=1).min()),minimum_clearance=float(z.min())))
 windows=[]
 for lo,hi in [(660,762),(763,802),(803,884),(885,1032)]:
  x=[r for r in rows if lo<=r['frame']<=hi]
  windows.append(dict(first_frame=lo,last_frame=hi,**{key:float(np.max([r[key] for r in x])) for key in ['ankle_speed','angular_speed','lowest_material_point_speed']}))
 reports.append(dict(reference=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Nominal source-contact foot velocity; full-sole fixed-contact and rolling contact require different interpretations. No simulator feasibility claim.',windows=windows,frames=rows))
 print(name,windows,flush=True)
(OUT/'nominal_support_motion_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
