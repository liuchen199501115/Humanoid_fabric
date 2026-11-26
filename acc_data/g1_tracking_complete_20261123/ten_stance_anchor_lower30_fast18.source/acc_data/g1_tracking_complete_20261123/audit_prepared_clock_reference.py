"""Pair a clock-aware pose fit with its fixed clock and audit exported dynamics.

The clock was optimized for the parent pose, then the child pose was optimized
under that clock. Record both hashes; no claim that original clock slacks apply
to the child. This is offline preparation, not predictive runtime control.
"""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from smooth_clock import smooth_clock
from g1_prepared_clock import PreparedSourceClock
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_centroidal_geometry import CentroidalGeometry
from batched_reference_dynamics import reference_outputs
from g1_armature import unitree_armatures
reference=OUT/(sys.argv[1]+'.npz');preparation=json.loads(reference.with_suffix('.json').read_text())
assert preparation['preflight_passed']
clock_path=Path(preparation['prepared_clock_file']);clock_hash=hashlib.sha256(clock_path.read_bytes()).hexdigest()
assert clock_hash==preparation['prepared_clock_sha256']
clock_data=dict(np.load(clock_path));data=dict(np.load(reference))
assert clock_data['parent_sha256'].item()==preparation['parent_sha256']
assert clock_data['source_sha256'].item()==data['source_sha256'].item()
target=reference.with_name(reference.stem+'_clock.npz');assert not target.exists()
clock_data.update(parent_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
 planning_parent_sha256=preparation['parent_sha256'],planning_clock_sha256=clock_hash)
clock_data['planning_physical_slack']=clock_data.pop('physical_slack')
np.savez_compressed(target,**clock_data)
base_clock=lambda t:smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
clock=PreparedSourceClock(target,reference,data['source_sha256'].item(),200.,base_clock)
sample=lambda t:clock.sample(t,base_clock(t))
def curve(q):
 q=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
sampler=curve(data['control_q']);parent=curve(np.load(OUT/preparation['parent_file'])['control_q'])
prefix_end=min(float(clock_data['frame'][0]),float(preparation['first_free_knot_frame'])-12)
for k in np.arange(0,prefix_end+.01,.5):
 for a,b in zip(sampler.kinematics(k,200.,0.),parent.kinematics(k,200.,0.)):
  np.testing.assert_array_equal(a.numpy(),b.numpy())
frames=np.arange(660,1033);times=np.array([brentq(lambda t:sample(t)[0]-k/200.,0.,12.) for k in frames])
states=[sampler.kinematics(float(k),sample(t)[1]*200,sample(t)[2]*200) for k,t in zip(frames,times)]
q,v,a=[np.concatenate([s[j].numpy() for s in states]).astype(float) for j in range(3)]
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=CentroidalGeometry(str(urdf),names);contacts=np.load(OUT/'reference_com_cache.npz')['contacts'][frames]
out=reference_outputs(model,q,v,a,contacts,np.asarray(unitree_armatures(names)).ravel())
limits={j.get('name'):[float(j.find('limit').get(k)) for k in ['effort','velocity']] for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
effort,velocity=np.array([limits[n] for n in names]).T;budget=effort[None]*np.where(frames[:,None]<810,.8,1.)
ratios=abs(out[:,6:])/budget;ratios[contacts.sum(-1)!=1,:12]=np.nan
corners=np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
rows=[]
for i,k in enumerate(frames):
 c,_,poses=model.evaluate(q[i]);F,H=out[i,:3],out[i,3:6];active=contacts[i]
 cop=[]
 if active.any():
  points=np.concatenate([(poses[s+'_ankle_roll_link'][1]@corners.T).T+poses[s+'_ankle_roll_link'][0] for s,on in zip(['left','right'],active) if on])[:,:2]
  for nx,ny,b in ConvexHull(points).equations:
   cop.append(float((nx*(c[0]*F[2]-c[2]*F[0]-H[1])+ny*(c[1]*F[2]-c[2]*F[1]+H[0])+b*F[2])/(model.mass*9.81)))
 friction=max(float((np.cos(th)*F[0]+np.sin(th)*F[1]-.55*np.cos(np.pi/8)*F[2])/(model.mass*9.81)) for th in np.arange(8)*np.pi/4)
 rows.append(dict(frame=int(k),phase_speed=float(sample(times[i])[1]),maximum_effort_ratio=float(np.nanmax(ratios[i])),maximum_velocity_ratio=float(np.max(abs(v[i,6:])/velocity)),cop_residual_m=max(cop) if cop else None,friction_residual=friction,normal_force_n=float(F[2])))
report=dict(method=__doc__,reference=str(reference),clock=clock.report,source_sha256=data['source_sha256'].item(),
 original_planning_clock_sha256=clock_hash,original_planning_reference_sha256=preparation['parent_sha256'],
 exact_export_scope='Nominal full-sole support; SS leg effort and all upper joints, no impact model',
 prefix_q_v_a_equal_through_frame=prefix_end,maximum_effort_ratio=float(np.nanmax(ratios)),
 maximum_cop_residual_m=max(r['cop_residual_m'] for r in rows if r['cop_residual_m'] is not None),
 maximum_velocity_ratio=max(r['maximum_velocity_ratio'] for r in rows),full_tracking_passed=False,frames=rows)
reference.with_suffix('.clock_dynamics_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='frames'})
