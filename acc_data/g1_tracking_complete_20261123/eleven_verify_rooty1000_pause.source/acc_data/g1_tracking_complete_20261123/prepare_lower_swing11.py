"""Bounded late swing11 lowering with optional COM drop; both foot angles retained."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
parent=OUT/(sys.argv[1]+'.npz');depth=float(sys.argv[2]);assert 0<depth<=.05
com_drop=float(sys.argv[3]) if len(sys.argv)>3 else 0.
assert 0<=com_drop<=.03
landing_floor='--landing-floor' in sys.argv
drop_tag=('_comdown'+str(round(com_drop*1000))) if com_drop else ''
if landing_floor:drop_tag+='_floor3'
target=OUT/(parent.stem+'_swing11lower'+str(round(depth*1000))+drop_tag+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
frames=np.arange(len(source));fade_start,fade_end=(900,924) if landing_floor else (860,884)
weight=smooth((frames-822)/20)*(1-smooth((frames-fade_start)/(fade_end-fade_start)))
offsets=np.zeros((len(source),2,3));offsets[:,0,2]=-depth*weight
plan=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.asarray(plan['collision_points'][0]);radii=np.asarray(plan['collision_radii'][0])
if landing_floor:
 for k in np.flatnonzero(weight):
  pos,rot,_=legs.evaluate(base['control_q'][k],0)
  height=float(np.min((points@rot.T)[:,2]+pos[2]-radii))
  offsets[k,0,2]=-min(depth*weight[k],max(0.,height-.003))
com_delta=np.zeros((len(source),3));com_delta[:,2]=-com_drop*weight
out,report=fit_com_reference(base,source,com_delta,legs,model,
                            foot_offsets=offsets,max_iterations=600)
np.testing.assert_array_equal(out[:823],base['control_q'][:823]);np.testing.assert_array_equal(out[fade_end:],base['control_q'][fade_end:])
sampler=BlendedSplineReference(MotionInterpolator(torch.tensor(out)),torch.tensor(out),stride=6,start_frame=130,ramp_frames=20)
clearances=[];swing_clearances=[]
for f in np.arange(802.,fade_end+.01,.25):
 q=sampler.kinematics(f,240.,0.)[0].numpy()[0];pos,rot,_=legs.evaluate(q,0)
 clearances.append(float(np.min((points@rot.T)[:,2]+pos[2]-radii)))
 if f<=884:swing_clearances.append(clearances[-1])
report.update(method=__doc__,reduction_m=depth,com_drop_m=com_drop,landing_floor_m=.003 if landing_floor else None,
 fade_start_frame=fade_start,fade_end_frame=fade_end,maximum_native_swing11_clearance_m=max(swing_clearances),
 minimum_native_swing11_clearance_m=min(clearances),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
report['preflight_passed']=bool(report['preflight_passed'] and max(swing_clearances)>.025 and min(clearances)>-.002)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(target.name,report,flush=True)
