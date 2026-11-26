"""Bounded late swing11 clearance reduction; COM and both foot angles retained."""
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
target=OUT/(parent.stem+'_swing11lower'+str(round(depth*1000))+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
frames=np.arange(len(source));weight=smooth((frames-822)/20)*(1-smooth((frames-860)/24))
offsets=np.zeros((len(source),2,3));offsets[:,0,2]=-depth*weight
out,report=fit_com_reference(base,source,np.zeros((len(source),3)),legs,model,
                            foot_offsets=offsets,max_iterations=600)
np.testing.assert_array_equal(out[:823],base['control_q'][:823]);np.testing.assert_array_equal(out[884:],base['control_q'][884:])
sampler=BlendedSplineReference(MotionInterpolator(torch.tensor(out)),torch.tensor(out),stride=6,start_frame=130,ramp_frames=20)
plan=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.asarray(plan['collision_points'][0]);radii=np.asarray(plan['collision_radii'][0]);clearances=[]
for f in np.arange(802.,884.01,.25):
 q=sampler.kinematics(f,240.,0.)[0].numpy()[0];pos,rot,_=legs.evaluate(q,0)
 clearances.append(float(np.min((points@rot.T)[:,2]+pos[2]-radii)))
report.update(method=__doc__,reduction_m=depth,maximum_native_swing11_clearance_m=max(clearances),
 minimum_native_swing11_clearance_m=min(clearances),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
report['preflight_passed']=bool(report['preflight_passed'] and max(clearances)>.025 and min(clearances)>-.002)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(target.name,report,flush=True)
