"""Offline stance10 clearance correction with fixed root attitude and COM.

Uses the native collision spheres. Correcting reference penetration does not
change the simulated ground, contact bounds, source clip or acceptance.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
label,parent_name,start=sys.argv[1],sys.argv[2],int(sys.argv[3])
assert start in (802,810,822)
parent=OUT/(parent_name+'.npz');target=OUT/('prepared_stance10_clearance_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
g=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.asarray(g['collision_points'][1]);radii=np.asarray(g['collision_radii'][1])
def clearance(q):
 p,R,_=legs.evaluate(q,1);return float(np.min((points@R.T)[:,2]+p[2]-radii))
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
frames=np.arange(len(source));envelope=smooth((frames-start)/12)*(1-smooth((frames-884)/24))
height=np.array([clearance(q) for q in base['control_q']]);lift=gaussian_filter1d(np.maximum(.001-height,0.),2.)*envelope
assert lift.max()<.02
offsets=np.zeros((len(source),2,3));offsets[:,1,2]=lift
out,report=fit_com_reference(base,source,np.zeros((len(source),3)),legs,model,foot_offsets=offsets,fixed_root=True,max_iterations=600)
np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1])
np.testing.assert_array_equal(out[908:],base['control_q'][908:])
def sampler(q):
 t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out)
for k in np.arange(0,start-6+.01,.5):
 for old,new in zip(a.kinematics(k,240.,0.),b.kinematics(k,240.,0.)):np.testing.assert_array_equal(old.numpy(),new.numpy())
fine=np.arange(start,908.01,.25);before=[clearance(a(float(k)).numpy()[0]) for k in fine];after=[clearance(b(float(k)).numpy()[0]) for k in fine]
report.update(method=__doc__,parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),maximum_foot_lift_m=float(lift.max()),prefix_q_v_a_equal_through_frame=start-6,minimum_stance10_clearance_before_m=min(before),minimum_stance10_clearance_after_m=min(after),full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(target.name,report,flush=True)
