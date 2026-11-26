"""Scale a saved free-leg fit, then repeat nonlinear pose preflight."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
name=sys.argv[1];scale=float(sys.argv[2]);assert 0<scale<=1
fit=json.loads((OUT/(name+'.json')).read_text());assert fit['solver_status'] in ('solved','solved inaccurate')
parent=OUT/'prepared_combined_root_y50.npz';assert hashlib.sha256(parent.read_bytes()).hexdigest()==fit['parent_sha256']
with np.load(parent) as d:base={k:d[k] for k in d.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
nodes=np.unique(np.r_[np.arange(0,len(source),6),len(source)-1]);knots=np.r_[np.full(4,nodes[0]),nodes[2:-2],np.full(4,nodes[-1])]
greville=np.array([np.mean(knots[i+1:i+4]) for i in range(len(nodes))]);select=np.flatnonzero((greville>=672)&(greville<=1008))
x=np.load(OUT/(name+'.solver_iterate.npz'))['x'][:len(select)*32].reshape(len(select),32)
scales=np.r_[np.full(3,fit['root_axis_bound_m']),np.full(12,fit['leg_axis_bound_rad']),np.full(17,fit['upper_axis_bound_rad'])]
coeff=np.zeros((len(nodes),32));coeff[select]=x*scales*scale
correction=np.stack([np.interp(np.arange(len(source)),greville,coeff[:,j]) for j in range(32)],axis=1)
out=base['control_q'].copy();out[:,np.r_[np.arange(3),np.arange(7,36)]]+=correction
maxp=maxr=0.
for k in np.flatnonzero(np.max(abs(correction),axis=1)>1e-10):
 for side in (0,1):
  p,R,_=legs.evaluate(out[k],side);p0,R0,_=legs.evaluate(base['control_q'][k],side)
  maxp=max(maxp,float(np.linalg.norm(p-p0)));maxr=max(maxr,float(Rotation.from_matrix(R@R0.T).magnitude()))
rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
limits=float(np.maximum(legs.lower-out[:,7:19],out[:,7:19]-legs.upper).max())
# Upper corrections are convexly scaled from a fit whose full joint limits passed.
assert fit['maximum_upper_joint_limit_violation_rad']<1e-5
report=dict(method='Scaled independent-leg dynamics fit; nonlinear dynamics recheck required',fit=name,scale=scale,maximum_foot_position_residual_m=maxp,maximum_foot_rotation_residual_rad=maxr,peak_source_joint_rmse_rad=rmse,maximum_leg_limit_violation_rad=limits,full_tracking_passed=False,preflight_passed=maxp<.025 and maxr<.2 and rmse<.19 and limits<1e-5,parent_sha256=fit['parent_sha256'],source_sha256=base['source_sha256'].item())
target=OUT/(name+'_scale'+str(round(scale*100))+'.npz');assert not target.exists()
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(report)
