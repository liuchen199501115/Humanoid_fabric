"""Lower only the tenth swing's middle while retaining its endpoint foot poses."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
name=sys.argv[1] if len(sys.argv)>1 else 'prepared_combined_root_y50';parent=OUT/(name+'.npz')
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
g=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.array(g['collision_points'][1]);radii=np.array(g['collision_radii'][1])
for depth in [.015,.025]:
 target=OUT/(name+f'_swing10lower{round(depth*1000)}.npz');assert not target.exists()
 out=base['control_q'].copy();maxp=maxr=0.;clearances=[]
 for k in range(781,802):
  u=(k-780)/22.;p,R,_=legs.evaluate(base['control_q'][k],1);p[2]-=depth*64*u**3*(1-u)**3
  q=out[k].astype(float).copy();loq=legs.lower[6:];hiq=legs.upper[6:]
  for it in range(180):
   pos,rot,J=legs.evaluate(q,1);err=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
   if abs(err).max()<5e-7:break
   lo=np.maximum(-.05,loq-q[13:19]);hi=np.minimum(.05,hiq-q[13:19])
   if np.any(lo>=hi):q[13:19]=np.clip(q[13:19],loq+1e-8,hiq-1e-8);continue
   q[13:19]+=lsq_linear(np.vstack([J,.0001*np.eye(6)]),np.r_[err,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
  out[k]=q;pos,rot,_=legs.evaluate(out[k],1)
  maxp=max(maxp,float(np.linalg.norm(pos-p)));maxr=max(maxr,float(Rotation.from_matrix(R@rot.T).magnitude()))
  clearances.append(float(np.min((points@rot.T)[:,2]+pos[2]-radii)))
 rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
 report=dict(method='C2 reduction of right swing height over source780-802; endpoints and all other coordinates retained',height_reduction_m=depth,maximum_foot_position_residual_m=maxp,maximum_foot_rotation_residual_rad=maxr,prepared_native_peak_clearance_m=max(clearances),peak_source_joint_rmse_rad=rmse,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False,preflight_passed=bool(maxp<2e-5 and maxr<2e-4 and max(clearances)>.025 and rmse<.19))
 np.testing.assert_array_equal(out[:781],base['control_q'][:781]);np.testing.assert_array_equal(out[802:],base['control_q'][802:]);np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 if report['preflight_passed']:
  with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
 print(report,flush=True)
