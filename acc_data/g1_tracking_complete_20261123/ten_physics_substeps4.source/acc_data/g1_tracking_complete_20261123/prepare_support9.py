"""Bring the seventh landing under the COM without changing source acceptance."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch,warp as wp
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from landing_offset import adapt_landing
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz') as a:source=a['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.npz') as a:names=a['final_joint_order'].tolist()
events=json.loads((ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.steps_plan.json').read_text())['events']
for mm in [15,25]:
 target=OUT/f'prepared_support9_fixed_y{mm}.npz'
 if target.exists():raise FileExistsError(target)
 q,report=adapt_landing(f,torch.tensor(base['control_q']),names,events[8],events[10],[0.,-mm/1000.],smoothing=2.)
 arr=q.numpy();start=events[8]['takeoff_frame'];end=events[10]['end_frame']
 arr[:start+1]=base['control_q'][:start+1];arr[end:]=base['control_q'][end:]
 report.update(source_sha256=base['source_sha256'].item(),parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
  peak_source_joint_rmse_rad=float(np.sqrt(np.mean((arr[:,7:]-source[:,7:])**2,axis=1)).max()),
  peak_source_root_displacement_m=float(np.linalg.norm(arr[:,:3]-source[:,:3],axis=1).max()),original_audit_retained=True)
 assert report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.1
 with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=arr))
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
