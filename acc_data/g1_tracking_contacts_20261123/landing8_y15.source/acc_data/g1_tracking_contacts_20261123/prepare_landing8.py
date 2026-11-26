"""Offline full-body IK correction of step 8, preserving the original PKL audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch, warp as wp
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from landing_offset import adapt_landing
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
parent=ROOT/'acc_data/g1_tracking_fix_20261123/landing25.npz'
with np.load(parent) as d:base={k:d[k] for k in d.files}
with np.load(ROOT/'acc_data/g1_tracking_continue_20261123/slow_landing25.control_reference.npz') as d:
 source=d['source_q'];assert np.array_equal(d['control_q'],base['control_q'])
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
with np.load(ROOT/'acc_data/g1_tracking_continue_20261123/slow_landing25.npz') as d:names=d['final_joint_order'].tolist()
events=json.loads((ROOT/'acc_data/g1_tracking_continue_20261123/slow_landing25.steps_plan.json').read_text())['events']
for millimeters in [15,20]:
 target=OUT/f'landing8_y{millimeters}.npz'
 if target.exists():raise FileExistsError(target)
 q,report=adapt_landing(f,torch.tensor(base['control_q']),names,events[7],events[9],[0.,millimeters/1000.],smoothing=2.)
 arr=q.numpy();arr[:573]=base['control_q'][:573];arr[802:]=base['control_q'][802:];np.testing.assert_array_equal(arr[:573],base['control_q'][:573])
 report.update(original_source_sha256=base['source_sha256'].item(),parent=str(parent.relative_to(ROOT)),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),peak_source_joint_rmse_rad=float(np.sqrt(np.mean((arr[:,7:]-source[:,7:])**2,axis=1)).max()),peak_source_root_change_m=float(np.linalg.norm(arr[:,:3]-source[:,:3],axis=1).max()),original_audit_retained=True)
 if report['peak_source_joint_rmse_rad']>=.19 or report['peak_source_root_change_m']>=.1:raise ValueError(report)
 with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=arr))
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
