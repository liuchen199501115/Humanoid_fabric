"""Audit actual single-QP torque and acceptance, including preserved history prefix."""
import argparse, json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
rows=[]
parser=argparse.ArgumentParser()
parser.add_argument('names',nargs='+')
parser.add_argument('--prefix-time',type=float,default=4.344)
args=parser.parse_args()
for name in args.names:
 r=json.loads((OUT/(name+'.result.json')).read_text());t=r['tracking'];s=r['steps']
 assert t['tolerances']==dict(joint_rmse_rad=.2,root_position_m=.2,root_orientation_rad=.35)
 data=OUT/(name+'.npz')
 if not data.exists():data=OUT/(name+'.failure.npz')
 with np.load(data) as a, np.load(OUT/'eleven_resume10_leg16.npz') as b:
  assert (a['history_qp_native_solve_count']==1).all()
  if 'diagnostic_qp_native_solve_count' in a:assert (a['diagnostic_qp_native_solve_count']==1).all()
  for key in ['history_priority_active','history_fabric_task_requests_cached','history_joint_tube_recovery_used','history_torque_pd_estimate']: assert not a[key].any(),key
  names=a['joint_names'] if 'joint_names' in a else a['final_joint_order']
  order=[a['servo_joint_order'].tolist().index(n) for n in names]
  np.testing.assert_array_equal(a['servo_torque'][:,:,order],a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
  # Default covers the original 11-step certificate; earlier experiments
  # can explicitly audit a shorter unchanged prefix.
  count=int((b['history_time']<=args.prefix_time).sum());equal=[];different=[]
  for key in b.files:
   if key.startswith('history_') and key in a and b[key].ndim and a[key].ndim and b[key].shape[0]==len(b['history_time']) and a[key].shape[0]==len(a['history_time']) and 'wall' not in key and 'solve_time' not in key:
    same=np.array_equal(a[key][:count],b[key][:count],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key][:count],b[key][:count])
    (equal if same else different).append(key)
  row=dict(name=name,landings=s['measured_steps'],source_frame=t['last_source_frame'],prefix_end_time_s=args.prefix_time,history_equal_prefix_fields=len(equal),different_prefix_fields=different,one_native_qp=True,exact_applied_qp_torque=True,unchanged_fidelity=True,next_step=s['next_step_diagnostics'])
  (OUT/(name+'.strict_audit.json')).write_text(json.dumps(row,indent=2)+'\n')
  rows.append(row);print(name,row['landings'],row['source_frame'],'prefix differences',different)
(OUT/'eleven_latest_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
