import ast,json,pickle,hashlib,math,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,'examples')
from inspect_mocap_reference import urdf_fk
root=Path('/home/sky/Humanoid_fabric');out=root/'acc_data/g1_full_tracking_20261106'
source=root/'retarget/amp_reference/amp_walk_1ms.pkl';raw=pickle.load(source.open('rb'))
input_report=json.loads((root/'acc_data/amp_walk_20261102/input_report.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==input_report['source_sha256']
tree=ast.parse((root/'examples/test_g1_two_tasks.py').read_text())
names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_JOINT_NAMES' for t in n.targets))
order=[input_report['source_joint_order'].index(n) for n in names]
q=np.concatenate((raw['root_pos'],raw['root_rot'][:,[1,2,3,0]],raw['dof_pos'][:,order]),1).astype(np.float32)
q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
node=next(n for n in ast.parse((root/'examples/test_g1.py').read_text()).body if getattr(n,'name',None)=='interpolate_motion_trajectory')
ns={'torch':torch,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'sampler','exec'),ns)
reports=[]
selected=set(sys.argv[1:])
for audit in sorted(out.glob('*.audit.json')):
 r=json.loads(audit.read_text());name=r['run'];record=out/(name+'.npz')
 if selected and name not in selected:continue
 if not record.exists():record=out/(name+'.failure.npz')
 # A launch error or a failed serialization has no physical record to audit.
 if not record.exists():continue
 with np.load(record) as d:
  if 'history_q' not in d:continue
  shift=float(d['motion_ground_shift']);trajectory=torch.from_numpy(q.copy());trajectory[:,2]+=shift
  frames=d['history_frame'].ravel();times=d['history_time'].ravel()
  ref=torch.cat([ns['interpolate_motion_trajectory'](trajectory,float(f)) for f in frames]).numpy()
  actual=d['history_q'].reshape(-1,36)
  joint=np.sqrt(((actual[:,7:]-ref[:,7:])**2).mean(-1));position=np.linalg.norm(actual[:,:3]-ref[:,:3],axis=-1)
  angle=(Rotation.from_quat(actual[:,3:7])*Rotation.from_quat(ref[:,3:7]).inv()).magnitude()
  good=(joint<=.2)&(position<=.2)&(angle<=.35)
  steps=json.loads((out/(name+'.steps.json')).read_text())
  poses,_,_=urdf_fk(ref,names,root/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
  measured=d['history_feet'].reshape(-1,2,7)
  events=[]
  for event in steps['landings']:
   foot=0 if event['swing']=='left' else 1;index=int(abs(times-event['time']).argmin())
   foot_ref=poses[('left' if foot==0 else 'right')+'_ankle_roll_link'][0][index]
   error=float(np.linalg.norm(measured[index,foot,:3]-foot_ref))
   events.append(dict(step=event['step'],raw_source_foot_error_m=error,prefix_pose_passed=bool(good[times<=event['time']+1e-9].all()),same_frame_position_passed=error<=.04))
  result=dict(run=name,source_sha256=input_report['source_sha256'],source_frame_count=len(q),fps=float(raw['fps']),
   ground_shift_m=shift,last_source_frame=float(frames[-1]),raw_source_pose_fidelity_passed=bool(good.all()),
   peak_joint_rmse_rad=float(joint.max()),peak_root_position_m=float(position.max()),peak_root_angle_rad=float(angle.max()),
   raw_source_prefix_landings=sum(e['prefix_pose_passed'] and e['same_frame_position_passed'] for e in events),
   recorded_reference_max_joint_difference_rad=float(abs(ref[:,7:]-d['history_reference_q'].reshape(-1,36)[:,7:]).max()),
   landing_events=events,scope='Raw unclipped source poses; wxyz/xyzw and joint-order conversion plus the single recorded ground shift. Flight/load-hold evidence comes from the independent recorded step tracker; event positions rechecked by URDF FK.')
  (out/(name+'.raw_source.json')).write_text(json.dumps(result,indent=2)+'\n');reports.append(result)
  print(name,result['raw_source_prefix_landings'],result['last_source_frame'],result['raw_source_pose_fidelity_passed'])
if selected:reports=[json.loads(path.read_text()) for path in sorted(out.glob('*.raw_source.json'))]
(out/'raw_source_audit.json').write_text(json.dumps(dict(source=str(source),source_unchanged=True,source_sha256=input_report['source_sha256'],trials=reports),indent=2)+'\n')
