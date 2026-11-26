"""Inspect the candidate prediction; this is not an actual tracking verdict."""
from pathlib import Path
import ast,math,pickle,json
import numpy as np,torch
from types import SimpleNamespace
from preview_manifold import difference
from scipy.spatial.transform import Rotation
p=Path(__file__).parent
run='fixed_clock120_body10_full_body_preview42_collision_state_release28'
tree=ast.parse((p/'preview_worker.py').read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize')
namespace=dict(ast=ast,math=math,Path=Path,SimpleNamespace=SimpleNamespace)
exec(compile(ast.Module(body=[function],type_ignores=[]),'<worker_initializer>','exec'),namespace)
with (p/(run+'.planner_worker.configuration.pkl')).open('rb') as f:config=pickle.load(f)
planner=namespace['initialize'](config)
with np.load(p/(run+'.full_body_preview.latest_successful_plan.npz'),allow_pickle=True) as z:
 plan={k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
rows=[]
for i,(frame,q,v,mode) in enumerate(zip(plan['frames'],plan['physical_q'],plan['physical_v'],plan['contact_modes'])):
 source=planner.source.kinematics(float(frame),plan['rate'])[0].numpy().ravel();err=difference(q,source);state=planner.model.evaluate(q,v)
 height=[]
 for side in [0,1]:
  points,radii=planner.geometry[side];foot=state['feet'][side];height.append(float(foot[2]+np.min((points@Rotation.from_quat(foot[3:]).as_matrix().T)[:,2]-radii)))
 row=dict(frame=float(frame),mode=int(mode),root_position=float(np.linalg.norm(err[:3])),root_orientation=float(np.linalg.norm(err[3:6])),joint_rmse=float(np.sqrt(np.mean(err[6:]**2))),native_clearance=height)
 if i<len(plan['controls']):row['normal_force']=plan['controls'][i,[37,43]].tolist()
 rows.append(row)
result=dict(prediction_only=True,actual_start_time=plan['time'],rows=rows)
(p/'collision_state_release_prediction_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
