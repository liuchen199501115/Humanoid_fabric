"""Read-only snapshots of actual states at online replanning; no simulator APIs."""
from pathlib import Path
import ast,math,time,json
import numpy as np,torch
from preview_manifold import difference
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_state_release28'
folder=p/(run+'.replan_states');folder.mkdir(exist_ok=True)
tree=ast.parse(Path('examples/test_g1.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interpolate_motion_trajectory');ns=dict(torch=torch,math=math);exec(compile(ast.Module(body=[node],type_ignores=[]),'<source>','exec'),ns)
source=torch.tensor(np.load(p/(run+'.control_reference.npz'))['source_q']);seen=set();start=time.monotonic()
while time.monotonic()-start<7200:
 try:
  with np.load(p/(run+'.full_body_preview.latest_problem.npz'),allow_pickle=True) as raw:d={k:raw[k].copy() for k in raw.files}
  t=float(d['time'])
  if t not in seen:
   qsrc=ns['interpolate_motion_trajectory'](source,float(d['frame'])).numpy().ravel();e=difference(d['q'],qsrc)
   row=dict(time=t,frame=float(d['frame']),actual_mode=int(d['problem_mode']),joint_rmse=float(np.sqrt(np.mean(e[6:]**2))),root_position=float(np.linalg.norm(e[:3])),root_orientation=float(np.linalg.norm(e[3:6])))
   np.savez_compressed(folder/f'{t:.3f}.npz',**d)
   with (folder/'fidelity.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True);seen.add(t)
 except (OSError,ValueError,EOFError):pass
 if (p/(run+'.tracking.json')).exists():break
 time.sleep(5)
