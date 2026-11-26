from pathlib import Path
import ast,math,pickle,cProfile,pstats
from types import SimpleNamespace
import numpy as np
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_separating28'
c=pickle.load(open(p/(run+'.planner_worker.configuration.pkl'),'rb'));c['surface_kinematics']=True;c['contact_schedule_path']=None
fn=next(n for n in ast.parse((p/'preview_worker.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='initialize');ns=dict(ast=ast,math=math,Path=Path,SimpleNamespace=SimpleNamespace);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<init>','exec'),ns);planner=ns['initialize'](c)
def read(path):
 with np.load(path,allow_pickle=True) as z:return {k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
d=read(p/(run+'.replan_states/3.708.npz'));nominal=read(p/'surface_actual_3708_plan.npz');first={k.removeprefix('problem_'):v for k,v in d.items() if k.startswith('problem_')}
import time,json
import preview_constraint_generation as cuts
original=cuts.solve;results=[]
for count in [8,16,24]:
 def solve(*args,**kwargs):
  kwargs['initial_per_block']=count
  return original(*args,**kwargs)
 cuts.solve=solve
 planner._fixed_patches=None
 start=time.monotonic();r=planner.solve(d['q'],d['v'],d['frame'],d['rate'],first,nominal=nominal)
 r.update(planner.nonlinear_audit(r,d['rate']))
 np.savez_compressed(p/('surface_cuts'+str(count)+'_plan.npz'),**r)
 row={k:r[k] for k in ['success','status','objective','nonlinear_verified','linear_violation','tube_violation','collision_cuts']}
 row.update(initial_per_block=count,seconds=time.monotonic()-start)
 results.append(row);print(row,flush=True)
(p/'surface_cut_sizes_audit.json').write_text(json.dumps(results,indent=2)+'\n')
