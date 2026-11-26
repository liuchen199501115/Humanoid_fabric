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
profiler=cProfile.Profile();profiler.enable();result=planner.solve(d['q'],d['v'],d['frame'],d['rate'],first,nominal=nominal);profiler.disable();profiler.dump_stats(p/'surface_solve.profile')
print(result['success'],result['status']);pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(35)
