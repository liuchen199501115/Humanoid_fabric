from pathlib import Path
import ast,math,pickle,json,time
from types import SimpleNamespace
import numpy as np
from full_body_preview_collision_anchored_release import FullBodyPreview
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_state_release28'
tree=ast.parse((p/'preview_worker.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize');ns=dict(ast=ast,math=math,Path=Path,SimpleNamespace=SimpleNamespace);exec(compile(ast.Module(body=[fn],type_ignores=[]),'<initializer>','exec'),ns)
with (p/(run+'.planner_worker.configuration.pkl')).open('rb') as f:config=pickle.load(f)
old=ns['initialize'](config)
planner=FullBodyPreview(old.model,old.reference,old.source,old.modes,old.geometry,old.limits,old.effort,old.velocity,horizon=28,guarded=old.guarded)
with np.load(p/(run+'.replan_states/3.846.npz'),allow_pickle=True) as z:d={k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
first={k.removeprefix('problem_'):v for k,v in d.items() if k.startswith('problem_')}
from reference_contact_schedule import delayed_touchdowns,native_reference_clearance
clearance=native_reference_clearance(planner,d['rate'])
original_modes=planner.modes.copy();planner.modes,changes=delayed_touchdowns(original_modes,clearance,840)
np.savez_compressed(p/'prepared_contact_schedule.npz',original_modes=original_modes,modes=planner.modes,clearance=clearance)
(p/'prepared_contact_schedule.json').write_text(json.dumps(changes,indent=2)+'\n');print(changes,flush=True)
from preview_warm_start import shift_prediction
from preview_manifold import integrate,difference
with np.load(p/'anchored_native_contact_initial_3680_plan.npz',allow_pickle=True) as z:previous={k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
planner.grid_origin=float(previous['frames'][0]);intervals=planner.prediction_intervals(d['frame'],d['rate'])
planner._initial_guess=shift_prediction(previous,planner.reference,d['frame'],d['rate'],intervals,len(intervals),integrate,difference,integration='semiimplicit')
start=time.monotonic();r=planner.refine(d['q'],d['v'],d['frame'],d['rate'],first,iterations=20)
np.savez_compressed(p/'anchored_native_contact_actual_3846_warm_plan.npz',**r)
result={k:v for k,v in r.items() if k in ('success','status','refinement_history','contact_patch_release_history') or k.startswith('nonlinear_')};result.update(actual_start_time=d['time'],solve_seconds=time.monotonic()-start,prediction_only=True)
(p/'anchored_native_contact_actual_3846_warm_audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if not isinstance(v,(dict,list))},flush=True)
