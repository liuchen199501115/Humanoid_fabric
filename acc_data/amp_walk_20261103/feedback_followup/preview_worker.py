"""Standalone predictor: no Isaac imports, simulator API, or state writes."""
import ast,contextlib,math,sys,traceback
from pathlib import Path
from types import SimpleNamespace
from preview_process import receive,send

def initialize(c):
 import torch,numpy as np
 from mujoco_reference_dynamics import ReferenceDynamics
 from blended_spline import BlendedSplineReference
 from fabrics_sim.reference.interpolation import MotionInterpolator
 if c.get('collision_geometry') is not None:
  from preview_collision import CollisionDynamics as ReferenceDynamics
  if c.get('fine_control'):
   if c.get('surface_kinematics'):
    if c['warm_start']:from full_body_preview_collision_surface_warm import FullBodyPreview
    else:from full_body_preview_collision_surface import FullBodyPreview
   elif c.get('separating_points'):
    if c['warm_start']:from full_body_preview_collision_separating_warm import FullBodyPreview
    else:from full_body_preview_collision_separating import FullBodyPreview
   elif c.get('damp_first_control'):
    if c['warm_start']:from full_body_preview_collision_first_damped_warm import FullBodyPreview
    else:from full_body_preview_collision_first_damped import FullBodyPreview
   elif c.get('recover_joint_reserve'):
    if c['warm_start']:from full_body_preview_collision_recover_margin_warm import FullBodyPreview
    else:from full_body_preview_collision_recover_margin import FullBodyPreview
   elif c.get('exact_rotation'):
    if c['warm_start']:from full_body_preview_collision_exact_rotation_warm import FullBodyPreview
    else:from full_body_preview_collision_exact_rotation import FullBodyPreview
   elif c.get('release_points'):
    if c['warm_start']:from full_body_preview_collision_fine_release_warm import FullBodyPreview
    else:from full_body_preview_collision_fine_release import FullBodyPreview
   elif c['warm_start']:from full_body_preview_collision_fine_warm import FullBodyPreview
   else:from full_body_preview_collision_fine import FullBodyPreview
  elif c['warm_start']:from full_body_preview_collision_warm import FullBodyPreview
  else:from full_body_preview_collision import FullBodyPreview
 elif c['warm_start']:from full_body_preview_warm import FullBodyPreview
 else:from full_body_preview import FullBodyPreview
 raw=np.load(c['reference_path']);base=torch.tensor(raw['control_q']);source=torch.tensor(raw['source_q'])
 reference=MotionInterpolator(base)
 if int(raw['smooth_stride']):reference=BlendedSplineReference(reference,base,stride=int(raw['smooth_stride']),start_frame=float(raw['smooth_start'])*float(raw['fps']),ramp_frames=float(raw['smooth_ramp'])*float(raw['fps']))
 tree=ast.parse(Path('examples/test_g1.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interpolate_motion_trajectory');namespace={'torch':torch,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original_source_sampler>','exec'),namespace)
 original=SimpleNamespace(kinematics=lambda f,r:(namespace['interpolate_motion_trajectory'](source,f),None,None))
 model=(ReferenceDynamics(c['urdf'],c['names'],c['collision_geometry']) if c.get('collision_geometry') is not None else ReferenceDynamics(c['urdf'],c['names']))
 planner=FullBodyPreview(model,reference,original,c['modes'],c['geometry'],c['limits'],c['effort'],c['velocity'],horizon=c['horizon'],guarded=c['guarded'])
 if c.get('native_contact_timing'):
  from reference_contact_schedule import delayed_touchdowns,native_reference_clearance
  clearance=native_reference_clearance(planner,float(raw['fps']))
  planner.modes,changes=delayed_touchdowns(planner.modes,clearance,c['contact_timing_start_frame'])
  if c.get('contact_schedule_path'):
   import json
   destination=Path(c['contact_schedule_path'])
   np.savez_compressed(destination,original_modes=c['modes'],modes=planner.modes,clearance=clearance)
   destination.with_suffix('.json').write_text(json.dumps(changes,indent=2)+'\n')
 return planner

output=sys.stdout.buffer;inputs=sys.stdin.buffer
try:
 configuration=receive(inputs)
 with contextlib.redirect_stdout(sys.stderr):planner=initialize(configuration)
 send(output,dict(ready=True))
except Exception:
 send(output,dict(error=traceback.format_exc()));raise SystemExit(1)
while True:
 try:job=receive(inputs)
 except EOFError:break
 try:
  with contextlib.redirect_stdout(sys.stderr):
   result=planner.refine(**job)
   if result['success']:
    import numpy as np
    from preview_manifold import integrate
    physical_q=[];physical_v=[]
    for k,frame in enumerate(result['frames']):
     rq,rv,_=[x.numpy().ravel() for x in planner.reference.kinematics(float(frame),job['rate'])]
     physical_q.append(integrate(rq,result['states'][k,:35]));physical_v.append(rv+result['states'][k,35:])
    result['physical_q']=np.asarray(physical_q);result['physical_v']=np.asarray(physical_v)
    result['contact_modes']=np.array([job['first']['mode']]+[planner.planned_mode(f) for f in result['frames'][1:]])
  send(output,dict(result=result))
 except Exception:send(output,dict(error=traceback.format_exc()))
