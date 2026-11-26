from pathlib import Path
import numpy as np,torch,json,pickle,time
from full_body_preview_anchored import FullBodyPreview
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);source=MotionInterpolator(torch.tensor(np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];history=np.load(p/'fixed_clock120_body10_overdue.npz');velocity=history['history_joint_velocity_limits'][0].ravel() if 'history_joint_velocity_limits' in history else np.full(29,30.);reports=[]


import ast,math
from types import SimpleNamespace
run='fixed_clock120_body10_full_body_preview42_semi_process24'
module=ast.parse(Path('examples/test_g1.py').read_text());node=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='interpolate_motion_trajectory');namespace={'torch':torch,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original_source_sampler>','exec'),namespace)
original=torch.tensor(np.load(p/(run+'.control_reference.npz'))['source_q']);source=SimpleNamespace(kinematics=lambda f,r:(namespace['interpolate_motion_trajectory'](original,f),None,None))
d=np.load(p/(run+'.full_body_preview.failure_plan.npz'),allow_pickle=True);first={k:d['problem_'+k] for k in ['A_eq','b_eq','G','b','lb','ub']};first['mode']=int(d['problem_mode']);frame=float(d['frame']);rate=float(d['rate'])
first['contact_patch_indices']=d['problem_contact_patch_indices'].item()
for iterations in [30]:
 preview=FullBodyPreview(model,reference,source,modes,geometry,saved['joint_position_limits'][0],saved['effort_limits'].ravel(),velocity,horizon=24,guarded=True);start=time.monotonic();result=preview.refine(d['q'],d['v'],frame,rate,first,iterations=iterations);np.savez_compressed(p/f'full_body_preview_anchored_current_{iterations}.npz',**result);summary={k:v for k,v in result.items() if k not in ['first_control','controls','states','frames','impulses','impact_maps','patch_impact_nodes','contact_patch_indices','impact_patch_indices','point_forces','impulse_point_forces','step_durations']};summary.update(time=float(d['time']),frame=frame,maximum_iterations=iterations,solve_seconds=time.monotonic()-start);reports.append(summary);print(summary,flush=True)
 if result['success']:break
(p/'full_body_preview_anchored_current_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
