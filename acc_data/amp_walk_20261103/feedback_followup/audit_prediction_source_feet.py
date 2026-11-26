from pathlib import Path
import ast,math,json,numpy as np,torch
from mujoco_reference_dynamics import ReferenceDynamics
from full_body_preview import integrate
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20)
module=ast.parse(Path('examples/test_g1.py').read_text());node=next(x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='interpolate_motion_trajectory');namespace={'torch':torch,'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original_source_sampler>','exec'),namespace)
source=torch.tensor(np.load(p/'fixed_clock120_body10_full_body_preview42_distributed12.control_reference.npz')['source_q']);reports=[]
for name in ['full_body_preview_frozen_30','full_body_preview_frozen_long_30']:
 plan=np.load(p/(name+'.npz'));rows=[]
 for k,frame in enumerate(plan['frames'][:-1]):
  q,v,_=[x.numpy().ravel() for x in reference.kinematics(float(frame),240.)];q=integrate(q,plan['states'][k,:35]);v=v+plan['states'][k,35:];actual=model.evaluate(q,v)
  original=namespace['interpolate_motion_trajectory'](source,float(frame)).numpy().ravel();target=model.evaluate(original,np.zeros(35));error=np.linalg.norm(actual['feet'][:,:3]-target['feet'][:,:3],axis=1)
  row=dict(frame=float(frame),foot_errors=error.tolist(),support_force=plan['controls'][k,[37,43]].tolist());rows.append(row)
 reports.append(dict(name=name,nodes=rows));print(name,[(round(r['frame'],1),[round(x,3) for x in r['foot_errors']],[round(x) for x in r['support_force']]) for r in rows])
(p/'prediction_source_feet_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
