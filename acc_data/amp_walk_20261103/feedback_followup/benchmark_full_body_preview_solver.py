from pathlib import Path
import numpy as np,torch,json,pickle,time
from full_body_preview import FullBodyPreview
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);source=MotionInterpolator(torch.tensor(np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];history=np.load(p/'fixed_clock120_body10_overdue.npz');velocity=history['history_joint_velocity_limits'][0].ravel() if 'history_joint_velocity_limits' in history else np.full(29,30.);reports=[]

import cProfile,pstats
snapshot=np.load(p/'fixed_clock120_body10_preview_snapshots.snapshot_3.700.npz');first={k:snapshot['problem_'+k][0] for k in ['A_eq','b_eq','G','b','lb','ub']};first['mode']=int(snapshot['contact_mode'].ravel()[0]);old=np.load(p/'full_body_preview_long_3.700.npz',allow_pickle=True);nominal={k:old[k] for k in ['states','controls']};preview=FullBodyPreview(model,reference,source,modes,geometry,saved['joint_position_limits'][0],saved['effort_limits'].ravel(),velocity,horizon=24)

import clarabel
real_solver=clarabel.DefaultSolver;captured={}
def capture(*args):
 captured['args']=args
 return real_solver(*args)
clarabel.DefaultSolver=capture
start=time.monotonic();baseline=preview.solve(snapshot['q'][0].astype(float),snapshot['v'][0].astype(float),845.,240.,first,nominal=nominal);baseline_time=time.monotonic()-start
P,g,A,b,cones,original_settings=captured['args'];reports=[]
for method,threads in [('auto',1),('qdldl',1),('faer',1)]:
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=150;settings.tol_gap_abs=1e-7;settings.tol_gap_rel=1e-7;settings.tol_feas=1e-7;settings.direct_solve_method=method;settings.max_threads=threads
 start=time.monotonic()
 try:
  solver=real_solver(P,g,A,b,cones,settings);solution=solver.solve();info=solver.get_info();x=np.array(solution.x);nx=70*25;report=dict(method=method,threads=threads,status=str(solution.status),seconds=time.monotonic()-start,objective=float(solution.obj_val),first_control_max_difference=float(abs(x[nx:nx+76]-baseline['first_control']).max()),solver_info=str(info.linsolver),iterations=info.iterations)
 except Exception as error:report=dict(method=method,error=str(error))
 reports.append(report);print(report,flush=True)
(p/'full_body_preview_solver_benchmark.json').write_text(json.dumps(dict(original_total_seconds=baseline_time,cases=reports),indent=2)+'\n')
