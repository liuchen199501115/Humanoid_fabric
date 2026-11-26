from pathlib import Path
import numpy as np,torch,json,pickle,time
from full_body_preview import FullBodyPreview
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);source=MotionInterpolator(torch.tensor(np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];history=np.load(p/'fixed_clock120_body10_overdue.npz');velocity=history['history_joint_velocity_limits'][0].ravel() if 'history_joint_velocity_limits' in history else np.full(29,30.);reports=[]

import cProfile,pstats
snapshot=np.load(p/'fixed_clock120_body10_preview_snapshots.snapshot_3.700.npz');first={k:snapshot['problem_'+k][0] for k in ['A_eq','b_eq','G','b','lb','ub']};first['mode']=int(snapshot['contact_mode'].ravel()[0]);old=np.load(p/'full_body_preview_long_3.700.npz',allow_pickle=True);nominal={k:old[k] for k in ['states','controls']};preview=FullBodyPreview(model,reference,source,modes,geometry,saved['joint_position_limits'][0],saved['effort_limits'].ravel(),velocity,horizon=24)
profiler=cProfile.Profile();profiler.enable();result=preview.solve(snapshot['q'][0].astype(float),snapshot['v'][0].astype(float),845.,240.,first,nominal=nominal);profiler.disable();profiler.dump_stats(str(p/'full_body_preview_profile.pstats'));pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(25);print(result['status'])
