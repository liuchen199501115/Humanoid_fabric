from pathlib import Path
import numpy as np,torch,json,pickle,time
from full_body_preview import FullBodyPreview
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);source=MotionInterpolator(torch.tensor(np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];history=np.load(p/'fixed_clock120_body10_overdue.npz');velocity=history['history_joint_velocity_limits'][0].ravel() if 'history_joint_velocity_limits' in history else np.full(29,30.);reports=[]

for name in ['fixed_clock120_body10_full_body_preview42_contact24','fixed_clock120_body10_full_body_preview42_guarded24']:
 d=np.load(p/(name+'.full_body_preview.first_plan.npz'),allow_pickle=True);result={k:d[k] for k in ['states','controls','frames','impulses']};result['impact_maps']=d['impact_maps'].item();preview=FullBodyPreview(model,reference,source,modes,geometry,saved['joint_position_limits'][0],saved['effort_limits'].ravel(),velocity,horizon=len(result['controls']));audit=preview.nonlinear_audit(result,float(d['rate']));audit['run']=name;reports.append(audit);print(audit,flush=True)
(p/'full_body_preview_legacy_wrench_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
