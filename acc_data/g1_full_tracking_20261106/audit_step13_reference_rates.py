"""Constant phase-rate inverse-dynamics diagnostics; no commands are applied."""
from pathlib import Path
import json,pickle
import numpy as np,torch
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from mujoco_reference_dynamics import ReferenceDynamics
from full_effort_linearization import inverse_output
from g1_model_effort import model_effort_limits
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
p=Path(__file__).parent;d=np.load(p/'rolling_contact.npz');names=d['final_joint_order'];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';model=ReferenceDynamics(urdf,names);caps=.8*model_effort_limits(urdf,names,d['effort_limits'].reshape(-1));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);reports=[]
for name,path in [('baseline','acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz'),('force_fit',p/'thirteenth_force_fit150.npz')]:
 q=torch.tensor(np.load(path)['control_q'],dtype=torch.float64);sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
 for rate in [.6,.8,.9,1.,1.1,1.2]:
  samples=[]
  for frame in np.arange(960.,1011.1,3.):
   if contacts[int(frame)].sum()!=1:continue
   foot=int(np.flatnonzero(contacts[int(frame)])[0]);pose,v,a=[x.numpy()[0] for x in sampler.kinematics(frame,200*rate)];state=model.evaluate(pose,v,a);value=inverse_output(state,foot);R=Rotation.from_quat(state['feet'][:,3:]).as_matrix();G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([foot+1]),foot_rotation_world=torch.tensor(R[None]),friction_coefficient=.6,coupled_yaw=True,toe_length=.085,heel_length=.025,half_width=.02,sole_depth=.035,minimum_normal_force=20.,maximum_normal_force=800.);G=G.numpy()[0,:,6*foot:6*foot+6];b=b.numpy()[0];active=np.any(abs(G)>0,axis=1);G,b=G[active],b[active];scale=np.linalg.norm(G*np.array([350.,350.,350.,35.,35.,35.]),axis=1);j=int(np.argmax(abs(value[6:])/caps));samples.append(dict(frame=float(frame),effort_fraction=float(abs(value[6+j])/caps[j]),joint=str(names[j]),torque=float(value[6+j]),wrench_violation=float(np.maximum((G@value[:6]-b)/scale,0).max())))
  row=dict(reference=name,phase_rate=rate,worst_effort=max(samples,key=lambda x:x['effort_fraction']),worst_wrench=max(samples,key=lambda x:x['wrench_violation']));reports.append(row);print(json.dumps(row),flush=True)
(p/'step13_reference_rates.json').write_text(json.dumps(dict(scope='Constant rate, optimistic full sole; does not include clock-transition acceleration or certify contact kinematics/physical tracking.',cases=reports),indent=2)+'\n')
