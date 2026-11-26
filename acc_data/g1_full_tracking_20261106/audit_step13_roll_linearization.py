"""Finite-difference audit of world-roll fitting through the runtime sampler."""
from pathlib import Path
import json
import numpy as np,torch
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from reference_smoothing_basis import runtime_smoothing_basis
from mujoco_reference_dynamics import ReferenceDynamics
from full_effort_linearization import effort_variations,inverse_output
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;seed=np.load(p/'step13_turn_roll_seed.npz');q=seed['rejected_or_unexecuted_control_q'];knots=np.r_[np.full(4,696),np.arange(708,1092,12),np.full(4,1092)];K=len(knots)-4;raw_basis=BSpline(knots,np.eye(K),3);basis=runtime_smoothing_basis(raw_basis,len(q),6);frames=np.array([760.,798.,960.,1000.]);feet=[1,0,1,1];model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'rolling_contact.npz')['final_joint_order'])
def curve(poses):
 t=torch.tensor(poses,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
c=curve(q);data=[[x.numpy()[0] for x in c.kinematics(float(f),240.)] for f in frames];qs,vs,acc=tuple(np.array([row[i] for row in data]) for i in range(3));T=np.zeros((4,35,1));T[:,3,0]=1;maps=effort_variations(model,qs,vs,acc,T,np.zeros_like(T),frames/240.,range(4),feet);rows=[];eps=1e-5
for i,f in enumerate(frames):
 k=int(np.argmax(abs(basis(f))));value,P,V,A=maps[i];expected=(P*basis(f)[k]+V*basis(f,1)[k]*240+A*basis(f,2)[k]*240**2).ravel();results=[]
 for sign in [1,-1]:
  sample=q.copy();rotation=np.zeros((len(q),3));rotation[:,0]=sign*eps*raw_basis(np.arange(len(q)))[:,k];sample[:,3:7]=(Rotation.from_rotvec(rotation)*Rotation.from_quat(q[:,3:7])).as_quat();pose,v,a=[x.numpy()[0] for x in curve(sample).kinematics(float(f),240.)];results.append(inverse_output(model.evaluate(pose,v,a),feet[i]))
 fd=(results[0]-results[1])/(2*eps);rows.append(dict(frame=float(f),coefficient=k,maximum_derivative_absolute_error=float(abs(fd-expected).max()),derivative_relative_norm_error=float(np.linalg.norm(fd-expected)/max(np.linalg.norm(fd),1e-12)),maximum_predicted_output_error_for_002_rad_increment=float(abs(fd-expected).max()*.02)))
report=dict(scope='Audit of approximate runtime-smoothed roll tangent linearization. Final nonlinear dynamics checks remain required.',samples=rows)
(p/'step13_roll_linearization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
