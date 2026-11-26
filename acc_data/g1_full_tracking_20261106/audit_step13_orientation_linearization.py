"""Finite difference of all three Euler correction directions through runtime interpolation."""
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
p=Path(__file__).parent;seed=np.load(p/'step13_turn_orientation_seed.npz');base=np.load(p/'step13_leg_smooth5_clearance_audit.npz')['control_q'].astype(float);q=seed['rejected_or_unexecuted_control_q'];knots=np.r_[np.full(4,696),np.arange(708,1092,12),np.full(4,1092)];K=len(knots)-4;raw_basis=BSpline(knots,np.eye(K),3);basis=runtime_smoothing_basis(raw_basis,len(q),6);frames=np.arange(696.,1093.);indices=[64,102,264,304];feet=[1,0,1,1];model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'rolling_contact.npz')['final_joint_order'])
def curve(poses):
 t=torch.tensor(poses,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
def samples(poses):
 c=curve(poses);data=[[x.numpy()[0] for x in c.kinematics(float(f),240.)] for f in frames];return tuple(np.array([row[i] for row in data]) for i in range(3))
qs,vs,acc=samples(q);bq,bv,_=samples(base);relative=Rotation.from_quat(qs[:,3:7])*Rotation.from_quat(bq[:,3:7]).inv();angles=relative.as_euler('XYZ');rx=Rotation.from_euler('X',angles[:,0]).as_matrix();rxy=Rotation.from_euler('XY',angles[:,:2]).as_matrix();T=np.zeros((len(frames),35,3));T[:,3,0]=1.;T[:,3:6,1]=rx[:,:,1];T[:,3:6,2]=rxy[:,:,2];rates=np.linalg.solve(T[:,3:6],(vs[:,3:6]-relative.apply(bv[:,3:6]))[...,None])[...,0];Td=np.zeros_like(T);Td[:,3:6,1]=np.cross(T[:,3:6,0]*rates[:,0,None],T[:,3:6,1]);Td[:,3:6,2]=np.cross(T[:,3:6,0]*rates[:,0,None]+T[:,3:6,1]*rates[:,1,None],T[:,3:6,2]);maps=effort_variations(model,qs,vs,acc,T,Td,frames/240.,indices,feet);rows=[];eps=1e-5
raw_angles=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(base[:,3:7]).inv()).as_euler('XYZ')
for i,foot in zip(indices,feet):
 f=frames[i];k=int(np.argmax(abs(basis(f))));value,P,V,A=maps[i]
 for axis in range(3):
  expected=P[:,axis]*basis(f)[k]+V[:,axis]*basis(f,1)[k]*240+A[:,axis]*basis(f,2)[k]*240**2;results=[]
  for sign in [1,-1]:
   sample=q.copy();perturbed=raw_angles.copy();perturbed[:,axis]+=sign*eps*raw_basis(np.arange(len(q)))[:,k];sample[:,3:7]=(Rotation.from_euler('XYZ',perturbed)*Rotation.from_quat(base[:,3:7])).as_quat();pose,v,a=[x.numpy()[0] for x in curve(sample).kinematics(float(f),240.)];results.append(inverse_output(model.evaluate(pose,v,a),foot))
  fd=(results[0]-results[1])/(2*eps);rows.append(dict(frame=float(f),axis=['roll','pitch','yaw'][axis],relative_derivative_error=float(np.linalg.norm(fd-expected)/max(np.linalg.norm(fd),1e-12)),maximum_generalized_output_error_for_002_increment=float(abs(fd-expected).max()*.02)))
report=dict(scope='Approximate orientation tangent map including time-varying axes; nonlinear validation remains mandatory.',samples=rows,maximum_relative_derivative_error=max(r['relative_derivative_error'] for r in rows));(p/'step13_orientation_linearization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
