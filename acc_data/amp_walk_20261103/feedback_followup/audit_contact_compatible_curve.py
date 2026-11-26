from pathlib import Path
import json
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent
candidate=np.load(p/'contact_compatible_b15.npz');z=np.load(p/'fixed_smooth_early.control_reference.npz');tensor=torch.tensor(candidate['control_q'])
curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
original=MotionInterpolator(torch.tensor(z['source_q']));baseline=MotionInterpolator(torch.tensor(z['control_q']))
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order'])
frames=np.arange(0,1999.01,.5);q=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in frames]);source=np.array([original.kinematics(f,200)[0].numpy()[0] for f in frames]);limits=np.load(p/'wrench_fixed_ten.npz')['joint_position_limits'][0];margin=float(np.minimum(q[:,7:]-limits[:,0],limits[:,1]-q[:,7:]).min())
# Finite differences in float64 at the interpolator's raw curve resolution.
# Float32 output requires an appropriately sized time stencil.
velocity_errors=[];acceleration_errors=[];dt=.0005
for f in [131.3,139.1,148.9,159.3,175.3,230.3,255.3,674.3,828.3,1235.3,1497.3,1910.3,1979.3]:
 q0,v,a=curve.kinematics(f,200);qm,vm,_=curve.kinematics(f-dt*200,200);qp,vp,_=curve.kinematics(f+dt*200,200)
 velocity_errors.append(float(np.max(abs((qp.numpy()[0,7:]-qm.numpy()[0,7:])/(2*dt)-v.numpy()[0,6:]))))
 acceleration_errors.append(float(np.max(abs((vp.numpy()[0,6:]-vm.numpy()[0,6:])/(2*dt)-a.numpy()[0,6:]))))
report=dict(candidate='contact_compatible_b15.npz',dense_sample_count=len(frames),minimum_joint_limit_margin=margin,peak_joint_reference_rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=1)).max()),peak_root_position_delta=float(np.linalg.norm(q[:,:3]-source[:,:3],axis=1).max()),initial_qva_equal=all(torch.equal(a,b) for a,b in zip(curve.kinematics(0,200),baseline.kinematics(0,200))),joint_velocity_finite_difference_max_error=max(velocity_errors),joint_acceleration_finite_difference_max_error=max(acceleration_errors),physical_tracking_certified=False)
report['curve_audit_passed']=bool(margin>=0 and report['peak_joint_reference_rmse']<.19 and report['peak_root_position_delta']<.19 and report['initial_qva_equal'] and max(velocity_errors)<.02 and max(acceleration_errors)<1.)
(p/'contact_compatible_b15.curve_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
