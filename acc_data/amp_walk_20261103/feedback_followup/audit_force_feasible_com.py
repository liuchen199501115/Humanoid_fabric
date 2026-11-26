from pathlib import Path
import sys,json,pickle,numpy as np,torch
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';com=CenterOfMassModel(urdf,d['final_joint_order']);leg=LegKinematics(urdf,d['final_joint_order']);contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);frames=np.arange(740,1000.01,.25);records=[]
for name in (sys.argv[1:] or ['turn_recovery_y40','turn_force_feasible_com_floor']):
 t=torch.tensor(np.load(p/(name+'.npz'))['control_q']);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240.) for f in frames];states=np.array([com.momentum(q.numpy()[0],v.numpy()[0],return_state=True) for q,v,a in values]);derivative=CubicSpline(frames/240,states[:,1:])(frames/240,1);force=com.mass*(derivative[:,0]+[0,0,9.81]);hdot=derivative[:,1];cop=states[:,0,:2]-states[:,0,2,None]*force[:,:2]/force[:,2,None]+np.c_[-hdot[:,1],hdot[:,0]]/force[:,2,None];violations=[];rotated_friction=[]
 for i,f in enumerate(frames):
  vertices=[];ratios=[]
  for side in range(2):
   if not contacts[int(f),side]:continue
   pos,R,_=leg.evaluate(values[i][0].numpy()[0],side);corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]]);vertices.extend((pos+corners@R.T)[:,:2]);yaw=np.arctan2(R[1,0],R[0,0]);c,s=np.cos(yaw),np.sin(yaw);xy=np.array([[c,s],[-s,c]])@force[i,:2];ratios.append(np.sum(abs(xy))/force[i,2])
  if len(vertices)>=3:
   hull=ConvexHull(np.array(vertices));violations.append(max(0.,float(np.max(hull.equations[:,:2]@cop[i]+hull.equations[:,2]))));rotated_friction.append(max(ratios))
 record=dict(reference=name,source_frames=[740,1000],dense_samples=len(frames),minimum_normal_force=float(force[:,2].min()),maximum_normal_force=float(force[:,2].max()),negative_force_samples=int((force[:,2]<=0).sum()),maximum_world_friction_l1=float((np.sum(abs(force[:,:2]),axis=1)/force[:,2]).max()),maximum_planned_sole_friction_l1=float(max(rotated_friction)),maximum_cop_outside_planned_sole_hull=float(max(violations)),peak_momentum_rate=np.max(abs(hdot),axis=0).tolist(),physical_tracking_certified=False)
 np.savez_compressed(p/(name+'.dense_centroidal.npz'),frames=frames,force=force,momentum_rate=hdot,cop=cop,com=states[:,0],velocity=states[:,1]);records.append(record);print(record,flush=True)
(p/('centroidal_pose_fit_audit.json' if sys.argv[1:] else 'force_feasible_com_audit.json')).write_text(json.dumps(records,indent=2)+'\n')
