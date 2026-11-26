"""Nominal material-point contact velocity audit; no simulator stepping."""
from pathlib import Path
import json,pickle,hashlib,argparse
import numpy as np,torch
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from mujoco_reference_dynamics import ReferenceDynamics
from fabrics_sim.reference.interpolation import MotionInterpolator
parser=argparse.ArgumentParser();parser.add_argument('candidate');parser.add_argument('--name',required=True);args=parser.parse_args();p=Path(__file__).parent
path=p/(args.name+'.contact_kinematics.json')
if path.exists():raise RuntimeError('Existing audit report')
c=np.load(args.candidate);q=c['control_q'] if 'control_q' in c else c['rejected_or_unexecuted_control_q'];t=torch.tensor(q.astype(np.float32),dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
record=np.load(p/'rolling_contact.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',record['final_joint_order']);plan=json.load(open(p/'rolling_contact.steps_plan.json'));points=[np.array(x) for x in plan['collision_points']];radii=[np.array(x) for x in plan['collision_radii']];contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);events=np.flatnonzero(np.any(contacts[1:]!=contacts[:-1],axis=1))+1
rows=[]
for f in np.arange(720.,1035.01,.5):
 pose,v,a=[x.numpy()[0] for x in curve.kinematics(float(f),240.)];state=model.evaluate(pose,v,a,kinematic_bias=True)
 for foot in np.flatnonzero(contacts[int(f)]):
  R=Rotation.from_quat(state['feet'][foot,3:]).as_matrix();offsets=points[foot]@R.T;offsets[:,2]-=radii[foot];height=state['feet'][foot,2]+offsets[:,2];k=int(height.argmin());twist=state['jacobian'][foot]@v;material_v=twist[:3]+np.cross(twist[3:],offsets[k]);acc=state['jacobian'][foot]@a+state['foot_bias'][foot];material_a=acc[:3]+np.cross(acc[3:],offsets[k])+np.cross(twist[3:],np.cross(twist[3:],offsets[k]))
  rows.append(dict(frame=float(f),foot='left' if foot==0 else 'right',mode=int(contacts[int(f),0])+2*int(contacts[int(f),1]),distance_to_contact_event_frames=float(np.min(abs(events-f))),lowest_surface_height_m=float(height[k]),lowest_material_point_speed_m_s=float(np.linalg.norm(material_v)),lowest_material_point_velocity_m_s=material_v.tolist(),material_point_acceleration_m_s2=material_a.tolist()))
interior=[r for r in rows if r['distance_to_contact_event_frames']>=6];grounded=[r for r in interior if abs(r['lowest_surface_height_m'])<=.002]
report=dict(candidate=args.candidate,candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),scope='Nominal material-point velocity of the lowest authored collision sphere under source contact labels. Nonzero velocity is inconsistent with sticking/ideal rolling at that point. Does not establish actual contact or impact; nonzero material acceleration alone is not a rolling-contact failure.',frame_range=[720,1035],frame_rate=240.,sample_count=len(rows),interior_excludes_event_distance_frames=6,peak_interior_speed=max(interior,key=lambda r:r['lowest_material_point_speed_m_s']),peak_near_ground_interior_speed=max(grounded,key=lambda r:r['lowest_material_point_speed_m_s']) if grounded else None,interior_speed_p95_m_s=float(np.quantile([r['lowest_material_point_speed_m_s'] for r in interior],.95)),physical_tracking_certified=False)
path.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(p/(args.name+'.contact_kinematics.npz'),rows_json=np.asarray(json.dumps(rows)));print(json.dumps(report,indent=2))
