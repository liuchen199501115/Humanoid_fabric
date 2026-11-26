from pathlib import Path
import json,numpy as np,torch,warp as wp
from scipy.spatial.transform import Rotation
from test_g1_two_tasks import load_motion_trajectory,load_motion_contact_sequence
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');model=f.dynamics_tasks.model
q,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'));sampler=MotionInterpolator(q)
plan=json.loads(Path('acc_data/amp_walk_20261102/contact_clock_followup/vertical_feedback_verified.steps_plan.json').read_text());points=np.array(plan['collision_points']);radii=np.array(plan['collision_radii']);ids=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
rows=[]
with torch.no_grad():
 for i,e in enumerate(plan['events']):
  side=1-e['support'];onset=e['end_frame'];samples=[]
  for k in range(max(0,onset-2),min(onset+25,len(q))):
   pose,v,_=sampler.kinematics(k,fps);st=model.snapshot(pose);link=ids[side];R=st['rotations'][0,link].numpy();offsets=(R@points[side].T).T;bottom=offsets[:,2]-radii[side];j=int(bottom.argmin());offset=offsets[j]-np.array([0,0,radii[side,j]])
   contact_velocity=(st['origin_jac'][0,link].numpy()-skew(offset)@st['angular_jac'][0,link].numpy())@v[0].numpy()
   samples.append(dict(frame=k,contact_speed=float(np.linalg.norm(contact_velocity)),velocity=contact_velocity.tolist()))
  low=next((r['frame'] for r in samples if r['frame']>=onset and r['contact_speed']<.2),None)
  row=dict(step=i+1,onset=onset,source_speed_at_onset=next(r['contact_speed'] for r in samples if r['frame']==onset),first_below_point2=low,samples=samples);rows.append(row)
print([(r['step'],r['onset'],round(r['source_speed_at_onset'],3),r['first_below_point2']) for r in rows],flush=True)
(p/'source_touchdown_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
