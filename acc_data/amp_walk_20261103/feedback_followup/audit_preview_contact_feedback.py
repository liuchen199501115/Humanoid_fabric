from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation
from mujoco_reference_dynamics import ReferenceDynamics
from centroidal_pose_linearization import skew
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_fixed_process24';d=np.load(p/(run+'.full_body_preview.first_plan.npz'),allow_pickle=True);initial=np.load(p/(run+'.full_body_preview.initial_problem.npz'),allow_pickle=True);meta=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order']);rows=[]
for k in range(6):
 q=d['physical_q'][k];v=d['physical_v'][k];a=d['controls'][k,:35];state=model.evaluate(q,v,kinematic_bias=True)
 patches=initial['problem_contact_patch_indices'].item() if k==0 else d['contact_patch_indices'].item().get(k,{})
 for side,indices in patches.items():
  J=state['jacobian'][side];foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();omega=J[3:]@v;bias=state['foot_bias'][side]
  for i in indices:
   offset=R@np.array(meta['collision_points'][side][i])-np.array([0.,0.,meta['collision_radii'][side][i]])
   jp=J[:3]-skew(offset)@J[3:];point_bias=bias[:3]-skew(offset)@bias[3:]+np.cross(omega,np.cross(omega,offset))
   rows.append(dict(stage=k,frame=float(d['frames'][k]),side=int(side),point=int(i),velocity=(jp@v).tolist(),acceleration=(jp@a+point_bias).tolist()))
print(rows);(p/'preview_contact_feedback_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
