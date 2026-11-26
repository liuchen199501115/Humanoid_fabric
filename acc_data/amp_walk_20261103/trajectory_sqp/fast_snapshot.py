"""Same URDF geometry as G1CentroidalModel.snapshot, without unused momentum reductions."""
import torch
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotKinematics
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix,_skew

def snapshot(model,q):
 with torch.no_grad():
  transforms,jac=RobotKinematics.apply(torch.tensor(q,dtype=torch.float32),model.kinematics)
  transforms,jac=transforms.clone(),jac.transpose(2,3).clone()
  rotations=_quat_xyzw_to_matrix(transforms[...,3:])
  angular=(model.kinematics.batch_joint_axes_torch[:,None]*model.com_task.angular_mask[None,:,:,None]).transpose(2,3).clone()
  offsets=(rotations@model.com_task.link_com_offsets[None,:,:,None]).squeeze(-1)
  result=dict(transforms=transforms,rotations=rotations,origin_jac=jac,angular_jac=angular,link_com=transforms[...,:3]+offsets,link_com_jac=jac-_skew(offsets)@angular)
  return {k:v.numpy().astype(float) for k,v in result.items()}
