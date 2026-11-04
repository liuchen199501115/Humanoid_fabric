"""Nearby contact geometry for spawn preparation or reference planning.

The pure function returns a new pose, never changing its input or a simulation.
The caller may use it once at spawn or to derive a consistent landing height.
It establishes geometric contact consistency, not static/dynamic feasibility.
"""
import xml.etree.ElementTree as ET
import math

import torch

from fabrics_sim.fabrics.dynamics_aware import rotation_error


def flat_foot_spawn(fabric, source, joint_names, sole_depth=.035):
    """Prepare a nearby initial pose with strict spawn displacement guards."""
    q = flat_foot_reference(fabric, source, joint_names, sole_depth)
    if (abs(q[:,2]-source[:,2]).max()>.05 or (q[:,7:]-source[:,7:]).abs().max()>.35):
        raise RuntimeError('Flat-foot spawn would change the initial pose too far')
    return q


@torch.no_grad()
def flat_foot_reference(fabric, source, joint_names, sole_depth=.035,
                        foot_poses=None, com_xy=None, _preserve_target_pose=False,
                        com_height=None, tolerance=2e-5):
    """Solve reference-only flat-sole geometry within physical joint limits.

    By default world foot XY/yaw and pelvis XY/orientation are retained.
    Explicit ``foot_poses`` and ``com_xy`` instead fit the measured stance
    anchor, next footstep and horizontal COM endpoint by freeing pelvis XYZ.
    Pelvis orientation and upper-body joints are retained in both cases.
    This pose is for planning; it must not be used to reset a live robot.
    Unlike initial spawning, rolling-foot source frames can require larger
    ankle/knee changes to derive a consistent flat-ground COM height.
    """
    if (source.shape != (1,36) or not torch.isfinite(source).all()
            or source[:,3:7].norm(dim=-1).min()<1e-8
            or not math.isfinite(sole_depth) or sole_depth<=0
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Flat-foot initialization requires one finite G1 pose')
    if fabric.dynamics_tasks is None:
        raise ValueError('Flat-foot initialization requires the physical kinematics model')
    model=fabric.dynamics_tasks.model
    feet=[model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
    legs=[joint_names.index(side+'_'+name+'_joint') for side in ('left','right')
          for name in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    if (foot_poses is None) != (com_xy is None):
        raise ValueError('Explicit landing feet and COM XY must be supplied together')
    if com_height is not None and (com_xy is None or not math.isfinite(com_height) or com_height<=0):
        raise ValueError('Explicit COM height requires contact poses, COM XY and a positive finite height')
    if _preserve_target_pose and foot_poses is None:
        raise ValueError('Preserving contact poses requires explicit targets')
    if foot_poses is not None:
        for value, shape in ((foot_poses,(1,2,7)),(com_xy,(1,2))):
            if (value.shape != shape or value.dtype != source.dtype or value.device != source.device
                    or not torch.isfinite(value).all()):
                raise ValueError('Landing geometry must match source batch, dtype and device')
    columns=([2] if foot_poses is None else [0,1,2])+[6+i for i in legs]
    source=source.detach().clone()
    initial=model.snapshot(source)
    target_position=initial['transforms'][:,feet,:3].clone()
    target_position[:,:,2]=sole_depth
    rotations=initial['rotations'][:,feet].clone()
    if foot_poses is not None:
        from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix
        if (foot_poses[...,3:].norm(dim=-1)<1e-8).any():
            raise ValueError('Landing foot quaternions must be nonzero')
        target_position=foot_poses[...,:3].clone()
        if not _preserve_target_pose:
            target_position[:,:,2]=sole_depth
        rotations=_quat_xyzw_to_matrix(foot_poses[...,3:])
    yaw=torch.atan2(rotations[:,:,1,0],rotations[:,:,0,0])
    target_rotation=torch.eye(3,device=source.device,dtype=source.dtype).expand(1,2,3,3).clone()
    target_rotation[:,:,0,0]=yaw.cos();target_rotation[:,:,0,1]=-yaw.sin()
    target_rotation[:,:,1,0]=yaw.sin();target_rotation[:,:,1,1]=yaw.cos()
    if _preserve_target_pose:
        target_rotation=rotations
    limits={node.attrib['name']:node.find('limit') for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    lower=source.new_tensor([float(limits[n].attrib['lower']) for n in joint_names])+.005
    upper=source.new_tensor([float(limits[n].attrib['upper']) for n in joint_names])-.005
    if ((source[:,7:]<lower-.005)|(source[:,7:]>upper+.005)).any():
        raise ValueError('Source initialization is outside physical joint limits')
    from fabrics_sim.reference.posture import crouched_posture
    q=source.clone()
    # Explicit source contact poses can already be feasible. Seeding their
    # redundant IK with a forced crouch silently lowers the planned COM even
    # when no correction is needed. Retain the source seed for rolling poses;
    # flat-foot spawn preparation still uses its singularity-avoiding seed.
    if not _preserve_target_pose:
        q[:,7:],_=crouched_posture(q[:,7:],joint_names,1.5,minimum_knee=.2)
    com_dimensions = 2 if com_height is None else 3
    com_target = com_xy
    if com_height is not None:
        com_target = torch.cat((com_xy, source.new_full((1,1), com_height)), dim=1)
    identity=torch.eye(12 if com_xy is None else 12+com_dimensions,device=q.device,dtype=q.dtype)[None]
    for _ in range(80):
        state=model.snapshot(q)
        error=torch.cat((state['transforms'][:,feet,:3]-target_position,
                         rotation_error(state['rotations'][:,feet],target_rotation)),dim=-1).flatten(1)
        if com_xy is not None:
            error=torch.cat((error,state['com'][:,:com_dimensions]-com_target),dim=1)
        if error.abs().max()<tolerance:
            break
        jac=torch.cat((state['origin_jac'][:,feet],state['angular_jac'][:,feet]),dim=2).flatten(1,2)[:,:,columns]
        if com_xy is not None:
            jac=torch.cat((jac,state['com_jac'][:,:com_dimensions,columns]),dim=1)
        delta=-(jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+1e-5*identity,error[...,None])).squeeze(-1)
        delta*=torch.clamp(.1/delta.abs().amax(-1,keepdim=True).clamp_min(1e-8),max=1.)
        if com_xy is None:
            q[:,2]+=delta[:,0]
            q[:,[7+i for i in legs]]+=delta[:,1:]
        else:
            q[:,:3]+=delta[:,:3]
            q[:,[7+i for i in legs]]+=delta[:,3:]
        leg_positions=[7+i for i in legs]
        q[:,leg_positions]=torch.maximum(lower[legs],torch.minimum(upper[legs],q[:,leg_positions]))
    else:
        saturated = [joint_names[i] for i in legs
                     if bool((q[:,7+i] <= lower[i]+1e-5).any()
                             or (q[:,7+i] >= upper[i]-1e-5).any())]
        raise RuntimeError(f'Could not find joint-limited flat-foot reference geometry: '
                           f'residual={error.abs().max().item():.8g}, tolerance={tolerance:g}, '
                           f'saturated_joints={saturated}')
    return q


def contact_pose_reference(fabric, source, joint_names, foot_poses, com_xy, com_height=None, *, tolerance=2e-5):
    """Fit explicit foot contact poses, including a rolled stance foot.

    Same joint limits, source pelvis orientation and upper-body preservation
    as flat_foot_reference. Contact support is verified separately by the caller.
    """
    return flat_foot_reference(fabric,source,joint_names,foot_poses=foot_poses,
                               com_xy=com_xy,_preserve_target_pose=True,com_height=com_height,tolerance=tolerance)
