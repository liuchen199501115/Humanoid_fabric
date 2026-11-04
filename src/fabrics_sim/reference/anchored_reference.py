"""Kinematic reference projection onto fixed measured contacts, never a state reset."""
import torch
from .initial_stance import contact_pose_reference
from .whole_body import WholeBodyReference


@torch.no_grad()
def anchor_reference(fabric, source, names, anchors, contacts, *, seed=None, com_offset=None, held_upper=None, held_pelvis=None):
    """Keep source swing/body targets; optionally offset COM and fix stance derivatives.

    Requires one fixed contact mode. The caller owns touchdown anchors and must
    plan continuous transitions and audit dynamics before executing this result.
    """
    if (source.q.shape!=(1,36) or anchors.shape!=(1,2,7) or contacts.shape!=(1,2)
        or contacts.dtype!=torch.bool or not contacts.any() or not torch.isfinite(anchors).all()
        or anchors.device!=source.q.device or anchors.dtype!=source.q.dtype
        or contacts.device!=source.q.device
        or not torch.allclose(anchors[...,3:].norm(dim=-1),anchors.new_ones(1,2),atol=1e-5,rtol=1e-5)):
        raise ValueError('Expected one G1 reference, unit measured foot poses and nonempty Boolean contacts')
    model=fabric.dynamics_tasks.model;eps=fabric.dynamics_tasks.config['difference_step']
    feet=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
    columns=[0,1,2]+[6+names.index(s+'_'+j+'_joint') for s in ('left','right')
                     for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    def jac(st):
        return torch.cat((torch.cat((st['origin_jac'][:,feet],st['angular_jac'][:,feet]),2).flatten(1,2),st['com_jac']),1)
    def bias(st):
        return torch.cat((torch.cat((st['origin_jac_bias'][:,feet],st['angular_jac_bias'][:,feet]),2).flatten(1),st['com_jac_bias']),1)
    st,_,_=model.evaluate(source.q,source.v,fabric.integrate_root_position,eps)
    target_feet=torch.where(contacts[...,None],anchors,st['transforms'][:,feet])
    target_v=(jac(st)@source.v[...,None]).squeeze(-1)
    target_a=(jac(st)@source.a[...,None]).squeeze(-1)+bias(st)
    for side in range(2):
        if bool(contacts[0,side]):
            target_v[:,6*side:6*(side+1)]=0
            target_a[:,6*side:6*(side+1)]=0
    com_target=st['com'].clone()
    if com_offset is not None:
        if (com_offset.shape!=(1,3) or com_offset.dtype!=source.q.dtype
            or com_offset.device!=source.q.device or not torch.isfinite(com_offset).all()
            or float(com_offset.norm())>.15):
            raise ValueError('Constant world COM offset must be finite [1,3] and at most 0.15 m')
        com_target+=com_offset
    # A constant world offset changes position only. Its time derivatives vanish.
    start=source.q.clone()
    if held_upper is not None:
        if (held_upper.shape!=(1,17) or held_upper.dtype!=source.q.dtype
            or held_upper.device!=source.q.device or not torch.isfinite(held_upper).all()):
            raise ValueError('Held waist/arm positions must be finite [1,17] and match reference')
        # Targets above stay those of the source. Only the IK body is adapted.
        start[:,19:]=held_upper
    if held_pelvis is not None:
        if (held_pelvis.shape!=(1,4) or held_pelvis.dtype!=source.q.dtype
            or held_pelvis.device!=source.q.device or not torch.isfinite(held_pelvis).all()
            or not torch.allclose(held_pelvis.norm(dim=-1),source.q.new_ones(1),atol=1e-5,rtol=1e-5)):
            raise ValueError('Held pelvis orientation must be a unit xyzw quaternion [1,4]')
        start[:,3:7]=held_pelvis
    if seed is not None:
        if seed.shape!=source.q.shape or seed.dtype!=source.q.dtype or seed.device!=source.q.device or not torch.isfinite(seed).all():
            raise ValueError('IK seed must match the finite reference configuration')
        positions=[0,1,2]+[column+1 for column in columns[3:]]
        start[:,positions]=seed[:,positions]
    q=contact_pose_reference(fabric,start,names,target_feet,com_target[:,:2],float(com_target[0,2]),tolerance=2e-6)
    projected=model.snapshot(q);J=jac(projected)
    v=source.v.clone()
    if held_pelvis is not None: v[:,3:6]=0
    if held_upper is not None: v[:,18:]=0
    v[:,columns]=0
    v[:,columns]=torch.linalg.solve(J[:,:,columns],(target_v-(J@v[...,None]).squeeze(-1))[...,None]).squeeze(-1)
    projected,_,_=model.evaluate(q,v,fabric.integrate_root_position,eps)
    J=jac(projected);a=source.a.clone()
    if held_pelvis is not None: a[:,3:6]=0
    if held_upper is not None: a[:,18:]=0
    a[:,columns]=0
    a[:,columns]=torch.linalg.solve(J[:,:,columns],(target_a-bias(projected)-(J@a[...,None]).squeeze(-1))[...,None]).squeeze(-1)
    return WholeBodyReference(q,v,a)
