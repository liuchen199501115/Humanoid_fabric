"""Necessary single-support inverse-dynamics audit of a whole-body reference."""
import torch


def required_single_support(model, reference, integrate, support_link, armature=None, epsilon=1e-3):
    """Return world ankle wrench and actuator torque for exact reference motion.

    The full 6-D stance is assumed fixed. This is a rigid-body calculation,
    not a closed-loop tracking or impact/collision validation.
    """
    q, v, a = reference.q, reference.v, reference.a
    state, plus, minus = model.evaluate(q, v, integrate, epsilon)
    jv, jw = state['link_com_jac'], state['angular_jac']
    linear_bias = (((plus['link_com_jac']-minus['link_com_jac'])/(2*epsilon))
                   @v[:, None, :, None]).squeeze(-1)
    linear_acceleration = (jv@a[:, None, :, None]).squeeze(-1)+linear_bias
    angular_acceleration = (jw@a[:, None, :, None]).squeeze(-1)+state['angular_jac_bias']
    omega = (jw@v[:, None, :, None]).squeeze(-1)
    rotation = state['rotations']
    inertia = rotation@model.inertias[None]@rotation.transpose(-1, -2)
    force = model.com_task.link_masses[None, :, None]*(linear_acceleration+q.new_tensor([0., 0., 9.81]))
    moment = ((inertia@angular_acceleration[..., None]).squeeze(-1)
              +torch.linalg.cross(omega, (inertia@omega[..., None]).squeeze(-1)))
    generalized = ((jv.transpose(-1, -2)@force[..., None])
                   +(jw.transpose(-1, -2)@moment[..., None])).sum(1).squeeze(-1)
    if armature is not None:
        if armature.shape != (q.shape[0], 29) or not torch.isfinite(armature).all() or (armature < 0).any():
            raise ValueError('Armature must be finite nonnegative [B,29]')
        generalized[:, 6:] += armature*a[:, 6:]
    jac = torch.cat((state['origin_jac'][:, support_link], state['angular_jac'][:, support_link]), 1)
    wrench = torch.linalg.solve(jac[:, :, :6].transpose(-1, -2), generalized[:, :6, None]).squeeze(-1)
    remainder = generalized-(jac.transpose(-1, -2)@wrench[..., None]).squeeze(-1)
    contact_velocity = (jac@v[..., None]).squeeze(-1)
    contact_acceleration = ((jac@a[..., None]).squeeze(-1)
        +torch.cat((state['origin_jac_bias'][:, support_link], state['angular_jac_bias'][:, support_link]), 1))
    return dict(wrench=wrench, torque=remainder[:, 6:], base_residual=remainder[:, :6],
                contact_velocity=contact_velocity, contact_acceleration=contact_acceleration, state=state)
