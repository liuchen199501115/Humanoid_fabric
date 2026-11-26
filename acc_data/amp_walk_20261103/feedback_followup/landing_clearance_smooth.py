"""Reference-only swing lift with fixed pelvis, stance and landing poses."""
import math
import xml.etree.ElementTree as ET

import torch

from fabrics_sim.fabrics.dynamics_aware import rotation_error


@torch.no_grad()
def swing_clearance_reference(fabric, source, joint_names, events, lift, *, collision_geometry=None, fall_fraction=.15):
    """Raise complete swing arcs through leg IK, preserving their endpoints.

    A C2 bump adds ``lift`` metres at mid-swing. With collision geometry,
    it instead specifies mid-swing clearance: one constant bump amplitude
    fills the midpoint deficit, avoiding framewise corrections that amplify
    source noise. The source remains untouched and is still the audit target. Initial partial swings are retained
    to avoid changing the measured-state initialization. Joint limits and IK
    residuals are checked; geometric feasibility is not dynamic certification.
    """
    if (source.ndim != 2 or source.shape[1] != 36
            or not torch.isfinite(source).all() or not math.isfinite(lift) or lift < 0):
        raise ValueError('Swing lift requires finite G1 poses and nonnegative lift')
    if collision_geometry is not None:
        if len(collision_geometry) != 2:
            raise ValueError('Expected native collision geometry for both feet')
        for points, radii in collision_geometry:
            points, radii = source.new_tensor(points), source.new_tensor(radii)
            if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                    or radii.shape != (len(points),) or not torch.isfinite(points).all()
                    or not torch.isfinite(radii).all() or (radii < 0).any()):
                raise ValueError('Invalid native foot collision geometry')
    if not math.isfinite(fall_fraction) or not 0 < fall_fraction <= 1:
        raise ValueError("Landing fall fraction must be in (0,1]")
    result = source.detach().clone()
    if lift == 0:
        return result
    model = fabric.dynamics_tasks.model
    limits = {node.attrib['name']: node.find('limit')
              for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    for event in events:
        if event.get('starts_mid_swing', False):
            continue
        start, end = event['takeoff_frame'], event['end_frame']
        if not 0 <= start < end < len(source) or event['support'] not in (0, 1):
            raise ValueError('Invalid swing event')
        side = ('left', 'right')[1-event['support']]
        foot = model.kinematics.get_link_index(side+'_ankle_roll_link')
        joints = [joint_names.index(side+'_'+name+'_joint') for name in
                  ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        columns = [6+i for i in joints]
        positions = [7+i for i in joints]
        lower = source.new_tensor([float(limits[joint_names[i]].attrib['lower']) for i in joints])
        upper = source.new_tensor([float(limits[joint_names[i]].attrib['upper']) for i in joints])
        identity = torch.eye(6, device=source.device, dtype=source.dtype)[None]
        amplitude = lift
        if collision_geometry is not None:
            points, radii = collision_geometry[1-event['support']]
            points, radii = source.new_tensor(points), source.new_tensor(radii)
            midpoint = (start+end)//2
            state = model.snapshot(source[midpoint:midpoint+1])
            clearance = (state['transforms'][:,foot,2,None]
                +torch.einsum('bj,kj->bk',state['rotations'][:,foot,2],points)-radii).min()
            amplitude = max(0., lift-max(0.,float(clearance)))
        for frame in range(start+1, end):
            phase = (frame-start)/(end-start)
            rise=min(1.,phase/.35); fall=min(1.,(1-phase)/fall_fraction)
            height=amplitude*rise**3*(10-15*rise+6*rise*rise)*fall**3*(10-15*fall+6*fall*fall)
            q = source[frame:frame+1].clone()
            target = model.snapshot(q)
            position = target['transforms'][:,foot,:3].clone()
            rotation = target['rotations'][:,foot].clone()
            if height < 1e-7:
                continue
            position[:,2] += height
            # Near extension, Newton can choose an infeasible hyperextension
            # branch. Seed the physical flexion branch while retaining the
            # original target pose and fixed pelvis.
            q[:,positions[3]] = q[:,positions[3]].clamp_min(.1)
            for _ in range(50):
                state = model.snapshot(q)
                error = torch.cat((state['transforms'][:,foot,:3]-position,
                    rotation_error(state['rotations'][:,foot],rotation)),dim=-1)
                if error.abs().max() < 2e-5:
                    break
                jac = torch.cat((state['origin_jac'][:,foot],
                                 state['angular_jac'][:,foot]),dim=1)[:,:,columns]
                delta = -(jac.transpose(1,2)@torch.linalg.solve(
                    jac@jac.transpose(1,2)+1e-7*identity,error[...,None])).squeeze(-1)
                delta *= torch.clamp(.1/delta.abs().amax(-1,keepdim=True).clamp_min(1e-8),max=1.)
                q[:,positions] = (q[:,positions]+delta).maximum(lower).minimum(upper)
            else:
                raise RuntimeError(f'Swing lift IK failed at frame {frame}: residual {float(error.abs().max()):.6g}')
            result[frame] = q[0]
    return result


@torch.no_grad()
def feasible_swing_clearance_reference(fabric, source, joint_names, events, lift, *, collision_geometry, fall_fraction=.15):
    """Cap each whole arc's lift when bounded pose IK does not converge.

    The entire arc keeps one C2 bump amplitude. Failed IK candidates never
    enter the reference; neither the joint limits nor the IK tolerances change.
    Return an explicit report of reduced targets, which are not clearance
    guarantees and still need measured flight evidence in simulation.
    """
    result = swing_clearance_reference(fabric, source, joint_names, [], lift,
                                      collision_geometry=collision_geometry, fall_fraction=fall_fraction)
    reports = []
    for event in events:
        if event.get('starts_mid_swing', False):
            continue
        target, lower, upper = lift, 0., lift
        accepted = source
        reason = None
        for attempt in range(11):
            try:
                candidate = swing_clearance_reference(fabric, source, joint_names, [event],
                    target, collision_geometry=collision_geometry, fall_fraction=fall_fraction)
            except RuntimeError as exc:
                if not str(exc).startswith('Swing lift IK failed'):
                    raise
                reason = str(exc)
                upper = target
            else:
                accepted = candidate
                lower = target
                if attempt == 0:
                    break
            target = .5*(lower+upper)
        start, end = event['takeoff_frame'], event['end_frame']
        result[start+1:end] = accepted[start+1:end]
        reports.append(dict(takeoff_frame=start, end_frame=end,
            requested_midpoint_clearance_m=lift, feasible_target_m=lower,
            target_reduced=lower < lift, limiting_ik_failure=reason))
    return result, reports
