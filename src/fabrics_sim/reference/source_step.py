"""Source-guided step candidate with explicit optional bridge adaptations.

The active full_body_source mode uses full-robot source FK.
Whole-body IK and physical checks must pass. Even then this
is not a torque-feasibility certificate: touchdown and the following step
must be planned before sending the candidate to a physical controller.
A shortened landing is recorded explicitly. A downward contact-search target
is virtual penetration, not a collision-free trajectory or proof of contact.
"""
import math

import numpy as np
import torch

from .contact_transfer import pose_path, smooth_progress
from .initial_stance import contact_pose_reference
from .interpolation import MotionInterpolator
from .whole_body import WholeBodyReference


def smooth_body_samples(trajectory, passes):
    """Smooth root/upper samples with fixed endpoints; this adapts the source."""
    if type(passes) is not int or not 0 <= passes <= 64:
        raise ValueError('Body smoothing passes must be an integer in [0,64]')
    result = trajectory.clone()
    if not passes:
        return result
    columns = list(range(3, 7))+list(range(19, 36))
    for index in range(1, len(result)):
        if (result[index-1, 3:7]*result[index, 3:7]).sum() < 0:
            result[index, 3:7] *= -1
    for _ in range(passes):
        filtered = result.clone()
        filtered[1:-1, columns] = (.25*result[:-2, columns]+.5*result[1:-1, columns]
                                   +.25*result[2:, columns])
        filtered[:, 3:7] /= filtered[:, 3:7].norm(dim=-1, keepdim=True)
        result = filtered
    return result


def blend_source_body(trajectory, initial):
    """Spread the corrected torso/arm boundary over the source interval."""
    result = trajectory.clone()
    u = torch.linspace(0., 1., len(result), device=result.device, dtype=result.dtype)
    weight = (u**3*(10-15*u+6*u*u))[:,None]
    result[:,19:] = initial[:,19:]+weight*(result[:,19:]-initial[:,19:])
    quat = result[:,3:7]
    quat = quat*torch.where((quat*initial[:,3:7]).sum(-1,keepdim=True)<0,-1.,1.)
    quat = initial[:,3:7]+weight*(quat-initial[:,3:7])
    result[:,3:7] = quat/quat.norm(dim=-1,keepdim=True)
    result[0] = initial[0]
    return result


def centroidal_support_check(model, reference, state, support_link, friction=.6):
    """Necessary single flat-sole wrench checks; not a joint-torque certificate."""
    mass = model.com_task.link_masses.sum()
    force = mass*((state['com_jac']@reference.a[..., None]).squeeze(-1)
                  +state['com_jac_bias']+reference.q.new_tensor([0., 0., 9.81]))
    momentum_rate = ((state['momentum_matrix']@reference.a[..., None]).squeeze(-1)
                     +state['momentum_matrix_bias'])
    ankle = state['transforms'][:, support_link, :3]
    moment = momentum_rate+torch.linalg.cross(state['com']-ankle, force)
    rotation = state['rotations'][:, support_link].transpose(-1, -2)
    force = (rotation@force[..., None]).squeeze(-1)
    moment = (rotation@moment[..., None]).squeeze(-1)
    moment -= torch.linalg.cross(reference.q.new_tensor([[0., 0., -.035]]), force)
    cop = torch.stack((-moment[:, 1], moment[:, 0]), -1)/force[:, 2:3].clamp_min(1e-8)
    # Conservative inner rectangle; exact wrench-cone, torque and collision
    # checks must still be performed before this candidate controls a robot.
    passed = ((force[:, 2] > 0.) & (force[:, :2].abs() <= friction*force[:, 2:3]).all(-1)
              & (cop[:, 0] >= -.04) & (cop[:, 0] <= .11) & (cop[:, 1].abs() <= .02))
    return force, cop, passed


def adapt_pelvis_orientation(trajectory, pitch=0., roll=0.):
    """Adapt the body seed only, with local pitch then roll; never mutate PKL."""
    if not all(math.isfinite(x) for x in (pitch, roll)):
        raise ValueError('Pelvis offsets must be finite')
    result = trajectory.clone()
    u = torch.linspace(0., 1., len(result), device=result.device, dtype=result.dtype)
    for axis, offset in ((1, pitch), (0, roll)):
        if not offset:
            continue
        angle = offset*u.square()*(3-2*u)
        delta = torch.zeros_like(result[:,3:7])
        delta[:,axis], delta[:,3] = torch.sin(angle/2), torch.cos(angle/2)
        quat = result[:,3:7].clone()
        result[:,3:6] = (quat[:,3,None]*delta[:,:3]+delta[:,3,None]*quat[:,:3]
                        +torch.linalg.cross(quat[:,:3],delta[:,:3]))
        result[:,6] = quat[:,3]*delta[:,3]-(quat[:,:3]*delta[:,:3]).sum(-1)
    return result


class SourceStepCandidate:
    @torch.no_grad()
    def __init__(self, fabric, initial, joint_names, source, start_frame, end_frame,
                 support, landing_pose, *, duration=.8, landing_height=.62, clearance=.04, source_clearance_bump=0., arm_clearance_offset=0.,
                 body_smoothing_passes=0, swing_phase_blend=0.,
                 landing_com_fraction=.5, terminal_capture="landing", pelvis_pitch_offset=0., pelvis_roll_offset=0., pelvis_roll_bump=0., body_interpolation="pchip", body_join="first_sample", landing_fraction=1.,
                 landing_search_depth=0., landing_search_duration=.2, com_preview="full_body_source", landing_com_lateral_offset=0., capture_margin=0.):
        if (support not in (0, 1) or not 0 <= start_frame < end_frame < len(source)
                or int(start_frame) != start_frame or int(end_frame) != end_frame
                or landing_pose.shape != (1, 7) or source.shape[1:] != (36,)
                or not torch.isfinite(landing_pose).all()
                or not math.isfinite(clearance) or clearance < 0
                or not torch.isfinite(initial.v).all()):
            raise ValueError('Source-step candidate requires a finite G1 reference and a valid source interval')
        if (not math.isfinite(landing_com_fraction) or not 0 <= landing_com_fraction <= 1
                or not math.isfinite(pelvis_pitch_offset) or not math.isfinite(pelvis_roll_offset) or not math.isfinite(landing_com_lateral_offset)):
            raise ValueError('Expected landing COM fraction in [0,1] and finite pelvis pitch offset')
        if not math.isfinite(landing_fraction) or not 0 < landing_fraction <= 1:
            raise ValueError("Landing fraction must be finite and in (0,1]")
        if (not math.isfinite(landing_search_depth) or not 0 <= landing_search_depth <= .01
                or not math.isfinite(landing_search_duration) or landing_search_duration <= 0
                or (landing_search_depth > 0 and landing_search_duration > duration)):
            raise ValueError("Landing search requires depth in [0,.01] and duration within the step")
        if com_preview != 'full_body_source' or capture_margin != 0.:
            raise ValueError('Only full-body source references without capture margins are supported')
        self.landing_search_depth = float(landing_search_depth)
        self.landing_search_duration = float(landing_search_duration)
        self.fabric, self.names, self.support = fabric, joint_names, support
        self.model = fabric.dynamics_tasks.model
        self.feet = [self.model.kinematics.get_link_index(side+'_ankle_roll_link')
                     for side in ('left', 'right')]
        legs = [joint_names.index(side+'_'+name+'_joint') for side in ('left', 'right')
                for name in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        self.columns = [0, 1, 2]+[6+j for j in legs]
        self.positions = [0, 1, 2]+[7+j for j in legs]
        self.q = initial.q.clone()
        self.initial = initial
        state = self.model.snapshot(initial.q)
        self.start_feet = state['transforms'][:, self.feet].clone()
        self.source_landing_pose = landing_pose.clone()
        landing_pose = landing_pose.clone()
        if landing_fraction != 1.:
            landing_pose[:, :2] = (self.start_feet[:, 1-support, :2]
                                  +landing_fraction*(landing_pose[:, :2]-self.start_feet[:, 1-support, :2]))
        self.landing_pose = landing_pose.clone()
        if com_preview == 'full_body_source':
            if landing_fraction != 1. or landing_search_depth:
                raise ValueError('Full-body source feet forbid stride scaling or landing search')
            from .full_body_com_path import FullBodySourceFeet
            self.source_feet = FullBodySourceFeet(fabric, initial, source[start_frame:end_frame+1],
                                                  duration, support, self.feet, clearance_bump=source_clearance_bump)
            endpoint = self.q.new_tensor(self.source_feet.sample(duration)[0][1-support])[None]
            if not torch.allclose(endpoint,landing_pose,atol=2e-6,rtol=0.):
                raise ValueError('Full-body source foot endpoint must match the original source pose')
        self.landing_fraction = float(landing_fraction)
        self.end_feet = self.start_feet.clone()
        self.end_feet[:, 1-support] = landing_pose
        from .full_body_com_path import FullBodyComPath
        self.preview = FullBodyComPath(fabric, initial, source[start_frame:end_frame+1], duration)
        if not math.isfinite(swing_phase_blend) or not 0. <= swing_phase_blend <= 1.:
            raise ValueError("Swing phase blend must be finite and in [0,1]")
        self.swing_phase_blend = swing_phase_blend
        self.com_preview = com_preview
        self.phase_preview = self.preview
        self.duration, self.clearance = duration, clearance
        from .pelvis_reference import WindowedPelvisRoll
        self.pelvis_roll_path = WindowedPelvisRoll(duration,pelvis_roll_bump)
        self.start_frame, self.end_frame = int(start_frame), int(end_frame)
        # Retain source upper-body motion, joining it continuously to the
        # corrected support-transfer pose. Root XYZ and legs are solved below.
        trajectory = adapt_pelvis_orientation(source[start_frame:end_frame+1],
                                               pelvis_pitch_offset, pelvis_roll_offset)
        if not math.isfinite(arm_clearance_offset):
            raise ValueError('Arm clearance offset must be finite')
        # Optional reference posture adaptation, never a state reset. The
        # existing C2 body bridge preserves the live entry state and rates.
        for side, sign in (('left',1.),('right',-1.)):
            j = 7+joint_names.index(side+'_shoulder_roll_joint')
            trajectory[:,j] += sign*arm_clearance_offset
        if body_join not in ('first_sample','distributed','endpoint'):
            raise ValueError('Unknown source body boundary join')
        if body_join == 'distributed':
            trajectory = blend_source_body(trajectory,initial.q)
        if body_join == 'endpoint':
            trajectory = torch.cat((initial.q,trajectory[-1:]),0)
        trajectory[0] = initial.q[0]
        self.body_span = len(trajectory)-1
        trajectory = smooth_body_samples(trajectory, body_smoothing_passes)
        self.source = MotionInterpolator(trajectory, method=body_interpolation)
        if com_preview == 'full_body_source':
            from .bridge import WholeBodyReferenceBridge
            start_body = WholeBodyReference(*self.source.kinematics(0.,self.body_span/duration,0.))
            self.body_bridge = WholeBodyReferenceBridge(initial,start_body,duration)


    def progress(self, time):
        if not math.isfinite(time) or not 0. <= time <= self.duration:
            raise ValueError('Candidate ends at touchdown; no post-landing continuation is certified')
        u = time/self.duration
        return u, 1./self.duration, 0., smooth_progress(time, self.duration)

    def foot_kinematics(self, time):
        """Consistent foot targets, including a bounded virtual contact search."""
        if getattr(self,'com_preview',None) == 'full_body_source':
            feet,v,a = (self.q.new_tensor(value)[None] for value in self.source_feet.sample(time))
            # Body uses the same source clock as feet and COM.
            return feet,v,a,(time/self.duration,1./self.duration,0.)
        u, du, ddu, progress = self.progress(time)
        feet, fv, fa = pose_path(self.start_feet, self.end_feet, progress)
        bump = 64*self.clearance*u**3*(1-u)**3
        bump_d = 192*self.clearance*u**2*(1-u)**2*(1-2*u)
        bump_dd = 384*self.clearance*u*(1-u)*(1-5*u+5*u*u)
        swing = 1-self.support
        feet[:, swing, 2] += bump
        fv[:, swing, 2] += bump_d*du
        fa[:, swing, 2] += bump_dd*du**2+bump_d*ddu
        if self.landing_search_depth:
            # A bounded virtual target below the sole plane requests contact.
            # Runtime must confirm load; this segment is not collision-free motion.
            from .landing import contact_search
            offset, speed, acceleration = contact_search(
                max(0., time-self.duration+self.landing_search_duration),
                self.landing_search_depth, self.landing_search_duration)
            feet[:, swing, 2] += offset
            fv[:, swing, 2] += speed
            fa[:, swing, 2] += acceleration
        return feet, fv, fa, progress

    @torch.no_grad()
    def sample(self, time):
        feet, fv, fa, progress = self.foot_kinematics(time)
        c, cv, ca = (self.q.new_tensor(value)[None] for value in self.preview.sample(time))
        span = getattr(self,"body_span",self.end_frame-self.start_frame)
        source, v, a = self.source.kinematics(*(span*x for x in progress))
        if getattr(self,'com_preview',None) == 'full_body_source':
            joined = self.body_bridge.sample(time,WholeBodyReference(source,v,a))
            source,v,a = joined.q.clone(),joined.v.clone(),joined.a.clone()

        if getattr(self,'pelvis_roll_path',None) is not None:
            adjusted = self.pelvis_roll_path.sample(WholeBodyReference(source,v,a),time)
            source,v,a = adjusted.q,adjusted.v,adjusted.a
        seed = source.clone()
        seed[:, self.positions] = self.q[:, self.positions]
        self.q = contact_pose_reference(self.fabric, seed, self.names, feet,
                                       c[:, :2], float(c[0, 2]), tolerance=2e-6)
        state = self.model.snapshot(self.q)
        jac = torch.cat((torch.cat((state['origin_jac'][:, self.feet],
                                   state['angular_jac'][:, self.feet]), 2).flatten(1, 2),
                         state['com_jac']), 1)
        v[:, self.columns] = 0.
        target_v = torch.cat((fv.flatten(1), cv), 1)
        v[:, self.columns] = torch.linalg.solve(jac[:, :, self.columns],
            (target_v-(jac@v[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        state, _, _ = self.model.evaluate(self.q, v, self.fabric.integrate_root_position,
                                          self.fabric.dynamics_tasks.config['difference_step'])
        bias = torch.cat((torch.cat((state['origin_jac_bias'][:, self.feet],
                                    state['angular_jac_bias'][:, self.feet]), 2).flatten(1),
                          state['com_jac_bias']), 1)
        a[:, self.columns] = 0.
        target_a = torch.cat((fa.flatten(1), ca), 1)
        a[:, self.columns] = torch.linalg.solve(jac[:, :, self.columns],
            (target_a-bias-(jac@a[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        return WholeBodyReference(self.q, v, a)
