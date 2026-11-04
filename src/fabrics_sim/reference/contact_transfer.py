"""Reference-only, joint-limited whole-body transfer and first-lift diagnostic.

Measured load/balance gates remain owned by ReferenceManager. This planner
does not reset the robot or certify a complete walking trajectory.
"""
import math
import xml.etree.ElementTree as ET

import torch

from fabrics_sim.reference.initial_stance import contact_pose_reference
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix


class TransferStartGate:
    """Require sustained measured rest before a stopped transfer reference.

    Thresholds are transfer-start policy, not actuator or acceleration limits.
    """
    def __init__(self):
        self.elapsed = 0.

    def update(self, reference_stopped, com_velocity, joint_velocity, normal_forces, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Transfer gate timestep must be finite and positive')
        values = [torch.as_tensor(x) for x in (com_velocity, joint_velocity, normal_forces)]
        if [v.numel() for v in values] != [3, 29, 2]:
            raise ValueError('Transfer gate expects COM velocity, 29 joint velocities and two loads')
        cv, jv, loads = values
        stable = bool(reference_stopped and all(torch.isfinite(v).all() for v in values)
                      and cv.norm() <= .04 and jv.abs().max() <= .2 and (loads >= 20.).all())
        self.elapsed = self.elapsed+dt if stable else 0.
        return self.elapsed >= .2-1e-9


def smooth_progress(elapsed, duration):
    u = min(1., max(0., elapsed / duration))
    return (u**3*(10-15*u+6*u*u),
            30*u*u*(1-u)**2/duration,
            60*u*(1-u)*(1-2*u)/duration**2)


def pose_path(start, end, progress):
    """Quintic translation and normalized quaternion interpolation, world rates."""
    s, ds, dds = progress
    end = end.clone()
    end[..., 3:] *= torch.where((start[..., 3:]*end[..., 3:]).sum(-1, keepdim=True)<0, -1., 1.)
    delta = end-start
    pose = start+s*delta
    du, ddu = ds*delta[..., 3:], dds*delta[..., 3:]
    norm = pose[..., 3:].norm(dim=-1, keepdim=True)
    quat = pose[..., 3:]/norm
    nv = (quat*du).sum(-1, keepdim=True)
    qv = (du-quat*nv)/norm
    na = (du.square().sum(-1, keepdim=True)+(pose[..., 3:]*ddu).sum(-1, keepdim=True)-nv.square())/norm
    qa = (ddu-quat*na-2*qv*nv)/norm
    def angular(dq):
        return 2*(quat[..., 3:]*dq[..., :3]-dq[..., 3:]*quat[..., :3]
                  +torch.linalg.cross(quat[..., :3], dq[..., :3]))
    pose[..., 3:] = quat
    return pose, torch.cat((ds*delta[..., :3], angular(qv)), -1), torch.cat((dds*delta[..., :3], angular(qa)), -1)


class ContactTransferReference:
    """Retain torso/arm joints; optionally roll pelvis while solving feet and COM."""
    @torch.no_grad()
    def __init__(self, fabric, reference, joint_names, stance_poses, support,
                 start_time, transfer_duration=2., lift_duration=1., lift_height=.04,
                 support_lateral_offset=None, preflight_dt=.02, joint_bounds=None, com_height_offset=-.02, pelvis_roll_offset=0.):
        if (support not in (0, 1) or stance_poses.shape != (1, 2, 7)
                or not torch.isfinite(stance_poses).all()
                or stance_poses.dtype != reference.q.dtype or stance_poses.device != reference.q.device
                or not torch.allclose(stance_poses[..., 3:].norm(dim=-1),
                                      stance_poses.new_ones(1, 2), atol=1e-5, rtol=1e-5)
                or not all(math.isfinite(x) for x in (start_time, transfer_duration, lift_duration, lift_height))
                or (support_lateral_offset is not None and
                    (not math.isfinite(support_lateral_offset) or abs(support_lateral_offset) > .02))
                or not math.isfinite(preflight_dt) or preflight_dt <= 0
                or min(transfer_duration, lift_duration, lift_height) <= 0
                or reference.v.abs().max()>1e-4 or reference.a.abs().max()>1e-3):
            raise ValueError('Transfer requires a stopped reference, two stance poses and positive durations/height')
        if not math.isfinite(pelvis_roll_offset):
            raise ValueError("Pelvis roll offset must be finite")
        if not math.isfinite(com_height_offset):
            raise ValueError("COM height offset must be finite")
        self.fabric, self.names, self.support = fabric, joint_names, support
        self.start_time, self.duration = start_time, transfer_duration
        self.lift_duration, self.lift_height = lift_duration, lift_height
        self.release_time = None
        self.model = fabric.dynamics_tasks.model
        self.feet = [self.model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left', 'right')]
        legs = [joint_names.index(s+'_'+j+'_joint') for s in ('left', 'right')
                for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        self.columns = [0, 1, 2]+[6+i for i in legs]
        self.q = reference.q.clone()
        self.pelvis_roll_offset = float(pelvis_roll_offset)
        self.start_pelvis = reference.q[:, :7].clone()
        self.end_pelvis = self.start_pelvis.clone()
        if pelvis_roll_offset:
            quat = self.start_pelvis[:, 3:7]
            sine, cosine = math.sin(pelvis_roll_offset/2), math.cos(pelvis_roll_offset/2)
            # Right multiplication: roll about the initial pelvis local X axis.
            self.end_pelvis[:, 3:7] = torch.stack((quat[:,0]*cosine+quat[:,3]*sine,
                quat[:,1]*cosine+quat[:,2]*sine, quat[:,2]*cosine-quat[:,1]*sine,
                quat[:,3]*cosine-quat[:,0]*sine), -1)
        state = self.model.snapshot(self.q)
        self.start_feet = state['transforms'][:, self.feet].clone()
        self.start_com = state['com'].clone()
        self.stance = stance_poses.detach().clone()
        self.end_com = self.start_com.clone()
        # Leave leg-extension margin while correcting the rolled source foot.
        self.end_com[:, 2] += com_height_offset
        rotation = _quat_xyzw_to_matrix(self.stance[:, support, 3:])
        self.end_com[:, :2] = self.stance[:, support, :2]+.02*rotation[:, :2, 0]
        self.phase = 'TRANSFER'
        # Search inside the same support sole, keeping physical limits and
        # runtime braking constraints. A feasible endpoint alone is insufficient.
        limits = {n.attrib['name']: n.find('limit') for n in ET.parse(fabric.urdf_path).getroot().findall('joint')}
        lower = self.q.new_tensor([float(limits[n].attrib['lower']) for n in joint_names])
        upper = self.q.new_tensor([float(limits[n].attrib['upper']) for n in joint_names])
        bounds = dict(lower=lower, upper=upper, dt=.002, bandwidth=8.) if joint_bounds is None else joint_bounds
        if (not torch.isfinite(bounds['lower']).all() or not torch.isfinite(bounds['upper']).all()
                or bounds['lower'].numel() != len(joint_names) or bounds['upper'].numel() != len(joint_names)
                or not (bounds['lower'] < bounds['upper']).all()
                or not math.isfinite(bounds['dt']) or bounds['dt'] <= 0):
            raise ValueError('Transfer joint bounds require finite ordered G1 limits and a positive timestep')
        self._joint_bounds = bounds
        self._initial_q = reference.q.clone()
        center = self.end_com.clone()
        candidates = ([float(support_lateral_offset)] if support_lateral_offset is not None else
                      [0., -.004, .004, -.008, .008, -.012, .012, -.016, .016, -.019, .019])
        if joint_bounds is None:
            # Preserve the kinematic-only API when no controller bounds are
            # supplied. Runtime callers pass their actual bounds explicitly.
            offset = 0. if support_lateral_offset is None else float(support_lateral_offset)
            self.end_com[:, :2] += offset*rotation[:, :2, 1]
            for duration, start, mode in ((self.duration, self.start_time, 3),
                    (self.lift_duration, self.start_time+self.duration, self.support+1)):
                for index in range(21):
                    self.sample(start+duration*index/20, mode)
            self.support_lateral_offset = offset
            self.preflight = dict(sampled_joint_bounds_passed=False, support_lateral_offset=offset)
            self.q = self._initial_q.clone()
            self.release_time = None
            self.phase = 'TRANSFER'
            return
        rejected = []
        for offset in candidates:
            self.end_com = center.clone()
            self.end_com[:, :2] += offset*rotation[:, :2, 1]
            try:
                # Coarse rejection saves IK work; the accepted path receives
                # the caller's requested sample interval (2 ms in the runner).
                report = self._check_path(max(preflight_dt, .02), legs)
                if preflight_dt < .02:
                    report = self._check_path(preflight_dt, legs)
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(offset=offset, reason=str(exc)))
                continue
            self.support_lateral_offset = offset
            self.preflight = dict(**report, support_lateral_offset=offset, rejected=rejected,
                                  sampled_joint_bounds_passed=True, pelvis_roll_offset=self.pelvis_roll_offset)
            break
        else:
            raise RuntimeError('No transfer inside the support sole satisfies joint braking/limit margins: '
                               + str(rejected))
        self.q = self._initial_q.clone()
        self.release_time = None
        self.phase = 'TRANSFER'

    @torch.no_grad()
    def _check_path(self, dt, legs):
        from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
        self.q = self._initial_q.clone()
        self.release_time = None
        self.phase = 'TRANSFER'
        bounds = self._joint_bounds
        worst = 0.
        count = 0
        for duration, start, mode in ((self.duration, self.start_time, 3),
                (self.lift_duration, self.start_time+self.duration, self.support+1)):
            steps = math.ceil(duration/dt)
            for index in range(steps+1):
                time = start+duration*index/steps
                reference = self.sample(time, mode)
                lo, hi = measured_joint_acceleration_bounds(reference.q[:, 7:], reference.v[:, 6:],
                    bounds['lower'], bounds['upper'], bounds['dt'], barrier_bandwidth=bounds['bandwidth'],
                    velocity_damper_time=bounds.get('velocity_damper_time'),
                    impact_recovery=bounds.get('impact_recovery', False))
                violation = torch.maximum(lo-reference.a[:, 6:], reference.a[:, 6:]-hi).clamp_min(0)
                error = float(violation.max())
                if error > 1e-3:
                    joint = self.names[int(violation[0].argmax())]
                    raise RuntimeError(f'Transfer joint bound at t={time-self.start_time:.4f}: {joint}, violation={error:.6g}')
                worst = max(worst, error)
                count += 1
        lower = bounds['lower'].reshape(-1)
        upper = bounds['upper'].reshape(-1)
        margin = torch.minimum(reference.q[0, 7:]-lower, upper-reference.q[0, 7:])[legs]
        # Match the runtime barrier's 0.05-rad reserve, with numerical room.
        reserve = torch.minimum(torch.full_like(lower, .05), (upper-lower)*.25)[legs]+.005
        if (margin < reserve).any():
            raise RuntimeError('Transfer endpoint leaves insufficient joint margin: '+str(float(margin.min())))
        return dict(samples=count, sample_dt=dt, max_joint_bound_violation=worst,
                    minimum_terminal_leg_margin=float(margin.min()))

    def ready_for_release(self, time, com, com_velocity, q):
        """Measured load alone does not imply the transfer reference was reached."""
        if (not math.isfinite(time) or not torch.isfinite(com).all()
                or not torch.isfinite(com_velocity).all() or not torch.isfinite(q).all()
                or com.shape != self.end_com.shape or com_velocity.shape != com.shape
                or q.shape != self.q.shape):
            raise ValueError('Transfer release requires finite measured COM, velocity and configuration')
        if time < self.start_time+self.duration:
            return False
        legs = [column+1 for column in self.columns[3:]]
        return bool((com-self.end_com).norm(dim=-1).max() <= .015
                    and com_velocity.norm(dim=-1).max() <= .04
                    and (q[:, legs]-self.q[:, legs]).abs().max() <= .08)

    def _jacobian(self, state):
        feet = torch.cat((state['origin_jac'][:, self.feet], state['angular_jac'][:, self.feet]), 2).flatten(1, 2)
        return torch.cat((feet, state['com_jac']), 1)

    @torch.no_grad()
    def sample(self, time, contact_mode):
        if not math.isfinite(time) or time < self.start_time:
            raise ValueError('Transfer sample time must be finite and after its start')
        if (contact_mode not in (3, self.support+1)
                or (self.release_time is not None and contact_mode != self.support+1)):
            raise RuntimeError('Unexpected support change during first-transfer diagnostic')
        if contact_mode != 3 and self.release_time is None:
            if time < self.start_time+self.duration:
                raise RuntimeError('Contact released before full-body transfer finished')
            self.release_time = time
        if self.release_time is None:
            progress = smooth_progress(time-self.start_time, self.duration)
            feet, fv, fa = pose_path(self.start_feet, self.stance, progress)
            delta = self.end_com-self.start_com
            com = self.start_com+progress[0]*delta
            cv, ca = progress[1]*delta, progress[2]*delta
            self.phase = 'TRANSFER' if time < self.start_time+self.duration else 'WAIT_FOR_SUPPORT'
        else:
            endpoint = self.stance.clone()
            endpoint[:, 1-self.support, 2] += self.lift_height
            feet, fv, fa = pose_path(self.stance, endpoint, smooth_progress(time-self.release_time, self.lift_duration))
            com = self.end_com
            cv = ca = torch.zeros_like(com)
            self.phase = 'LIFT' if time < self.release_time+self.lift_duration else 'HOLD'
        body_v = body_a = None
        if self.pelvis_roll_offset:
            body, body_v, body_a = pose_path(self.start_pelvis, self.end_pelvis,
                smooth_progress(time-self.start_time, self.duration))
            self.q = self.q.clone()
            self.q[:, 3:7] = body[:, 3:7]
        self.q = contact_pose_reference(self.fabric, self.q, self.names, feet, com[:, :2], float(com[0, 2]), tolerance=2e-6)
        state = self.model.snapshot(self.q)
        full_jac = self._jacobian(state)
        jac = full_jac[:, :, self.columns]
        v = self.q.new_zeros(1, 35)
        target_v = torch.cat((fv.flatten(1), cv), 1)
        if body_v is not None:
            v[:, 3:6] = body_v[:, 3:]
            target_v = target_v-(full_jac@v[..., None]).squeeze(-1)
        v[:, self.columns] = torch.linalg.solve(jac, target_v[..., None]).squeeze(-1)
        state, _, _ = self.model.evaluate(self.q, v, self.fabric.integrate_root_position,
                                          self.fabric.dynamics_tasks.config['difference_step'])
        bias = torch.cat((torch.cat((state['origin_jac_bias'][:, self.feet],
                                    state['angular_jac_bias'][:, self.feet]), 2).flatten(1), state['com_jac_bias']), 1)
        a = torch.zeros_like(v)
        target_a = torch.cat((fa.flatten(1), ca), 1)-bias
        if body_a is not None:
            a[:, 3:6] = body_a[:, 3:]
            target_a = target_a-(full_jac@a[..., None]).squeeze(-1)
        a[:, self.columns] = torch.linalg.solve(jac, target_a[..., None]).squeeze(-1)
        return WholeBodyReference(self.q, v, a)


class MeasuredContactTransferReference(ContactTransferReference):
    """Enter the geometric transfer from a measured full-body reference state.

    The physical contact anchors remain owned by WBC. Small transient reference
    foot offsets caused by matching measured rates are bounded explicitly; this
    is not a certificate of exact contact-consistent reference motion.
    """
    def __init__(self, fabric, measured, joint_names, stance_poses, support, start_time,
                 *, maximum_entry_foot_offset=.002, maximum_entry_foot_rotation=.01, **options):
        from .bridge import WholeBodyReferenceBridge
        if (not math.isfinite(maximum_entry_foot_offset) or maximum_entry_foot_offset<=0
                or not math.isfinite(maximum_entry_foot_rotation) or maximum_entry_foot_rotation<=0):
            raise ValueError('Measured entry requires positive finite reference foot tolerances')
        stationary=WholeBodyReference(measured.q,torch.zeros_like(measured.v),torch.zeros_like(measured.a))
        self.entry_bridge=WholeBodyReferenceBridge(measured,stationary,options.get('transfer_duration',2.))
        self.maximum_entry_foot_offset=maximum_entry_foot_offset
        self.maximum_entry_foot_rotation=maximum_entry_foot_rotation
        self.entry_max_position_error=0.
        self.entry_max_rotation_error=0.
        super().__init__(fabric,stationary,joint_names,stance_poses,support,start_time,**options)
        self.preflight.update(measured_entry=True,
            initial_velocity_max=float(measured.v.abs().max()),
            initial_acceleration_max=float(measured.a.abs().max()),
            entry_max_reference_foot_offset_m=self.entry_max_position_error,
            entry_max_reference_foot_rotation_rad=self.entry_max_rotation_error,
            exact_contact_reference=False,full_dynamics_validated=False)

    @torch.no_grad()
    def sample(self,time,contact_mode):
        baseline=super().sample(time,contact_mode)
        elapsed=time-self.start_time
        joined=self.entry_bridge.sample(elapsed,baseline)
        if elapsed<self.duration:
            from fabrics_sim.fabrics.dynamics_aware import rotation_error
            original=self.model.snapshot(baseline.q)
            actual=self.model.snapshot(joined.q)
            distance=float((actual['transforms'][:,self.feet,:3]-original['transforms'][:,self.feet,:3]).norm(dim=-1).max())
            angle=float(rotation_error(actual['rotations'][:,self.feet],original['rotations'][:,self.feet]).norm(dim=-1).max())
            self.entry_max_position_error=max(self.entry_max_position_error,distance)
            self.entry_max_rotation_error=max(self.entry_max_rotation_error,angle)
            if distance>self.maximum_entry_foot_offset or angle>self.maximum_entry_foot_rotation:
                raise RuntimeError(f'Measured entry exceeds reference contact tolerance: position={distance}, rotation={angle}')
        return joined
