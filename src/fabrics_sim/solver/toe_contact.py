"""Five-coordinate toe-line contact in a fixed, horizontal frame.

Forces/moments are expressed at the toe midpoint in this frame. Pitch moment
is zero and pitch acceleration is free. This is an experimental rigid line
model, not a model of impact or compliant contact.
"""
import math
import numpy as np
import torch

from fabrics_sim.fabrics.dynamics_aware import rotation_error
from fabrics_sim.fabrics.task_request import TaskRequest
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _skew


def has_loaded_toe_line(local_points, normal_forces, separations):
    """Require two loaded toe points spanning the conservative 5 cm line."""
    points,forces,distance=(np.asarray(x) for x in (local_points,normal_forces,separations))
    forces,distance=forces.reshape(-1),distance.reshape(-1)
    if (points.shape != (len(forces),3) or distance.shape != forces.shape
            or not all(np.isfinite(x).all() for x in (points,forces,distance))):
        raise ValueError('Expected finite contact points [N,3], forces and separations [N]')
    loaded=(points[:,0]>.09)&(forces>.5)&(distance<=.001)
    # The physical toe corners are 6 cm apart. Retain at least the 5 cm
    # support width used in the cone, allowing only millimetre-scale roundoff.
    return bool(loaded.sum()>=2 and np.ptp(points[loaded,1])>=.05-1e-3
                and np.ptp(points[loaded,0])<=.01)


def has_touching_toe_line(local_points, separations):
    """Observe a toe line that can pivot after the OTHER foot takes the load.

    Zero normal force does not mean a flat, unloaded sole has left the floor.
    Requiring both toe corners to remain loaded prevents heel release precisely
    when load transfer succeeds. This checks geometry only; the caller must
    separately confirm incoming-foot load. It is not a support-load certificate.
    """
    points = np.asarray(local_points)
    distance = np.asarray(separations).reshape(-1)
    if (points.shape != (len(distance), 3)
            or not np.isfinite(points).all() or not np.isfinite(distance).all()):
        raise ValueError('Expected finite contact points [N,3] and separations [N]')
    touching = ((points[:, 0] > .09) & (points[:, 0] < .14)
                & (np.abs(points[:, 2] + .035) <= .003)
                & (np.abs(distance) <= .001))
    return bool(touching.sum() >= 2 and np.ptp(points[touching, 1]) >= .05-1e-3
                and np.ptp(points[touching, 0]) <= .01)


def loaded_transverse_line(local_points, normal_forces, separations):
    """Midpoint and half-width of a loaded heel/toe edge spanning 5 cm.

    A heel edge supplies roll moment just as a toe edge does. Require actual
    loaded endpoints; neither a single corner nor an unloaded second point
    establishes a line. Side edges are left to the point-contact fallback.
    """
    points=np.asarray(local_points,dtype=float)
    forces,distances=(np.asarray(v).reshape(-1) for v in (normal_forces,separations))
    if (points.shape!=(len(forces),3) or distances.shape!=forces.shape
            or not all(np.isfinite(v).all() for v in (points,forces,distances))):
        raise ValueError('Invalid measured line contacts')
    points=points[(forces>.5)&(distances<=.001)]
    best=None;span=.05-1e-3
    for i in range(len(points)):
        for j in range(i):
            delta=np.abs(points[i]-points[j])
            if delta[1]>=span and delta[0]<=.01 and delta[2]<=.002:
                best=(.5*(points[i]+points[j]),min(.025,float(delta[1])/2));span=delta[1]
    return best


class ToeContact:
    @classmethod
    def from_lifted_sole(cls, position, rotation, time, maximum_height,
                         world_points, normal_forces, separations, flatten=False):
        """Observe an existing edge/point contact before commanded toe rolling.

        A coarse control tick may see a flat sole change directly to one heel
        point. Waiting for a toe-line confirmation would retain fictitious
        rotational constraints. No loaded point means no inferred support.
        """
        points = np.asarray(world_points)
        forces, distances = (np.asarray(x).reshape(-1) for x in (normal_forces, separations))
        if (points.shape != (len(forces), 3) or distances.shape != forces.shape
                or not math.isfinite(maximum_height)
                or not all(np.isfinite(x).all() for x in (points, forces, distances))):
            raise ValueError('Invalid observed sole patch')
        if maximum_height <= .002 or not ((forces > .5) & (distances <= .001)).any():
            return None
        contact = cls(position, rotation, time)
        contact.flatten = bool(flatten)
        contact.update_support(position, rotation, points, forces, distances)
        return contact

    def __init__(self, position, rotation, start_time, toe_length=.12, sole_depth=.035):
        if (position.ndim != 2 or position.shape[-1] != 3
                or rotation.shape != (*position.shape[:-1], 3, 3)
                or rotation.dtype != position.dtype or rotation.device != position.device
                or not torch.isfinite(position).all() or not torch.isfinite(rotation).all()
                or not all(math.isfinite(x) for x in (start_time,toe_length,sole_depth))
                or toe_length<=0 or sole_depth<0):
            raise ValueError('Toe contact needs finite positions [B,3] and rotations [B,3,3]')
        self.local_point = position.new_tensor([toe_length, 0., -sole_depth]).expand_as(position).clone()
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        c, s = yaw.cos(), yaw.sin()
        self.frame = torch.eye(3, dtype=position.dtype, device=position.device)[None].repeat(len(position), 1, 1)
        self.frame[:, 0, 0], self.frame[:, 0, 1] = c, -s
        self.frame[:, 1, 0], self.frame[:, 1, 1] = s, c
        self.anchor = position.clone() + (rotation @ self.local_point[..., None]).squeeze(-1)
        self.initial_rotation = rotation.clone()
        self.start_time = float(start_time)
        self.point_support = False
        self.half_width = .025
        self.flatten = False
        self.flat_seconds = 0.

    def update_support(self, position, rotation, world_points, normal_forces, separations):
        """Downgrade a disappearing toe line to its remaining loaded point.

        Retain a material point while it still touches the measured floor;
        replace it only after it lifts and a new loaded point is observed.
        This single-robot observation update never moves the physical robot.
        """
        if len(position) != 1:
            raise ValueError('Measured toe support update requires one robot')
        points = np.asarray(world_points)
        forces, distances = (np.asarray(x).reshape(-1) for x in (normal_forces, separations))
        if (points.shape != (len(forces), 3) or distances.shape != forces.shape
                or not all(np.isfinite(x).all() for x in (points, forces, distances))):
            raise ValueError('Invalid measured toe contacts')
        if not len(points):
            return False
        origin = position[0].detach().cpu().numpy()
        orientation = rotation[0].detach().cpu().numpy()
        local = (orientation.T @ (points-origin).T).T
        loaded = (forces > .5) & (distances <= .001)
        if not loaded.any():
            return False
        line = loaded_transverse_line(local,forces,distances)
        if self.point_support and line is None:
            old_point=origin+orientation@self.local_point[0].detach().cpu().numpy()
            # Keep a touching material point even if the contact manifold
            # jitters. If it has lifted above the measured floor patch, it
            # cannot supply support: move the constraint to an observed
            # loaded point instead of pinning an airborne corner. A newly
            # loaded transverse edge MUST promote this point to a line even
            # while the original corner remains on the ground: the second
            # physical contact constrains roll/yaw and supplies roll moment.
            if old_point[2] <= points[loaded,2].max()+.002:
                return False
        if line is not None:
            line,half_width=line
            # Keep an existing material line anchored despite contact-manifold
            # jitter. When the loaded edge changes (e.g. toe to heel), anchor
            # the newly observed edge, never an airborne toe midpoint.
            old=self.local_point[0].detach().cpu().numpy()
            if not self.point_support and np.linalg.norm(line-old)<.01:
                return False
            self.local_point=position.new_tensor(line)[None]
            self.anchor=position+(rotation@self.local_point[...,None]).squeeze(-1)
            # Roll/yaw become constrained at THIS observed orientation. The
            # old point may have freely rotated before its second corner hit;
            # restoring that old orientation would push the new edge through
            # the floor. This changes a reference anchor, never robot state.
            self.initial_rotation=rotation.clone()
            self.point_support=False
            self.half_width=half_width
            return True
        index = int(np.where(loaded, forces, -np.inf).argmax())
        self.local_point = position.new_tensor(local[index])[None]
        self.anchor = position.new_tensor(points[index])[None]
        self.point_support = True
        return True

    def evaluate(self, position, rotation, jacobian, bias, velocity, time,
                 orientation_reference=None):
        if not math.isfinite(time) or time < self.start_time:
            raise ValueError('Toe contact must be evaluated at finite times after activation')
        offset = (rotation @ self.local_point[..., None]).squeeze(-1)
        omega = (jacobian[:, 3:] @ velocity[..., None]).squeeze(-1)
        linear_jac = jacobian[:, :3] - _skew(offset) @ jacobian[:, 3:]
        linear_bias = (bias[:, :3] - (_skew(offset) @ bias[:, 3:, None]).squeeze(-1)
                       + torch.linalg.cross(omega, torch.linalg.cross(omega, offset)))
        inverse = self.frame.transpose(-1, -2)
        mapped_jac = torch.cat((inverse @ linear_jac, inverse @ jacobian[:, 3:]), dim=1)
        mapped_bias = torch.cat(((inverse @ linear_bias[..., None]).squeeze(-1),
                                 (inverse @ bias[:, 3:, None]).squeeze(-1)), dim=-1)
        angle = (inverse @ rotation_error(rotation, self.initial_rotation)[..., None]).squeeze(-1)
        error = torch.cat(((inverse @ (position+offset-self.anchor)[..., None]).squeeze(-1), angle), dim=-1)
        if self.flatten:
            # An observed stance-edge contact is not a commanded toe-off.
            # Restore the level sole through its free angular coordinates,
            # retaining the measured support point/line and its wrench cone.
            flat_error = (inverse @ rotation_error(rotation,self.frame)[...,None]).squeeze(-1)
            angular_rate = (mapped_jac[:,3:] @ velocity[...,None]).squeeze(-1)
            desired = (-50*flat_error-14*angular_rate).clamp(-25,25)
            active = torch.full_like(desired,self.point_support,dtype=torch.bool)
            active[:,1] = True
            task = TaskRequest('stance_flatten',mapped_jac[:,3:],mapped_bias[:,3:],desired,
                torch.eye(3,device=position.device,dtype=position.dtype)[None].expand(len(position),-1,-1)*40,
                active,'medium')
            return mapped_jac,mapped_bias,error,task
        # Lift the heel smoothly over one second. Other contact coordinates
        # remain hard constraints; only this free pitch coordinate is a task.
        phase = min(1., max(0., float(time)-self.start_time))
        target = .25*(10*phase**3-15*phase**4+6*phase**5)
        rate = .25*(30*phase**2-60*phase**3+30*phase**4)
        acceleration = .25*(60*phase-180*phase**2+120*phase**3)
        pitch_jac = mapped_jac[:, 4:5]
        pitch_rate = (pitch_jac @ velocity[..., None]).squeeze(-1)
        desired = acceleration + 40*(target-angle[:, 1:2]) + 12*(rate-pitch_rate)
        source_desired = None
        if orientation_reference is not None:
            target_rotation, target_velocity, target_acceleration = orientation_reference
            if (target_rotation.shape != rotation.shape
                    or target_velocity.shape != position.shape or target_acceleration.shape != position.shape
                    or any(value.device != position.device or value.dtype != position.dtype
                           or not torch.isfinite(value).all() for value in orientation_reference)):
                raise ValueError('Toe orientation reference must match the finite contact batch/device/dtype')
            world_error = rotation_error(rotation,target_rotation)
            source_desired = (inverse @ (target_acceleration-40*world_error
                               +12*(target_velocity-omega))[...,None]).squeeze(-1)
            # Blend the soft angular request over the same 60 ms used for
            # contact confirmation. Hard contact rows and wrench limits do not
            # change. Source toe-off may need far more than the legacy 0.25 rad.
            s=min(1.,max(0.,(float(time)-self.start_time)/.06))
            blend=s**3*(10.-15.*s+6.*s*s)
            desired=(1.-blend)*desired+blend*source_desired[:,1:2]
        if self.point_support:
            # A line fixes roll/yaw kinematically. A single measured corner
            # fixes neither: add soft orientation damping when these rows are
            # released, instead of leaving the unloaded ankle free to spin.
            # No contact moments are introduced; all three rotations remain
            # absent from the hard contact constraint and wrench cone.
            angular_jac = mapped_jac[:, 3:]
            angular_rate = (angular_jac @ velocity[..., None]).squeeze(-1)
            angular_desired = -40*angle-12*angular_rate
            # A rolling support should follow source pitch while damping the
            # newly free roll/yaw about the measured support orientation. A
            # source yaw acceleration is not permission to twist on one corner.
            angular_desired[:, 1:2] = desired
            task = TaskRequest('toe_orientation', angular_jac, mapped_bias[:, 3:],
                angular_desired.clamp(-15, 15),
                torch.eye(3, device=position.device, dtype=position.dtype)[None].expand(len(position),-1,-1)*30.,
                torch.ones_like(angular_desired, dtype=torch.bool), 'medium')
            return mapped_jac,mapped_bias,error,task
        task = TaskRequest('toe_pitch', pitch_jac, mapped_bias[:, 4:5], desired.clamp(-15, 15),
                           position.new_full((len(position), 1, 1), 30.),
                           torch.ones(len(position), 1, dtype=torch.bool, device=position.device), 'medium')
        return mapped_jac, mapped_bias, error, task

    def wrench_to_world_ankle(self, wrench, rotation):
        force = (self.frame @ wrench[:, :3, None]).squeeze(-1)
        moment = (self.frame @ wrench[:, 3:, None]).squeeze(-1)
        offset = (rotation @ self.local_point[..., None]).squeeze(-1)
        return torch.cat((force, moment+torch.linalg.cross(offset, force)), dim=-1)
