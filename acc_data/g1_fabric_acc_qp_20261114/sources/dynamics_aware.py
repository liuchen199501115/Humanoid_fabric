"""G1 physical task requests, including nonholonomic momentum and contact gating."""
from pathlib import Path

import torch
import yaml

from fabrics_sim.fabrics.task_request import TaskRequest
from fabrics_sim.taskmaps.g1_centroidal_momentum import G1CentroidalModel
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix, _skew


def rotation_error(rotation, target):
    """World-frame Log(R Rref.T), with a finite, nonzero axis at pi."""
    r = rotation @ target.transpose(-1, -2)
    vee = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                       r[..., 1, 0] - r[..., 0, 1]), -1) * 0.5
    sine = torch.linalg.vector_norm(vee, dim=-1)
    cosine = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    regular = vee * (angle / sine.clamp_min(1e-8))[..., None]
    # Near pi, the largest column of R+I supplies a well-conditioned axis.
    symmetric = 0.5 * (r + r.transpose(-1, -2)) + torch.eye(3, device=r.device, dtype=r.dtype)
    index = symmetric.diagonal(dim1=-2, dim2=-1).argmax(-1)
    axis = symmetric.gather(-1, index[..., None, None].expand(*index.shape, 3, 1)).squeeze(-1)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = torch.where(((axis * vee).sum(-1) < 0)[..., None], -axis, axis)
    return torch.where((cosine < -0.9999)[..., None], axis * angle[..., None], regular)


def sphere_distance(delta, relative_jacobian, radii):
    norm = torch.linalg.vector_norm(delta, dim=-1)
    normal = delta / norm[..., None].clamp_min(1e-8)
    fallback = torch.zeros_like(normal)
    fallback[..., 0] = 1
    normal = torch.where((norm < 1e-8)[..., None], fallback, normal)
    return norm - radii, (normal[..., None, :] @ relative_jacobian).squeeze(-2)


class DynamicsAwareTasks:
    def __init__(self, fabric):
        self.fabric = fabric
        path = Path(__file__).resolve().parents[1] / "fabric_params/g1_dynamics_aware.yaml"
        self.config = yaml.safe_load(path.read_text())
        self.set_hand_tracking(self.config.get("hand_tracking", "position"))
        self.model = G1CentroidalModel(fabric.center_of_mass_taskmap)
        self.contacts = torch.zeros(fabric.batch_size, 2, device=fabric.device, dtype=torch.bool)
        self.hand_active = torch.ones_like(self.contacts)
        self.momentum_reference = torch.zeros(fabric.batch_size, 3, device=fabric.device)
        self.momentum_rate_reference = torch.zeros_like(self.momentum_reference)
        self.body_reference = None
        self.obstacles = torch.empty(fabric.batch_size, 0, 4, device=fabric.device)
        self.collision_enabled = True
        self.momentum_enabled = True
        self.last_snapshot = None
        self.cache_task_requests = True
        self.last_requests = None
        self.managed_walking = False
        # Opt-in for a dynamically checked full-body reference. Only feedback
        # is bounded here; QP torque/contact/acceleration limits remain hard.
        self.preserve_reference_acceleration = False
        self.source_pelvis_metric_scale = 0.0
        self.wrist_hip_barrier = False
        self.collision_barrier_bandwidth = 8.0
        self.convex_wrist_geometry = None
        self.stable_constraint_rows = False
        self.last_collision_constraints = None
        self.original_collision_pairs = len(self.config['collision']['pairs'])
        self._set_collision_geometry(self.config["collision"]["spheres"], self.config["collision"]["pairs"])

    def _set_collision_geometry(self, spheres, pairs, weights=None):
        self.sphere_names = list(spheres)
        self.sphere_indices = [self.model.kinematics.get_link_index(s["link"]) for s in spheres.values()]
        self.sphere_centers = torch.tensor([s["center"] for s in spheres.values()], device=self.fabric.device)
        self.sphere_radii = torch.tensor([s["radius"] for s in spheres.values()], device=self.fabric.device)
        self.pairs = torch.tensor([[self.sphere_names.index(a), self.sphere_names.index(b)]
                                   for a, b in pairs], device=self.fabric.device)
        self.collision_pair_weights = torch.tensor(
            [self.config['collision']['metric']]*len(pairs) if weights is None else weights,
            device=self.fabric.device)

    def enable_wrist_hip_avoidance(self, metric=20., include_upper_body=False, convex=False):
        """Add mesh-derived wrist/hip covers; preserve the original pair tasks."""
        from fabrics_sim.reference.collision_geometry import collision_mesh_spheres
        spheres = dict(self.config['collision']['spheres'])
        pairs = list(self.config['collision']['pairs'])
        weights = [self.config['collision']['metric']]*len(pairs)
        wrists, hips = [], []
        for side in ('left', 'right'):
            for joint, group in [('wrist_roll', wrists), ('wrist_pitch', wrists),
                                 ('wrist_yaw', wrists), ('hip_pitch', hips),
                                 ('hip_roll', hips), ('hip_yaw', hips), ('rubber_hand', wrists)]:
                link = f'{side}_{joint}' if joint == 'rubber_hand' else f'{side}_{joint}_link'
                # PhysX reports rubber-hand collisions in the merged wrist
                # actor although the matching URDF lists that mesh as visual.
                kind = 'visual' if joint == 'rubber_hand' else 'collision'
                for index, (center, radius) in enumerate(collision_mesh_spheres(
                        self.fabric.urdf_path, link, geometry_kind=kind)):
                    name = f'{link}_cover_{index}'
                    spheres[name] = dict(link=link, center=center, radius=radius)
                    group.append(name)
        for wrist in wrists:
            for hip in hips:
                pairs.append((wrist, hip)); weights.append(metric)
        if include_upper_body:
            groups={}
            for link in ['torso_link']+[f'{side}_shoulder_{axis}_link'
                                       for side in ('left','right') for axis in ('pitch','roll','yaw')]:
                groups[link]=[]
                for index,(center,radius) in enumerate(collision_mesh_spheres(self.fabric.urdf_path,link)):
                    name=f'{link}_cover_{index}'
                    spheres[name]=dict(link=link,center=center,radius=radius)
                    groups[link].append(name)
            for wrist in wrists:
                opposite='right' if wrist.startswith('left_') else 'left'
                for link,covers in groups.items():
                    if link=='torso_link' or link.startswith(opposite+'_'):
                        for cover in covers:
                            pairs.append((wrist,cover));weights.append(metric)
        self._set_collision_geometry(spheres, pairs, weights)
        self.convex_wrist_geometry = None
        if convex:
            from fabrics_sim.reference.convex_collision import ConvexLinkCollision
            link_pairs=tuple(dict.fromkeys((spheres[a]['link'],spheres[b]['link'])
                for a,b in pairs[self.original_collision_pairs:]))
            self.convex_wrist_geometry=ConvexLinkCollision(self.fabric.urdf_path,self.model.kinematics,link_pairs)
            self.convex_pair_weights=torch.cat((self.collision_pair_weights[:self.original_collision_pairs],
                self.collision_pair_weights.new_full((len(link_pairs),),metric)))

    @property
    def barrier_pair_count(self):
        return (len(self.pairs) if self.convex_wrist_geometry is None else
                self.original_collision_pairs+len(self.convex_wrist_geometry.pairs))

    def collision_geometry(self, snapshot):
        distance, jacobian = self._sphere_geometry(snapshot)
        if self.convex_wrist_geometry is None:
            return distance,jacobian
        convex_distance,convex_jacobian=self.convex_wrist_geometry.evaluate(snapshot)
        # Preserve original sphere tasks and obstacle rows; replace only the
        # optional wrist/body cover pairs, for both soft and hard constraints.
        return (torch.cat((distance[:,:self.original_collision_pairs],convex_distance,distance[:,len(self.pairs):]),1),
                torch.cat((jacobian[:,:self.original_collision_pairs],convex_jacobian,jacobian[:,len(self.pairs):]),1))


    def set_hand_tracking(self, mode):
        if mode not in ("off", "position", "pose"):
            raise ValueError("Hand tracking must be off, position, or pose")
        self.hand_tracking = mode

    def _reference(self, value, shape, name):
        if (value.shape != shape or value.device != self.momentum_reference.device
                or value.dtype != self.momentum_reference.dtype or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite {shape}, matching model device/dtype.")
        return value.clone()

    def set_contact_state(self, active):
        if active.shape != self.contacts.shape or active.dtype != torch.bool or active.device != self.contacts.device:
            raise ValueError("Contact state must be boolean [B,2] on the model device.")
        self.contacts.copy_(active)

    def set_momentum_target(self, reference, rate=None):
        reference = self._reference(reference, self.momentum_reference.shape, "Momentum reference")
        rate = torch.zeros_like(reference) if rate is None else self._reference(rate, reference.shape, "Momentum rate")
        self.momentum_reference.copy_(reference)
        self.momentum_rate_reference.copy_(rate)

    def set_body_target(self, quaternion, velocity=None, acceleration=None, link="pelvis"):
        quaternion = self._reference(quaternion, (self.fabric.batch_size, 4), "Body quaternion")
        if (torch.linalg.vector_norm(quaternion, dim=-1) < 1e-8).any():
            raise ValueError("Body quaternion must be nonzero.")
        zero = torch.zeros_like(self.momentum_reference)
        velocity = zero if velocity is None else self._reference(velocity, zero.shape, "Body angular velocity")
        acceleration = zero if acceleration is None else self._reference(acceleration, zero.shape, "Body angular acceleration")
        self.model.kinematics.get_link_index(link)
        self.body_reference = (quaternion, velocity, acceleration, link)

    def set_obstacles(self, spheres):
        """Static world spheres [B,N,4] = xyz,radius; ground support is not an obstacle."""
        if spheres.ndim != 3 or spheres.shape[0] != self.fabric.batch_size or spheres.shape[-1] != 4:
            raise ValueError("Obstacle spheres must have shape [B,N,4].")
        spheres = self._reference(spheres, spheres.shape, "Obstacle spheres")
        if (spheres[..., 3] <= 0).any():
            raise ValueError("Obstacle radii must be positive.")
        self.obstacles = spheres

    @staticmethod
    def references(features, template):
        zero = torch.zeros_like(template)
        if features is None:
            return zero, zero
        if torch.is_tensor(features):
            return features, zero
        velocity, acceleration = features.get("velocity"), features.get("acceleration")
        return zero if velocity is None else velocity, zero if acceleration is None else acceleration

    def _request(self, name, A, bias, desired, weights, active=None, priority="low"):
        weights = torch.as_tensor(weights, device=A.device, dtype=A.dtype)
        diagonal = weights.expand_as(desired)
        if active is None:
            active = torch.ones_like(desired, dtype=torch.bool)
        return TaskRequest(name, A, bias, desired, torch.diag_embed(diagonal), active, priority)

    def _sphere_geometry(self, snapshot):
        rotation = snapshot["rotations"][:, self.sphere_indices]
        offset = (rotation @ self.sphere_centers[None, :, :, None]).squeeze(-1)
        position = snapshot["transforms"][:, self.sphere_indices, :3] + offset
        jac = snapshot["origin_jac"][:, self.sphere_indices] - _skew(offset) @ snapshot["angular_jac"][:, self.sphere_indices]
        a, b = self.pairs.unbind(-1)
        distance, rows = sphere_distance(position[:, a] - position[:, b], jac[:, a] - jac[:, b],
                                         self.sphere_radii[a] + self.sphere_radii[b])
        if self.obstacles.shape[1]:
            delta = position[:, :, None] - self.obstacles[:, None, :, :3]
            obstacle_d, obstacle_j = sphere_distance(delta, jac[:, :, None],
                self.sphere_radii[None, :, None] + self.obstacles[:, None, :, 3])
            distance = torch.cat((distance, obstacle_d.flatten(1)), 1)
            rows = torch.cat((rows, obstacle_j.flatten(1, 2)), 1)
        return distance, rows

    def _limit_tracking(self, desired, reference, limit):
        if self.preserve_reference_acceleration:
            return reference + (desired-reference).clamp(-limit, limit)
        return desired.clamp(-limit, limit)

    @torch.no_grad()
    def compute(self, q, v):
        f = self.fabric
        if q.dtype != self.momentum_reference.dtype or q.device != self.momentum_reference.device or v.dtype != q.dtype or v.device != q.device:
            raise ValueError("Dynamics-aware states must match the float32 kinematics model's device/dtype.")
        if not torch.isfinite(q).all() or not torch.isfinite(v).all() or (q[:, 3:7].norm(dim=-1) < 1e-8).any():
            raise ValueError("States must be finite and base quaternions must be nonzero.")
        eps = self.config["difference_step"]
        state, plus, minus = self.model.evaluate(q, v, f.integrate_root_position, eps)
        self.last_snapshot = state
        self.last_collision_constraints = None
        requests = []
        def tensor(value):
            return torch.as_tensor(value, device=q.device, dtype=q.dtype)

        params = f.fabric_params["center_of_mass_attractor"]
        velocity, acceleration = self.references(f.fabrics_features["center_of_mass"]["center_of_mass_attractor"], state["com"])
        desired = acceleration - tensor(params["stiffness"]) * (state["com"] - f.center_of_mass_taskmap.position_target)
        desired -= tensor(params["damping"]) * ((state["com_jac"] @ v[..., None]).squeeze(-1) - velocity)
        limit = tensor(params["max_acceleration"])
        requests.append(self._request("center_of_mass", state["com_jac"], state["com_jac_bias"],
                                      self._limit_tracking(desired, acceleration, limit), tensor(params["metric"]) * params["toggle"], priority="high"))

        params = self.config["momentum"]
        desired = self._limit_tracking(
            self.momentum_rate_reference-tensor(params["damping"])*(state["momentum"]-self.momentum_reference),
            self.momentum_rate_reference, tensor(params["max_rate"]))
        requests.append(self._request("centroidal_momentum", state["momentum_matrix"], state["momentum_matrix_bias"],
                                      desired, params["metric"],
                                      torch.full_like(desired, self.momentum_enabled, dtype=torch.bool), priority="high"))

        params = self.config["body"]
        if self.body_reference is None:
            quaternion = f.pelvis_pose_taskmap.orientation_target
            velocity, acceleration = self.references(f.fabrics_features["pelvis_pose"]["pelvis_pose_attractor"], torch.zeros(q.shape[0], 6, device=q.device, dtype=q.dtype))
            velocity, acceleration = velocity[:, 3:], acceleration[:, 3:]
            link = params["link"]
        else:
            quaternion, velocity, acceleration, link = self.body_reference
        index = self.model.kinematics.get_link_index(link)
        A = state["angular_jac"][:, index]
        error = rotation_error(state["rotations"][:, index], _quat_xyzw_to_matrix(quaternion))
        desired = acceleration - tensor(params["stiffness"]) * error - tensor(params["damping"]) * ((A @ v[..., None]).squeeze(-1) - velocity)
        limit = tensor(params["max_acceleration"])
        requests.append(self._request("body_orientation", A, state["angular_jac_bias"][:, index], self._limit_tracking(desired, acceleration, limit), params["metric"], priority="high"))

        if self.source_pelvis_metric_scale > 0:
            # A torso target alone allows pelvis tilt/yaw to be compensated by
            # the waist. Retain a separate, soft pelvis reference from the
            # same motion clock; neither orientation overrides contact physics.
            index = self.model.kinematics.get_link_index("pelvis")
            A = state["angular_jac"][:, index]
            error = rotation_error(state["rotations"][:, index],
                                   _quat_xyzw_to_matrix(f.pelvis_pose_taskmap.orientation_target))
            velocity, acceleration = self.references(
                f.fabrics_features["pelvis_pose"]["pelvis_pose_attractor"],
                torch.zeros(q.shape[0], 6, device=q.device, dtype=q.dtype))
            desired = acceleration[:, 3:] - tensor(params["stiffness"]) * error
            desired -= tensor(params["damping"]) * ((A @ v[..., None]).squeeze(-1) - velocity[:, 3:])
            requests.append(self._request("source_pelvis_orientation", A,
                state["angular_jac_bias"][:, index], self._limit_tracking(desired, acceleration[:, 3:], limit),
                tensor(params["metric"]) * self.source_pelvis_metric_scale, priority="high"))

        for name, taskmap, mask, priority in (("feet_pose", f.feet_pose_taskmap, ~self.contacts, "medium"),
                                             ("hand_pose", f.hand_pose_taskmap, self.hand_active, "low")):
            params = f.fabric_params[name + "_attractor"]
            # Resolve by link name, not by another Kinematics instance's integer ordering.
            indices = [self.model.kinematics.get_link_index(taskmap.frame_groups[n][0]) for n in taskmap.end_effector_order]
            A = torch.cat((state["origin_jac"][:, indices], state["angular_jac"][:, indices]), 2).flatten(1, 2)
            bias = torch.cat((state["origin_jac_bias"][:, indices], state["angular_jac_bias"][:, indices]), 2).flatten(1)
            error = torch.cat((state["transforms"][:, indices, :3] - taskmap.position_target,
                               rotation_error(state["rotations"][:, indices], _quat_xyzw_to_matrix(taskmap.quaternion_target))), -1).flatten(1)
            velocity, acceleration = self.references(f.fabrics_features[name][name + "_attractor"], error)
            desired = acceleration - tensor(params["stiffness"]) * error - tensor(params["damping"]) * ((A @ v[..., None]).squeeze(-1) - velocity)
            offset = acceleration if self.preserve_reference_acceleration else torch.zeros_like(acceleration)
            blocks = (desired-offset).reshape(q.shape[0], 2, 2, 3)
            limits = tensor([params["max_linear_acceleration"], params["max_angular_acceleration"]]).T
            scale = (limits[None, :, :, None] / torch.linalg.vector_norm(blocks, dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
            active_rows = mask.repeat_interleave(6, dim=1).clone()
            if name == "hand_pose":
                if self.hand_tracking == "off":
                    active_rows.zero_()
                elif self.hand_tracking == "position":
                    active_rows.reshape(q.shape[0], 2, 6)[:, :, 3:] = False
            requests.append(self._request(name, A, bias, offset+(blocks * scale).flatten(1),
                tensor(params["metric"]) * params["toggle"] * (
                    self.config["managed_walking"]["swing_metric_scale"]
                    if self.managed_walking and name == "feet_pose" else 1.), active_rows, priority))

        if f.joint_posture_enabled:
            params = f.fabric_params["joint_posture_attractor"]
            error, A = f.joint_posture_taskmap(q, None)
            velocity, acceleration = self.references(f.fabrics_features["joint_posture"]["joint_posture_attractor"], error)
            desired = acceleration - params["stiffness"] * error - params["damping"] * (v[:, 6:] - velocity)
            requests.append(self._request("joint_posture", A, torch.zeros_like(error),
                self._limit_tracking(desired, acceleration, params["max_acceleration"]),
                params["metric"] * params["toggle"] * (
                    self.config["managed_walking"]["posture_metric_scale"] if self.managed_walking else 1.)))

        if self.collision_enabled:
            params = self.config["collision"]
            distance, A = self.collision_geometry(state)
            _, jac_plus = self.collision_geometry(plus)
            _, jac_minus = self.collision_geometry(minus)
            bias = (((jac_plus - jac_minus) / (2 * eps)) @ v[..., None]).squeeze(-1)
            speed = (A @ v[..., None]).squeeze(-1)
            safe, engage = params["safe_distance"], params["activation_distance"]
            if self.wrist_hip_barrier:
                from fabrics_sim.solver.collision_barrier import wrist_collision_barrier
                self.last_collision_constraints = wrist_collision_barrier(
                    distance, A, bias, speed, self.original_collision_pairs,
                    self.barrier_pair_count, safe, self.stable_constraint_rows, self.collision_barrier_bandwidth)
            t = ((engage - distance) / (engage - safe)).clamp(0, 1)
            gate = t.square() * (3 - 2 * t)
            desired = (params["stiffness"] * ((distance - safe).clamp_min(0.005).pow(-3) - (engage - safe)**-3).clamp_min(0)
                       - params["damping"] * speed.clamp_max(0)) * gate
            requests.append(self._request("collision", A, bias, desired.clamp(0, params["max_acceleration"]),
                torch.cat((self.collision_pair_weights if self.convex_wrist_geometry is None else self.convex_pair_weights,
                    q.new_full((distance.shape[1]-self.barrier_pair_count,), params['metric']))) * gate,
                distance < engage))
            state["collision_distance"] = distance
        # Acceleration-only control consumes these leaves locally, without
        # retaining a second task-level interface to the downstream QP.
        self.last_requests = requests if self.cache_task_requests else None
        return requests
