"""Whole-body centroidal data; angular momentum is a velocity map, not a BaseMap."""
import torch

from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix, _skew
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotKinematics


def angular_momentum_matrix(masses, positions, rotations, inertias, linear_jacobians, angular_jacobians):
    """Inertias about link COM in link axes; positions/Jv refer to link COM."""
    # Subtract a nearby origin before mass weighting to avoid cancellation
    # when the robot is far from the world origin. Subtracting COM velocity
    # leaves the momentum unchanged and removes spurious translation columns.
    relative = positions - positions[:, :1]
    com = (masses[None, :, None] * relative).sum(1) / masses.sum()
    com_jac = (masses[None, :, None, None] * linear_jacobians).sum(1) / masses.sum()
    world_inertia = rotations @ inertias[None] @ rotations.transpose(-1, -2)
    matrix = (world_inertia @ angular_jacobians
              + masses[None, :, None, None] * (_skew(relative - com[:, None]) @ (linear_jacobians - com_jac[:, None]))).sum(1)
    return matrix


class G1CentroidalModel:
    """Share the COM task's URDF model and perform one FK for all task geometry."""

    def __init__(self, com_task):
        self.com_task = com_task
        self.kinematics = com_task.robot_kinematics
        # The importer supplies link-axis inertias about the link COM.
        self.inertias = torch.as_tensor(self.kinematics.model.body_inertia.numpy(),
                                       device=com_task.device, dtype=torch.float32)
        if not torch.isfinite(self.inertias).all() or (torch.linalg.eigvalsh(self.inertias) < -1e-7).any():
            raise ValueError("Link inertias must be finite positive semidefinite matrices.")

    def snapshot(self, q):
        transforms, jac = RobotKinematics.apply(q, self.kinematics)
        # Warp tensors are persistent; never retain aliases across FK evaluations.
        transforms, jac = transforms.clone(), jac.transpose(2, 3).clone()
        rotations = _quat_xyzw_to_matrix(transforms[..., 3:])
        angular = (self.kinematics.batch_joint_axes_torch[:, None]
                   * self.com_task.angular_mask[None, :, :, None]).transpose(2, 3).clone()
        offsets = (rotations @ self.com_task.link_com_offsets[None, :, :, None]).squeeze(-1)
        positions = transforms[..., :3] + offsets
        com_jacs = jac - _skew(offsets) @ angular
        masses = self.com_task.link_masses
        weights = masses / masses.sum()
        return dict(transforms=transforms, rotations=rotations, origin_jac=jac,
                    angular_jac=angular, link_com=positions, link_com_jac=com_jacs,
                    com=(weights[None, :, None] * positions).sum(1),
                    com_jac=(weights[None, :, None, None] * com_jacs).sum(1),
                    momentum_matrix=angular_momentum_matrix(masses, positions, rotations,
                                                           self.inertias, com_jacs, angular))

    def mass_matrix(self, snapshot):
        """Physical generalized inertia, for verifying the WBC model convention."""
        jv, jw = snapshot["link_com_jac"], snapshot["angular_jac"]
        rotations = snapshot["rotations"]
        inertia = rotations @ self.inertias[None] @ rotations.transpose(-1, -2)
        return (self.com_task.link_masses[None, :, None, None] * (jv.transpose(-1, -2) @ jv)
                + jw.transpose(-1, -2) @ inertia @ jw).sum(1)

    def validate_mass_matrix(self, q, actual, tolerance=0.05, joint_armature=None):
        """Reject model/order/frame mismatches, including errors in low-inertia joints."""
        expected = self.mass_matrix(self.snapshot(q))
        if joint_armature is not None:
            armature = torch.as_tensor(joint_armature, device=expected.device, dtype=expected.dtype)
            if armature.shape != expected.shape[:1] + (29,) or not torch.isfinite(armature).all() or (armature < 0).any():
                raise ValueError('Joint armature must be finite nonnegative [B,29] in model joint order.')
            expected[:, 6:, 6:] += torch.diag_embed(armature)
        if actual.shape != expected.shape or actual.device != expected.device or not torch.isfinite(actual).all():
            raise ValueError("WBC mass matrix must be finite [B,35,35] on the model device.")
        scale = expected.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
        error = ((actual - expected) / (scale[..., None] * scale[:, None, :])).abs().amax(dim=(1, 2))
        if (error > tolerance).any():
            raise ValueError(f"URDF/WBC inertia mismatch (normalized max {error.tolist()}); check model, joint order and base frame.")
        return error

    def evaluate(self, q, v, integrate, epsilon=1e-3):
        """Centered directional derivative with v held fixed, on configuration manifold."""
        plus = self.snapshot(integrate(q, v, epsilon))
        minus = self.snapshot(integrate(q, v, -epsilon))
        current = self.snapshot(q)
        for key in ("com_jac", "origin_jac", "angular_jac", "momentum_matrix"):
            derivative = (plus[key] - minus[key]) / (2 * epsilon)
            current[key + "_bias"] = (derivative @ v.reshape(v.shape[0], *([1] * (derivative.ndim - 3)), 35, 1)).squeeze(-1)
        current["momentum"] = (current["momentum_matrix"] @ v[..., None]).squeeze(-1)
        return current, plus, minus
