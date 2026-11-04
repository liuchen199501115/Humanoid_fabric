import torch

from fabrics_sim.taskmaps.maps_base import BaseMap


class JointPostureTaskMap(BaseMap):
    """Actuated joint error; quaternion base coordinates are not joints."""

    def __init__(self, batch_size, device, num_joints=29):
        super().__init__(device)
        self.register_buffer("position_target", torch.zeros(batch_size, num_joints, device=device))
        jacobian = torch.zeros(batch_size, num_joints, 6 + num_joints, device=device)
        jacobian[:, :, 6:] = torch.eye(num_joints, device=device)
        self.register_buffer("jacobian", jacobian)

    def set_target(self, target):
        if target.shape != self.position_target.shape or not torch.isfinite(target).all():
            raise ValueError("Joint posture target must be finite and have shape [B,29].")
        self.position_target.copy_(target)

    def forward_position(self, q, features=None):
        return q[:, 7:] - self.position_target, self.jacobian
