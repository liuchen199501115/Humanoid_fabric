import torch

from fabrics_sim.taskmaps.maps_base import BaseMap


class FloatingBaseIdentityMap(BaseMap):
    """Identity map from a quaternion configuration to its 35-D tangent space."""

    def __init__(self, num_joints, device):
        super().__init__(device)
        self.velocity_dim = 6 + num_joints
        self.jacobian = None

    def forward_position(self, q, features=None):
        if q.shape[1] != self.velocity_dim + 1:
            raise ValueError(
                f"Expected floating-base configuration dimension "
                f"{self.velocity_dim + 1}; got {q.shape[1]}."
            )

        if self.jacobian is None or self.jacobian.shape[0] != q.shape[0]:
            self.jacobian = (
                torch.eye(self.velocity_dim, device=q.device, dtype=q.dtype)
                .unsqueeze(0)
                .repeat(q.shape[0], 1, 1)
            )

        # EuclideanEnergy only uses the mapped velocity. This 35-D value keeps
        # the map dimension consistent with the floating-base tangent space.
        x = torch.cat((q[:, :3], q[:, 3:6], q[:, 7:]), dim=-1)
        return x, self.jacobian
