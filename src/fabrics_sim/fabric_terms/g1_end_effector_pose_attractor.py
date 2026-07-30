from __future__ import annotations

import torch

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm


class G1EndEffectorPoseAttractor(BaseFabricTerm):
    """PD acceleration policy written in Fabric natural form.

    Task order:
        left_foot  [position(3), orientation(3)]
        right_foot [position(3), orientation(3)]
        left_hand  [position(3), orientation(3)]
        right_hand [position(3), orientation(3)]

    Optional features contain the desired world-frame twists in the same
    ordering and have shape (batch_size, 24).
    """

    TASK_DIMENSION = 24

    def __init__(
        self,
        is_forcing_policy: bool,
        params: dict,
        device: str,
        graph_capturable: bool = True,
    ):
        if not is_forcing_policy:
            raise ValueError(
                "G1EndEffectorPoseAttractor is a forcing policy."
            )
        super().__init__(
            is_forcing_policy,
            params,
            device,
            graph_capturable,
        )

        self.metric_diagonal = self._read_vector(
            params, "metric", device
        )
        self.stiffness = self._read_vector(
            params, "stiffness", device
        )
        self.damping = self._read_vector(
            params, "damping", device
        )
        self.max_linear_acceleration = self._read_limit(
            params, "max_linear_acceleration", device
        )
        self.max_angular_acceleration = self._read_limit(
            params, "max_angular_acceleration", device
        )

    @classmethod
    def _read_vector(
        cls,
        params: dict,
        key: str,
        device: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            params[key],
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if value.numel() != cls.TASK_DIMENSION:
            raise ValueError(
                f"{key} must have {cls.TASK_DIMENSION} values; "
                f"got {value.numel()}."
            )
        return value

    @staticmethod
    def _read_limit(
        params: dict,
        key: str,
        device: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            params[key],
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if value.numel() not in (1, 4):
            raise ValueError(f"{key} must contain one or four values.")
        if value.numel() == 1:
            value = value.repeat(4)
        return value

    def metric_eval(self, x, xd, features):
        if x.shape[-1] != self.TASK_DIMENSION:
            raise ValueError(
                f"Expected a 24-D task, got {x.shape[-1]}."
            )

        metric = torch.diag_embed(
            self.metric_diagonal.unsqueeze(0).expand(x.shape[0], -1)
        )

        toggle = self.params["toggle"]
        if torch.is_tensor(toggle):
            if toggle.ndim == 0:
                metric = toggle * metric
            else:
                metric = toggle.reshape(x.shape[0], 1, 1) * metric
        else:
            metric = float(toggle) * metric

        if self.metric is None or not self.graph_capturable:
            self.metric = metric
        else:
            self.metric.copy_(metric)

    @staticmethod
    def _limit_vector_norm(
        vector: torch.Tensor,
        maximum: torch.Tensor,
    ) -> torch.Tensor:
        norm = torch.linalg.vector_norm(
            vector, dim=-1, keepdim=True
        )
        scale = torch.clamp(
            maximum.unsqueeze(-1) / norm.clamp_min(1.0e-8),
            max=1.0,
        )
        return vector * scale

    def force_eval(self, x, xd, features):
        if features is None:
            target_twist = torch.zeros_like(x)
        else:
            if features.shape != x.shape:
                raise ValueError(
                    "Desired end-effector twist must have shape "
                    f"{tuple(x.shape)}, got {tuple(features.shape)}."
                )
            target_twist = features

        velocity_error = xd - target_twist
        desired_acceleration = (
            -self.stiffness * x
            -self.damping * velocity_error
        )

        # Limit each end effector independently.
        limited_blocks = []
        for index in range(4):
            block = desired_acceleration[:, 6 * index:6 * (index + 1)]
            linear = self._limit_vector_norm(
                block[:, :3],
                self.max_linear_acceleration[index].expand(block.shape[0]),
            )
            angular = self._limit_vector_norm(
                block[:, 3:],
                self.max_angular_acceleration[index].expand(block.shape[0]),
            )
            limited_blocks.append(torch.cat((linear, angular), dim=-1))

        desired_acceleration = torch.cat(limited_blocks, dim=-1)

        # Natural form: M xdd + f = 0, hence f = -M xdd_des.
        force = -torch.bmm(
            self.metric,
            desired_acceleration.unsqueeze(-1),
        ).squeeze(-1)

        if self.force is None or not self.graph_capturable:
            self.force = force
        else:
            self.force.copy_(force)