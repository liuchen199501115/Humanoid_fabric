from __future__ import annotations

import torch

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm


class G1EndEffectorPoseAttractor(BaseFabricTerm):
    """PD acceleration policy written in Fabric natural form.

    Each instance controls one selectable end-effector group (feet or hands).
    Every end effector contributes [position(3), orientation(3)].

    Optional features contain the desired world-frame twists in the same
    ordering and have the same shape as the task map output.
    """

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
        self.task_dimension = self.metric_diagonal.numel()
        if self.task_dimension == 0 or self.task_dimension % 6 != 0:
            raise ValueError("metric length must be a positive multiple of 6.")
        self.num_end_effectors = self.task_dimension // 6
        self.stiffness = self._read_vector(
            params, "stiffness", device, self.task_dimension
        )
        self.damping = self._read_vector(
            params, "damping", device, self.task_dimension
        )
        self.max_linear_acceleration = self._read_limit(
            params, "max_linear_acceleration", device
        )
        self.max_angular_acceleration = self._read_limit(
            params, "max_angular_acceleration", device
        )

    @staticmethod
    def _read_vector(
        params: dict,
        key: str,
        device: str,
        expected_size: int | None = None,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            params[key],
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if expected_size is not None and value.numel() != expected_size:
            raise ValueError(
                f"{key} must have {expected_size} values; "
                f"got {value.numel()}."
            )
        return value

    def _read_limit(
        self,
        params: dict,
        key: str,
        device: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            params[key],
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if value.numel() not in (1, self.num_end_effectors):
            raise ValueError(
                f"{key} must contain one or {self.num_end_effectors} values."
            )
        if value.numel() == 1:
            value = value.repeat(self.num_end_effectors)
        return value

    def metric_eval(self, x, xd, features):
        if x.shape[-1] != self.task_dimension:
            raise ValueError(
                f"Expected a {self.task_dimension}-D task, got {x.shape[-1]}."
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
        zeros = torch.zeros_like(x)
        if features is None:
            target_twist, target_acceleration = zeros, zeros
        elif torch.is_tensor(features):
            if features.shape != x.shape:
                raise ValueError(
                    "Desired end-effector twist must have shape "
                    f"{tuple(x.shape)}, got {tuple(features.shape)}."
                )
            target_twist, target_acceleration = features, zeros
        elif isinstance(features, dict):
            target_twist = features.get("velocity", zeros)
            target_acceleration = features.get("acceleration", zeros)
            if target_twist.shape != x.shape or target_acceleration.shape != x.shape:
                raise ValueError("End-effector reference tensors must match task shape.")
        else:
            raise TypeError("End-effector reference must be a tensor or dict.")

        velocity_error = xd - target_twist
        desired_acceleration = (
            target_acceleration - self.stiffness * x
            -self.damping * velocity_error
        )

        # Limit each end effector independently.
        limited_blocks = []
        for index in range(self.num_end_effectors):
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
