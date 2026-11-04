from __future__ import annotations

import torch

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm


class CenterOfMassAttractor(BaseFabricTerm):
    """Three-dimensional PD attractor for the whole-body center of mass."""

    def __init__(
        self,
        is_forcing_policy,
        params,
        device,
        graph_capturable=True,
    ):
        super().__init__(
            is_forcing_policy, params, device, graph_capturable
        )
        self.metric_diagonal = torch.as_tensor(
            params["metric"], dtype=torch.float32, device=device
        )
        self.stiffness = torch.as_tensor(
            params["stiffness"], dtype=torch.float32, device=device
        )
        self.damping = torch.as_tensor(
            params["damping"], dtype=torch.float32, device=device
        )
        self.max_acceleration = torch.as_tensor(
            params["max_acceleration"], dtype=torch.float32, device=device
        )

    def metric_eval(self, x, xd, features):
        del xd, features
        metric = torch.diag_embed(
            self.metric_diagonal.unsqueeze(0).expand(x.shape[0], -1)
        )
        toggle = self.params["toggle"]
        if torch.is_tensor(toggle):
            metric = toggle.reshape(-1, 1, 1) * metric
        else:
            metric = float(toggle) * metric
        if self.metric is None or not self.graph_capturable:
            self.metric = metric
        else:
            self.metric.copy_(metric)

    def requested_acceleration(self, x, xd, features):
        """Return the clipped task request, shared by control and diagnostics."""
        zeros = torch.zeros_like(x)
        if features is None:
            velocity_target, acceleration_target = zeros, zeros
        elif torch.is_tensor(features):
            if features.shape != x.shape:
                raise ValueError(
                    "COM velocity target must have shape "
                    f"{x.shape}, got {features.shape}."
                )
            velocity_target, acceleration_target = features, zeros
        elif isinstance(features, dict):
            velocity_target = features.get("velocity", zeros)
            acceleration_target = features.get("acceleration", zeros)
            if velocity_target.shape != x.shape or acceleration_target.shape != x.shape:
                raise ValueError("COM reference tensors must match task shape.")
        else:
            raise TypeError("COM reference must be a tensor or dict.")
        velocity_error = xd - velocity_target

        desired_acceleration = (
            acceleration_target - self.stiffness * x - self.damping * velocity_error
        )
        desired_acceleration = torch.clamp(
            desired_acceleration,
            min=-self.max_acceleration,
            max=self.max_acceleration,
        )
        return desired_acceleration

    def force_eval(self, x, xd, features):
        desired_acceleration = self.requested_acceleration(x, xd, features)
        force = -torch.bmm(
            self.metric, desired_acceleration.unsqueeze(-1)
        ).squeeze(-1)
        if self.force is None or not self.graph_capturable:
            self.force = force
        else:
            self.force.copy_(force)
