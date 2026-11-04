import torch

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm


class PelvisPoseAttractor(BaseFabricTerm):
    """
    Six-dimensional pelvis pose tracking Fabric term.

    Input:
        x  = [position_error, orientation_error]
        xd = [linear_velocity_error, angular_velocity_error]

    Output:
        metric: (batch_size, 6, 6)
        force:  (batch_size, 6)
    """

    def __init__(
        self,
        is_forcing_policy,
        params,
        device,
        graph_capturable=True,
    ):
        super().__init__(
            is_forcing_policy,
            params,
            device,
            graph_capturable,
        )

        self.position_metric = torch.as_tensor(
            params["position_metric"],
            dtype=torch.float32,
            device=device,
        )

        self.orientation_metric = torch.as_tensor(
            params["orientation_metric"],
            dtype=torch.float32,
            device=device,
        )

        self.position_stiffness = torch.as_tensor(
            params["position_stiffness"],
            dtype=torch.float32,
            device=device,
        )

        self.position_damping = torch.as_tensor(
            params["position_damping"],
            dtype=torch.float32,
            device=device,
        )

        self.orientation_stiffness = torch.as_tensor(
            params["orientation_stiffness"],
            dtype=torch.float32,
            device=device,
        )

        self.orientation_damping = torch.as_tensor(
            params["orientation_damping"],
            dtype=torch.float32,
            device=device,
        )

    def metric_eval(self, x, xd, features):
        batch_size = x.shape[0]

        metric_diagonal = torch.cat(
            [
                self.position_metric,
                self.orientation_metric,
            ],
            dim=0,
        )

        metric = torch.diag_embed(
            metric_diagonal.unsqueeze(0).expand(
                batch_size,
                -1,
            )
        )

        toggle = self.params["toggle"]

        if torch.is_tensor(toggle):
            if toggle.ndim == 0:
                metric = toggle * metric
            else:
                metric = toggle.reshape(
                    batch_size, 1, 1
                ) * metric
        else:
            metric = float(toggle) * metric

        if self.metric is None or not self.graph_capturable:
            self.metric = metric
        else:
            self.metric.copy_(metric)

    def force_eval(self, x, xd, features):
        position_error = x[:, 0:3]
        orientation_error = x[:, 3:6]

        linear_velocity = xd[:, 0:3]
        angular_velocity = xd[:, 3:6]

        reference_velocity, reference_acceleration = (
            self._parse_reference(features, x)
        )
        linear_velocity_error = linear_velocity - reference_velocity[:, 0:3]
        angular_velocity_error = angular_velocity - reference_velocity[:, 3:6]

        desired_linear_acceleration = (
            reference_acceleration[:, 0:3]
            -self.position_stiffness * position_error
            -self.position_damping * linear_velocity_error
        )

        desired_angular_acceleration = (
            reference_acceleration[:, 3:6]
            -self.orientation_stiffness * orientation_error
            -self.orientation_damping * angular_velocity_error
        )

        desired_task_acceleration = torch.cat(
            [
                desired_linear_acceleration,
                desired_angular_acceleration,
            ],
            dim=-1,
        )

        # Natural form:
        #
        #   M xdd + f = 0
        #   f = -M xdd_desired
        force = -torch.bmm(
            self.metric,
            desired_task_acceleration.unsqueeze(-1),
        ).squeeze(-1)

        if self.force is None or not self.graph_capturable:
            self.force = force
        else:
            self.force.copy_(force)

    @staticmethod
    def _parse_reference(features, x):
        """Return desired task velocity and acceleration.

        A tensor feature remains the legacy velocity-only interface.  A dict
        permits the trajectory controller to add a feed-forward acceleration:
        ``a_des = a_ref - Kp * e - Kd * (xd - xd_ref)``.
        """
        zeros = torch.zeros_like(x)
        if features is None:
            return zeros, zeros
        if torch.is_tensor(features):
            if features.shape != x.shape:
                raise ValueError(
                    "Pelvis pose velocity target must have shape "
                    f"{x.shape}, got {features.shape}."
                )
            return features, zeros
        if not isinstance(features, dict):
            raise TypeError("Pelvis reference must be a tensor or dict.")
        velocity = features.get("velocity", zeros)
        acceleration = features.get("acceleration", zeros)
        if velocity.shape != x.shape or acceleration.shape != x.shape:
            raise ValueError("Pelvis reference tensors must match task shape.")
        return velocity, acceleration
