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

        # Optional reference task velocity:
        # features = [reference_linear_velocity,
        #             reference_angular_velocity]
        if features is not None:
            if features.shape != x.shape:
                raise ValueError(
                    "Pelvis pose velocity target must have shape "
                    f"{x.shape}, got {features.shape}."
                )

            linear_velocity_error = (
                linear_velocity - features[:, 0:3]
            )

            angular_velocity_error = (
                angular_velocity - features[:, 3:6]
            )
        else:
            linear_velocity_error = linear_velocity
            angular_velocity_error = angular_velocity

        desired_linear_acceleration = (
            -self.position_stiffness * position_error
            -self.position_damping * linear_velocity_error
        )

        desired_angular_acceleration = (
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
