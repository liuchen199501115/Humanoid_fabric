import torch

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm


class JointPostureAttractor(BaseFabricTerm):
    """Soft joint posture task with optional velocity/acceleration feedforward."""

    def metric_eval(self, x, xd, features):
        metric = float(self.params["metric"]) * float(self.params["toggle"])
        value = metric * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        value = value.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.metric is None or not self.graph_capturable:
            self.metric = value.clone()
        else:
            self.metric.copy_(value)

    def force_eval(self, x, xd, features):
        zero = torch.zeros_like(x)
        if isinstance(features, dict):
            velocity, feedforward = features.get("velocity", zero), features.get("acceleration", zero)
        else:
            velocity, feedforward = zero if features is None else features, zero
        if velocity.shape != x.shape or feedforward.shape != x.shape:
            raise ValueError("Joint references must have shape [B,29].")
        acceleration = feedforward - self.params["stiffness"] * x - self.params["damping"] * (xd - velocity)
        limit = self.params["max_acceleration"]
        acceleration = acceleration.clamp(-limit, limit)
        value = -(self.metric @ acceleration.unsqueeze(-1)).squeeze(-1)
        if self.force is None or not self.graph_capturable:
            self.force = value
        else:
            self.force.copy_(value)
