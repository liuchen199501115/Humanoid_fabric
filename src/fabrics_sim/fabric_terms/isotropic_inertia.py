import torch

from fabrics_sim.fabric_terms.fabric_term import BaseFabricTerm


class IsotropicInertia(BaseFabricTerm):
    """Zero-acceleration geometry with a constant isotropic metric."""

    def __init__(self, isotropic_mass, device, graph_capturable=True):
        super().__init__(
            is_forcing_policy=False,
            params={"isotropic_mass": float(isotropic_mass)},
            device=device,
            graph_capturable=graph_capturable,
        )

    def metric_eval(self, x, xd, features):
        diagonal = self.params["isotropic_mass"] * torch.ones_like(x)
        metric = torch.diag_embed(diagonal)
        if self.metric is None or not self.graph_capturable:
            self.metric = metric
        else:
            self.metric.copy_(metric)

    def force_eval(self, x, xd, features):
        force = torch.zeros_like(x)
        if self.force is None or not self.graph_capturable:
            self.force = force
        else:
            self.force.copy_(force)
