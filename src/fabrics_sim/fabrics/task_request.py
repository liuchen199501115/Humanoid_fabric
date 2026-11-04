"""Acceleration-level task intentions shared by Fabric and constrained WBC."""
from dataclasses import dataclass

import torch


@dataclass
class TaskRequest:
    """Soft objective ||A a + bias - desired||_W; all rows use physical units.

    ``priority`` is descriptive grouping, not a lexicographic guarantee.
    Inactive rows are removed on both sides of the metric.
    """

    name: str
    A: torch.Tensor
    bias: torch.Tensor
    desired: torch.Tensor
    metric: torch.Tensor
    active: torch.Tensor
    priority: str = "low"

    def __post_init__(self):
        if self.A.ndim != 3 or self.A.shape[-1] != 35:
            raise ValueError("Task A must have shape [B,k,35].")
        batch, rows, _ = self.A.shape
        for name, shape in (("bias", (batch, rows)), ("desired", (batch, rows)),
                            ("metric", (batch, rows, rows)), ("active", (batch, rows))):
            value = getattr(self, name)
            if value.shape != shape or value.device != self.A.device:
                raise ValueError(f"Task {name} must have shape {shape} and share A's device.")
            if name != "active" and value.dtype != self.A.dtype:
                raise ValueError(f"Task {name} must share A's dtype.")
            if not torch.isfinite(value).all():
                raise ValueError(f"Task {name} must be finite.")
        if not torch.isfinite(self.A).all() or self.active.dtype != torch.bool:
            raise ValueError("Task A must be finite and active must be boolean.")
        if not torch.allclose(self.metric, self.metric.transpose(-1, -2), atol=1e-6):
            raise ValueError("Task metric must be symmetric.")
        diagonal = self.metric.diagonal(dim1=-2, dim2=-1)
        # Physical tasks normally use diagonal weights. A full eigensolve
        # would be cubic in the number of collision pairs at every tick.
        eigenvalues = (diagonal if torch.equal(self.metric, torch.diag_embed(diagonal))
                       else torch.linalg.eigvalsh(self.metric))
        if (eigenvalues < -1e-6).any():
            raise ValueError("Task metric must be positive semidefinite.")

    def natural_form(self, dtype=None):
        A = self.A.to(dtype=dtype or self.A.dtype)
        weights = self.metric.to(A.dtype) * self.active[..., None] * self.active[:, None, :]
        pullback = A.transpose(1, 2) @ weights
        return pullback @ A, (pullback @ (self.bias.to(A.dtype) - self.desired.to(A.dtype))[..., None]).squeeze(-1)

    def residual(self, acceleration):
        # A composed root may retain a double-precision metric/Jacobian while
        # the simulator supplies float32 acceleration. Evaluate in task dtype.
        acceleration = acceleration.to(device=self.A.device, dtype=self.A.dtype)
        return ((self.A @ acceleration[..., None]).squeeze(-1)
                + self.bias - self.desired) * self.active

    def project(self, projection):
        """Express selected task directions in an orthonormal row basis.

        Disabled coordinates contribute neither targets nor metric. The
        original task is unchanged, so its full objective can be retained.
        """
        if (projection.ndim != 3 or projection.shape[0] != self.A.shape[0]
                or projection.shape[2] != self.A.shape[1] or projection.shape[1] == 0
                or projection.device != self.A.device or projection.dtype != self.A.dtype
                or not torch.isfinite(projection).all()):
            raise ValueError("Task projection must be finite [B,r,k] and match task dtype/device.")
        identity = torch.eye(projection.shape[1], device=self.A.device, dtype=self.A.dtype)
        if not torch.allclose(projection @ projection.transpose(-1, -2), identity[None], atol=1e-6):
            raise ValueError("Task projection rows must be orthonormal.")
        p = projection * self.active[:, None, :]
        return TaskRequest(self.name, p @ self.A,
            (p @ self.bias[..., None]).squeeze(-1),
            (p @ self.desired[..., None]).squeeze(-1),
            p @ self.metric @ p.transpose(-1, -2),
            p.square().sum(-1) > 0, self.priority)


def task_objective(requests, transform, offset, accumulation_dtype=None):
    """Quadratic objective for physical acceleration = transform @ command + offset."""
    if not requests:
        raise ValueError("At least one task is required.")
    metric = torch.zeros_like(transform, dtype=accumulation_dtype)
    force = torch.zeros_like(offset, dtype=accumulation_dtype)
    for request in requests:
        if (request.A.shape[0] != offset.shape[0] or request.A.device != offset.device
                or request.A.dtype not in (offset.dtype, accumulation_dtype)):
            raise ValueError("Task requests must match the WBC batch, device and dtype.")
        h, f = request.natural_form(dtype=accumulation_dtype)
        metric = metric + h
        force = force + f
    transform = transform.to(metric.dtype)
    offset = offset.to(metric.dtype)
    transpose = transform.transpose(1, 2)
    return (transpose @ metric @ transform,
            (transpose @ (force + (metric @ offset[..., None]).squeeze(-1))[..., None]).squeeze(-1))
