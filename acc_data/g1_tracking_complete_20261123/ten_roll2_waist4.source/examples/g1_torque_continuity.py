"""Soft effort continuity in every physical QP stage, without filtering output."""
import math
import torch


class TorqueContinuitySolver:
    def __init__(self, backend):
        self.backend = backend
        self.weight = 0.
        self.previous = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def __setattr__(self, name, value):
        # Runtime tube, proximal and scaling settings belong to the wrapped
        # backend. Shadowing them here silently changes even a zero-weight run.
        if name in ('backend', 'weight', 'previous'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.backend, name, value)

    def update(self, previous, weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Torque continuity weight must be finite and nonnegative')
        if previous.ndim != 2 or previous.shape[1] != 29 or not torch.isfinite(previous).all():
            raise ValueError('Previous validated G1 torque must be finite [B,29]')
        self.previous = previous.detach().clone()
        self.weight = float(weight)

    def solve(self, qp, **kwargs):
        if self.weight or hasattr(qp, '_torque_continuity_base'):
            if qp.H.shape[-2:] != (76, 76) or qp.h.shape != qp.H.shape[:2]:
                raise ValueError('Torque continuity requires a G1 acceleration/wrench/effort QP')
            if not hasattr(qp, '_torque_continuity_base'):
                qp._torque_continuity_base = (qp.H, qp.h)
            base_H, base_h = qp._torque_continuity_base
            qp.H, qp.h = base_H.clone(), base_h.clone()
            if self.weight:
                if self.previous is None or len(self.previous) != len(qp.h):
                    raise ValueError('A matching previous validated torque is required')
                qp.H[:, 47:76, 47:76] += self.weight*torch.eye(29, device=qp.H.device, dtype=qp.H.dtype)
                qp.h[:, 47:76] -= self.weight*self.previous.to(qp.h)
        # Mutate this problem's objective so diagnostics retain the exact solve.
        # Dynamics/contact/limits, warm starts, and solver status are unchanged.
        return self.backend.solve(qp, **kwargs)
