"""Soft effort continuity in every physical QP stage, without filtering output."""
import math
import torch


def stance_torque_weights(contact_mode, torque):
    """Select the six leg motors of each QP support foot; flight stays free."""
    if (torque.ndim!=2 or torque.shape[1]!=29 or contact_mode.shape!=(len(torque),)
            or contact_mode.dtype.is_floating_point
            or bool(((contact_mode<0)|(contact_mode>3)).any())):
        raise ValueError('Expected G1 torque [B,29] and integer support modes [B] in 0..3')
    weights=torch.zeros_like(torque)
    for side in (0,1):
        weights[:,6*side:6*side+6]=((contact_mode&(1<<side))!=0).to(weights)[:,None]
    return weights


class TorqueContinuitySolver:
    def __init__(self, backend):
        self.backend = backend
        self.weight = 0.
        self.previous = None
        self.mask = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def __setattr__(self, name, value):
        # Runtime tube, proximal and scaling settings belong to the wrapped
        # backend. Shadowing them here silently changes even a zero-weight run.
        if name in ('backend', 'weight', 'previous', 'mask'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.backend, name, value)

    def update(self, previous, weight, mask=None):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Torque continuity weight must be finite and nonnegative')
        if previous.ndim != 2 or previous.shape[1] != 29 or not torch.isfinite(previous).all():
            raise ValueError('Previous validated G1 torque must be finite [B,29]')
        if mask is not None and (mask.shape!=previous.shape or not torch.isfinite(mask).all()
                or bool(((mask<0)|(mask>1)).any())):
            raise ValueError('Torque continuity weights must be finite [B,29] in [0,1]')
        self.previous = previous.detach().clone()
        self.weight = float(weight)
        self.mask = None if mask is None else mask.detach().clone()

    def solve(self, qp, **kwargs):
        if self.weight or hasattr(qp, '_torque_continuity_base'):
            n = qp.H.shape[-1]
            if not 76 <= n <= 105 or qp.H.shape[-2:] != (n, n) or qp.h.shape != qp.H.shape[:2]:
                raise ValueError('Torque continuity requires a G1 acceleration/wrench/effort QP')
            # Observed passive joint-stop reactions follow the 76 physical
            # variables. Smooth only commanded motor torques, never reactions.
            if not hasattr(qp, '_torque_continuity_base'):
                qp._torque_continuity_base = (qp.H, qp.h)
            base_H, base_h = qp._torque_continuity_base
            qp.H, qp.h = base_H.clone(), base_h.clone()
            if self.weight:
                if self.previous is None or len(self.previous) != len(qp.h):
                    raise ValueError('A matching previous validated torque is required')
                if self.mask is None:
                    qp.H[:, 47:76, 47:76] += self.weight*torch.eye(29, device=qp.H.device, dtype=qp.H.dtype)
                    qp.h[:, 47:76] -= self.weight*self.previous.to(qp.h)
                else:
                    weights=self.weight*self.mask.to(qp.h)
                    qp.H[:, 47:76, 47:76] += torch.diag_embed(weights)
                    qp.h[:, 47:76] -= weights*self.previous.to(qp.h)
        # Mutate this problem's objective so diagnostics retain the exact solve.
        # Dynamics/contact/limits, warm starts, and solver status are unchanged.
        return self.backend.solve(qp, **kwargs)
