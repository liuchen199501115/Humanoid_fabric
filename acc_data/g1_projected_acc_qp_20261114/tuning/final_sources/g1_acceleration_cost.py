"""Weight changes to COM acceleration using current kinematics, without tasks."""
import math
import torch


def preserve_com_acceleration(metric, com_jacobian, weight):
    """Add weight*||J_com*(a-a_reference)||^2 to an acceleration metric."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('COM acceleration preservation weight must be finite and nonnegative')
    if weight == 0:
        return metric
    if metric.ndim == 1:
        metric = torch.diag(metric).unsqueeze(0)
    return metric + weight * (com_jacobian.transpose(-1, -2) @ com_jacobian)
