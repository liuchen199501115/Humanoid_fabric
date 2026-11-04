"""Paper-style geometric losses, with flat ground and no terrain input."""
import torch
from .kinematics import PaperFK, normalize_quaternion


class PaperMotionLoss:
    def __init__(self, device):
        self.fk = PaperFK().to(device)

    def __call__(self, predicted):
        links, soles = self.fk(predicted, soles=True)
        return {
            'fk': (predicted[..., 30:].reshape(*predicted.shape[:-1], 23, 3)-links).square().mean(),
            'penetration': torch.relu(.005-soles[..., 2]).square().mean()
                + torch.relu(-links[..., 2]).square().mean(),
        }


def objective(model, batch, mean, std, alpha, velocity_weight, condition_noise,
              amp_dtype, geometry, weights):
    history, future, heading = [x.to(mean.device) for x in batch[:3]]
    target = (future-mean)/std
    condition = (history-mean)/std
    if condition_noise:
        perturbed = (condition + torch.randn_like(condition)*condition_noise)*std+mean
        perturbed[..., 3:7] = normalize_quaternion(perturbed[..., 3:7])
        perturbed[..., 30:] = geometry.fk(perturbed).flatten(-2)
        condition = (perturbed-mean)/std
    step = torch.randint(len(alpha), (len(target),), device=mean.device)
    a = alpha[step, None, None]
    noisy = a.sqrt()*target+(1-a).sqrt()*torch.randn_like(target)
    with torch.autocast(device_type=mean.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        output = model(noisy, step, condition, heading)
    physical = output.float()*std+mean
    q = normalize_quaternion(physical[..., 3:7])
    q = torch.where((q*future[..., 3:7]).sum(-1, keepdim=True)<0, -q, q)
    physical = torch.cat((physical[..., :3], q, physical[..., 7:]), -1)
    predicted = (physical-mean)/std
    reconstruction = (predicted-target).square().mean()
    # Frame increments include the history/future seam; normalized units.
    anchor = ((history-mean)/std)[:, -1:]
    velocity = (torch.diff(torch.cat((anchor,predicted),1),dim=1)
                -torch.diff(torch.cat((anchor,target),1),dim=1)).square().mean()
    terms = geometry(physical)
    loss = reconstruction+velocity_weight*velocity+sum(weights[k]*v for k,v in terms.items())
    model.last_geometry_losses = {k:float(v.detach()) for k,v in terms.items()}
    return loss, reconstruction.detach(), velocity.detach()
