"""Physical-unit soft losses; no claim of contact/dynamic feasibility."""
import torch
from .kinematics import G1FeetFK


class MotionGeometryLoss:
    def __init__(self, names, limits, fps, device):
        self.fk = G1FeetFK(names).to(device)
        self.limits = torch.as_tensor(limits, device=device)
        self.fps = fps

    def __call__(self, predicted, reference, history, contacts, known_mask=None):
        p = torch.cat((history, predicted), 1)
        r = torch.cat((history, reference), 1)
        v, vr = torch.diff(p, dim=1)*self.fps, torch.diff(r, dim=1)*self.fps
        # Rotation uses the continuous 6D representation; joint/root units retained.
        velocity = (v[:, 1:]-vr[:, 1:]).square().mean()
        acceleration = ((torch.diff(v, dim=1)-torch.diff(vr, dim=1))/100.).square().mean()*self.fps**2
        if known_mask is None:
            boundary = (v[:, 1]-vr[:, 1]).square().mean()
        else:
            previous_known = torch.cat((torch.ones_like(known_mask[:, :1]), known_mask[:, :-1]), 1)
            seam = previous_known*(1-known_mask)
            boundary = ((v[:, 1:]-vr[:, 1:]).square()*seam).sum()/(seam.sum()*p.shape[-1]).clamp_min(1)
        feet = self.fk(p)
        ref_feet = self.fk(r)
        valid = (contacts[:, 1:]*contacts[:, :-1])[..., None]
        # Teacher contacts cannot be switched off by the prediction to evade loss.
        planar = torch.diff(feet[..., :2], dim=1)*self.fps
        skating = (planar.square().sum(-1)*valid).sum()/(valid.sum()*4).clamp_min(1)
        height = feet[:, 2:, ..., 2]-.005
        active = contacts[:, 2:, :, None]
        # Lowest sole sphere can touch on heel/toe without forcing a flat foot.
        contact = (height.min(-1).values.square()*active.squeeze(-1)).sum()/active.sum().clamp_min(1)
        penetration = torch.relu(-height).square().mean()
        fk_match = (feet[:, 2:]-ref_feet[:, 2:]).square().mean()
        q = predicted[..., 9:]
        limit = (torch.relu(self.limits[:, 0]-q).square()+torch.relu(q-self.limits[:, 1]).square()).mean()
        return dict(velocity=velocity, acceleration=acceleration, boundary=boundary,
                    skating=skating, contact=contact, penetration=penetration, fk=fk_match, limits=limit)
