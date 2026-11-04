"""Cosine forward process and DDIM with noise-consistent prefix inpainting."""
import math
import torch

def schedule(steps, device):
    if steps < 2:
        raise ValueError('At least two diffusion steps required')
    t = torch.linspace(0, 1, steps+1, device=device)
    cumulative = torch.cos((t+.008)/1.008*math.pi/2).square()
    cumulative = cumulative/cumulative[0]
    beta = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999)
    return torch.cumprod(1-beta, 0)


@torch.no_grad()
def sample(model, history, heading, alpha, steps=50, known=None, mask=None):
    """DDIM eta=0, from fresh Gaussian noise. No two-step quality claim."""
    if not 1 <= steps <= len(alpha):
        raise ValueError('sampling steps must be between 1 and diffusion steps')
    x = torch.randn(history.shape[0], model.horizon, history.shape[-1], device=history.device)
    if (known is None) != (mask is None):
        raise ValueError('known and mask must be provided together')
    if known is not None and not getattr(model, 'inpainting', False):
        raise ValueError('Prefix inpainting requires a model trained with inpainting')
    known_noise = torch.randn_like(x) if known is not None else None
    times = torch.linspace(len(alpha)-1, 0, steps, device=history.device).round().long()
    for i, t in enumerate(times):
        if known is not None:
            noisy_known = alpha[t].sqrt()*known+(1-alpha[t]).sqrt()*known_noise
            x = torch.where(mask.bool(), noisy_known, x)
            x0 = model(x, t.expand(len(x)), history, heading, known, mask)
            x0 = torch.where(mask.bool(), known, x0)
        else:
            x0 = model(x, t.expand(len(x)), history, heading)
        if i == len(times)-1:
            return x0
        eps = (x-alpha[t].sqrt()*x0)/(1-alpha[t]).sqrt()
        next_alpha = alpha[times[i+1]]
        x = next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*eps
