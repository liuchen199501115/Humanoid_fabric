"""Conditional x0-predicting temporal Transformer and cosine diffusion schedule."""
import math
import torch
from torch import nn


class MotionDenoiser(nn.Module):
    def __init__(self, dim=38, horizon=25, width=128, layers=4, heads=4, ffn_dim=None, inpainting=False):
        super().__init__()
        if width % heads or width % 2 or horizon < 1:
            raise ValueError('width must be even and divisible by heads; horizon must be positive')
        self.inpainting = inpainting
        if inpainting:
            self.known_input = nn.Linear(dim+1, width)
        self.width = width
        self.horizon = horizon
        self.input = nn.Linear(dim, width)
        self.heading = nn.Linear(2, width)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(1, horizon+3, width)*.02)
        layer = nn.TransformerEncoderLayer(width, heads, (ffn_dim if ffn_dim is not None else width*4), dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, noisy, step, history, heading, known=None, mask=None):
        freq = torch.exp(-math.log(10000)*torch.arange(self.width//2, device=noisy.device)/(self.width//2))
        phase = step[:, None]*freq[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        tokens = torch.cat((self.heading(heading)[:, None], self.input(history), self.input(noisy)), 1)
        if self.inpainting:
            if known is None:
                known, mask = torch.zeros_like(noisy), torch.zeros_like(noisy[..., :1])
            condition = self.known_input(torch.cat((known*mask, mask.to(noisy.dtype)), -1))
            tokens = tokens + torch.cat((torch.zeros_like(tokens[:, :3]), condition), 1)
        encoded = self.encoder(tokens+self.position+time[:, None])
        return self.output(encoded[:, 3:])



# Backwards-compatible imports for existing callers.
from .diffusion import schedule, sample
