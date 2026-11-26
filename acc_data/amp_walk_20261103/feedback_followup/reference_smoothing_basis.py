"""Apply the runtime interior stride smoother to a linear reference basis.

The finite support must avoid fractional Greville nodes: runtime PCHIP at such
nodes is nonlinear in its control poses, so an exact linear basis cannot be
claimed there. The turn-local reference satisfies this restriction.
"""
import numpy as np
from scipy.interpolate import BSpline

def runtime_smoothing_basis(raw_spline,frame_count,stride=6):
    if stride<2 or frame_count<4*stride:
        raise ValueError('Expected at least four stride intervals')
    start,end=raw_spline.t[3],raw_spline.t[-4]
    if start<0 or end>frame_count-1:
        raise ValueError('Raw basis must lie inside the source clip')
    nodes=np.unique(np.r_[np.arange(0,frame_count,stride),frame_count-1])
    knots=np.r_[np.full(4,nodes[0]),nodes[2:-2],np.full(4,nodes[-1])]
    greville=np.array([np.mean(knots[i+1:i+4]) for i in range(len(nodes))])
    inside=(greville>=start)&(greville<=end)
    if np.any(abs(greville[inside]-np.round(greville[inside]))>1e-10):
        raise ValueError('Fractional runtime nodes require a nonlinear PCHIP model')
    coefficients=np.zeros((len(nodes),raw_spline.c.shape[1]))
    coefficients[inside]=raw_spline(greville[inside])
    return BSpline(knots,coefficients,3)
