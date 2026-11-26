"""Move orientation objective weight from torso to the same-reference pelvis.

The waist remains governed by source joint posture. No pose/velocity/acceleration
reference, contact row or physical limit is changed.
"""
import math

def replace_torso_weight(requests,blend,*,pelvis_scale=1.):
    if not math.isfinite(blend) or not 0<=blend<=1:
        raise ValueError('Blend must be finite and in [0,1]')
    if not math.isfinite(pelvis_scale) or pelvis_scale<=0:
        raise ValueError('Pelvis metric scale must be finite and positive')
    selected={r.name:r for r in requests}
    if 'body_orientation' not in selected or 'source_pelvis_orientation' not in selected:
        raise ValueError('Both torso and pelvis requests are required')
    body=selected['body_orientation'];pelvis=selected['source_pelvis_orientation']
    metric=body.metric.clone()
    if metric.shape!=pelvis.metric.shape:
        raise ValueError('Expected equal orientation metric shapes')
    pelvis.metric=metric*(blend*pelvis_scale)
    body.metric=metric*(1-blend)
