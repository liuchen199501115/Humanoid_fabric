"""Bound each swing's joint-space correction with one constant blend.

This preserves source timing, source/root poses and correction endpoints.
The measured contact/clearance and original-source fidelity gates still decide
whether the resulting trajectory actually tracks.
"""
import math
import torch

def bound_reference_correction(control,source,events,tails,*,first_step=8,joint_rmse_limit=.08,first_step_limit=None):
    if control.shape!=source.shape or not math.isfinite(joint_rmse_limit) or joint_rmse_limit<=0:
        raise ValueError('Invalid reference correction bound')
    result=control.clone();reports=[]
    ends={r['step']:r.get('end_frame') for r in tails}
    for index,event in enumerate(events):
        number=index+1
        if number<first_step or event['kind']!='swing':continue
        lo=event['takeoff_frame'];hi=ends.get(number) or event['end_frame']
        delta=control[lo:hi+1,7:]-source[lo:hi+1,7:]
        maximum=float(delta.square().mean(-1).sqrt().max())
        limit=first_step_limit if number==first_step and first_step_limit is not None else joint_rmse_limit
        if not math.isfinite(limit) or limit<=0:raise ValueError("Invalid first swing correction bound")
        scale=min(1.,limit/max(maximum,1e-12))
        if scale<1:result[lo:hi+1,7:]=source[lo:hi+1,7:]+scale*delta
        reports.append(dict(step=number,first_frame=lo,last_frame=hi,scale=scale,peak_correction_before=maximum,peak_correction_after=maximum*scale))
    return result,reports
