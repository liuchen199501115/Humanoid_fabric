"""One instantaneous full-configuration objective for a unified q/v/a reference."""
import math
import torch
from fabrics_sim.fabrics.task_request import TaskRequest


def configuration_tracking_request(q,v,reference,requests,angular_weight=1.):
    if not math.isfinite(angular_weight) or angular_weight<=0:
        raise ValueError('Angular weight must be positive and finite')
    by_name={r.name:r for r in requests}
    posture=by_name['joint_posture'];pelvis=by_name['source_pelvis_orientation'];com=by_name['center_of_mass']
    desired=reference.a.clone()
    feedback=(-(q[:,:3]-reference.q[:,:3])*q.new_tensor([35.,35.,25.])
              -(v[:,:3]-reference.v[:,:3])*q.new_tensor([12.,12.,10.]))
    desired[:,:3]+=feedback.clamp(-8.,8.)
    # Pelvis and posture requests already use this same unified reference,
    # including world angular rates and the established bounded feedback.
    desired[:,3:6]=pelvis.desired
    desired[:,6:]=posture.desired
    metric=q.new_zeros((len(q),35,35))
    metric[:,:3,:3]=com.metric
    metric[:,3:6,3:6]=angular_weight*pelvis.metric
    metric[:,6:,6:]=posture.metric
    return TaskRequest('configuration_tracking',torch.eye(35,device=q.device,dtype=q.dtype)[None].expand(len(q),-1,-1),
                       torch.zeros_like(v),desired,metric,torch.ones_like(v,dtype=torch.bool),'high')
