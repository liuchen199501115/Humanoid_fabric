"""Bounded response feedback for tracking objectives, using completed intervals."""
import math
import torch


class ResponseFeedback:
    """Estimate d = (v[k]-v[k-1])/dt - a_pred[k-1].

    d_hat <- (1-alpha)*d_hat + alpha*clip(d), alpha=1-exp(-dt/tau).
    Tracking requests use a_target <- a_target - gain*A*d_hat.
    Contact changes reset the estimate. Contact-force jumps suspend learning.
    Physical dynamics, collision/contact rows, references and audits are untouched.
    This empirical compensation is not a corrected physical dynamics model.
    """
    TRACKING = frozenset(('center_of_mass', 'body_orientation',
        'source_pelvis_orientation', 'joint_posture', 'feet_pose', 'hand_pose',
        'centroidal_momentum', 'root_position'))

    def __init__(self, gain=.25, time_constant=.04, settle_time=.02,
                 linear_limit=3., angular_limit=20., joint_limit=0., force_jump=200.):
        values=(gain,time_constant,settle_time,linear_limit,angular_limit,joint_limit,force_jump)
        if (not all(math.isfinite(x) for x in values) or not 0<=gain<=1
                or time_constant<=0 or settle_time<0 or min(linear_limit,angular_limit,joint_limit)<0
                or force_jump<=0):
            raise ValueError('Invalid response feedback parameters')
        self.gain,self.tau,self.settle=gain,time_constant,settle_time
        self.limits=(linear_limit,angular_limit,joint_limit)
        self.force_jump=force_jump
        self.seed=None
        self.estimate=None
        self.age=None

    def observe(self, velocity, mode, forces, dt):
        if (velocity.ndim!=2 or velocity.shape[1]!=35 or mode.shape!=(len(velocity),)
                or forces.shape!=(len(velocity),2) or not math.isfinite(dt) or dt<=0
                or not torch.isfinite(velocity).all() or not torch.isfinite(forces).all()):
            raise ValueError('Expected finite velocity [B,35], modes [B], forces [B,2] and positive dt')
        if self.estimate is None:
            self.estimate=torch.zeros_like(velocity)
            self.age=velocity.new_zeros(len(velocity))
        if self.estimate.shape!=velocity.shape:
            raise ValueError('Observer batch shape cannot change')
        raw=torch.zeros_like(velocity)
        ready=torch.zeros_like(mode,dtype=torch.bool)
        if self.seed is not None:
            old_v,prediction,old_mode,old_forces=self.seed
            same=mode==old_mode
            self.age=torch.where(same,self.age+dt,torch.zeros_like(self.age))
            self.estimate=torch.where(same[:,None],self.estimate,torch.zeros_like(self.estimate))
            raw=(velocity-old_v)/dt-prediction
            ready=same & (mode!=0) & (self.age>=self.settle) & ((forces-old_forces).abs().amax(-1)<=self.force_jump)
            cap=velocity.new_tensor([self.limits[0]]*3+[self.limits[1]]*3+[self.limits[2]]*29)
            target=torch.where(ready[:,None],raw.clamp(min=-cap,max=cap),torch.zeros_like(raw))
            self.estimate+=(1-math.exp(-dt/self.tau))*(target-self.estimate)
        return dict(response_residual=raw.clone(),response_estimate=self.estimate.clone(),response_ready=ready)

    def apply(self, requests, activation=None):
        if self.estimate is None:
            raise RuntimeError('Observe the current state before applying response feedback')
        if activation is None:
            activation=self.estimate.new_ones(len(self.estimate))
        if (activation.shape!=(len(self.estimate),) or not torch.isfinite(activation).all()
                or bool(((activation<0)|(activation>1)).any())):
            raise ValueError('Response activation must be finite [B] in [0,1]')
        for request in requests:
            if request.name in self.TRACKING:
                correction=(request.A@self.estimate.to(request.A)[...,None]).squeeze(-1)
                request.desired=request.desired-self.gain*activation[:,None]*correction

    def commit(self, velocity, prediction, mode, forces):
        if prediction.shape!=velocity.shape or not torch.isfinite(prediction).all():
            raise ValueError('A finite, validated prediction is required')
        self.seed=tuple(x.detach().clone() for x in (velocity,prediction,mode,forces))


def stopping_distance_request(position, velocity, lower, upper, indices, *, budget=25.,
                              weight=80., margin=.03, acceleration_limit=200.):
    """Soft braking when v^2/(2*remaining_distance) exceeds the chosen budget.

    The budget is an empirical trigger, not a guaranteed available deceleration.
    The physical QP still determines achievable braking and may reject the step.
    A C2 metric activation avoids switching the entire posture task by a clock.
    """
    from fabrics_sim.fabrics.task_request import TaskRequest
    if (position.ndim!=2 or position.shape[-1]!=29 or any(x.shape!=position.shape for x in (velocity,lower,upper))
            or not all(bool(torch.isfinite(x).all()) for x in (position,velocity,lower,upper))
            or bool((upper<=lower).any()) or not indices or len(set(indices))!=len(indices)
            or any(i<0 or i>=29 for i in indices)
            or not all(math.isfinite(x) and x>0 for x in (budget,weight,margin,acceleration_limit))):
        raise ValueError('Invalid stopping-distance request')
    room=torch.where(velocity>=0,upper-position,position-lower)-margin
    demand=velocity.square()/(2*room.clamp_min(.001))
    fraction=((demand-budget)/budget).clamp(0,1)
    gate=fraction**3*(10-15*fraction+6*fraction**2)
    mask=torch.zeros_like(position);mask[:,list(indices)]=1.
    gate*=mask
    desired=-velocity.sign()*demand.clamp(max=acceleration_limit)
    A=position.new_zeros(len(position),29,35)
    A[:,:,6:]=torch.eye(29,device=position.device,dtype=position.dtype)
    return TaskRequest('state_joint_stopping',A,torch.zeros_like(position),desired,
        torch.diag_embed(weight*gate),gate>0,'high'),demand


def root_recovery_request(q,v,reference_q,reference_v,reference_a,*,metric=500.,onset=.1,
                          stiffness=100.,damping=20.,feedback_limit=30.):
    """A root PD task faded in by measured position error, not elapsed time."""
    from fabrics_sim.fabrics.task_request import TaskRequest
    values=(metric,onset,stiffness,damping,feedback_limit)
    if (not all(math.isfinite(x) and x>0 for x in values) or q.ndim!=2 or q.shape[1]!=36
            or reference_q.shape!=q.shape or any(x.shape!=(len(q),35) for x in (v,reference_v,reference_a))):
        raise ValueError('Invalid state root-recovery inputs')
    error=q[:,:3]-reference_q[:,:3]
    s=((error.norm(dim=-1)-onset)/.05).clamp(0,1)
    activation=s**3*(10-15*s+6*s*s)
    desired=reference_a[:,:3]+(-stiffness*error-damping*(v[:,:3]-reference_v[:,:3])).clamp(-feedback_limit,feedback_limit)
    A=q.new_zeros(len(q),3,35);A[:,:,:3]=torch.eye(3,device=q.device,dtype=q.dtype)
    W=torch.diag_embed((metric*activation[:,None]).expand(-1,3))
    return TaskRequest('state_root_position',A,torch.zeros_like(desired),desired,W,
        (activation[:,None]>0).expand(-1,3)),activation
