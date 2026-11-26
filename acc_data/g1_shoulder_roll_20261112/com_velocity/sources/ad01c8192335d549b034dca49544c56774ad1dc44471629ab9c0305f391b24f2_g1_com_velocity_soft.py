"""Explicit velocity-error slack in the existing first-priority QP.

The constant-acceleration forecast is a local approximation, not a contact
preview. Existing physical constraints and subsequent priority locks remain.
"""
from dataclasses import replace
import math

import torch
from fabrics_sim.solver.admm_qp import QPData


class COMVelocitySoftConstraint:
    def __init__(self, backend, weight=1., horizon=.1, contraction=.8,
                 allowance=.02, onset=.2, width=.3):
        values=(weight,horizon,contraction,allowance,onset,width)
        if (not all(math.isfinite(x) for x in values) or weight<=0 or horizon<=0
                or not 0<contraction<1 or allowance<0 or onset<0 or width<=0):
            raise ValueError('Invalid COM velocity soft-constraint parameters')
        self.backend=backend
        self.weight,self.horizon,self.contraction=weight,horizon,contraction
        self.allowance,self.onset,self.width=allowance,onset,width
        self.ready=False
        self.last_augmented_problem=None

    def __getattr__(self, name):
        return getattr(self.backend,name)

    def prepare(self, request, velocity, reference_velocity, reference_acceleration, correction):
        self.error=velocity-reference_velocity
        phase=((self.error.norm(dim=-1)-self.onset)/self.width).clamp(0,1)
        self.activation=phase**3*(10-15*phase+6*phase**2)
        self.C=self.horizon*request.A
        self.offset=self.error+self.horizon*(request.bias-reference_acceleration)
        self.command_offset=self.offset-(self.C@correction[...,None]).squeeze(-1)
        self.bound=self.contraction*self.error.abs()+self.allowance
        self.penalty=(request.metric.diagonal(dim1=-2,dim2=-1)*self.weight
                      *self.activation[:,None]/self.horizon**2)
        self.slack=torch.zeros_like(self.error)
        self.ready=True
        self.last_augmented_problem=None

    def augment(self, qp):
        batch,n=qp.h.shape
        def pad(x):
            return torch.nn.functional.pad(x,(0,3))
        H=torch.nn.functional.pad(qp.H,(0,3,0,3))
        H[:,n:,n:]=torch.diag_embed(self.penalty.to(H.dtype).clamp_min(1e-9))
        C=torch.zeros(batch,3,n,device=qp.G.device,dtype=qp.G.dtype)
        C[:,:,:35]=self.C
        eye=torch.eye(3,device=C.device,dtype=C.dtype).expand(batch,-1,-1)
        extra=torch.cat((torch.cat((C,-eye),-1),torch.cat((-C,-eye),-1)),1)
        rhs=torch.cat((self.bound-self.command_offset,self.bound+self.command_offset),1)
        return QPData(H,pad(qp.h),torch.cat((pad(qp.G),extra),1),torch.cat((qp.b,rhs),1),
            torch.cat((qp.lb,torch.zeros_like(self.error)),1),
            torch.cat((qp.ub,torch.full_like(self.error,float('inf'))),1),
            None if qp.A_eq is None else pad(qp.A_eq),qp.b_eq)

    def solve(self, qp, warm_z=None, warm_lam_eq=None, workspace='tracking'):
        if not self.ready or workspace!='priority' or not bool((self.activation>0).any()):
            return self.backend.solve(qp,warm_z=warm_z,warm_lam_eq=warm_lam_eq,workspace=workspace)
        augmented=self.augment(qp)
        n=qp.h.shape[-1]
        if warm_z is not None:
            forecast=(self.C@warm_z[:,:35,None]).squeeze(-1)+self.command_offset
            warm_z=torch.cat((warm_z,(forecast.abs()-self.bound).clamp_min(0)),1)
        result=self.backend.solve(augmented,warm_z=warm_z,warm_lam_eq=warm_lam_eq,
                                  workspace='com_velocity_soft_priority')
        self.last_augmented_problem=augmented
        self.slack=result.z[:,n:].clone()
        violation=((augmented.G@result.z[...,None]).squeeze(-1)-augmented.b).clamp_min(0).amax(-1)
        ok=torch.isfinite(result.z).all(-1)&(violation<=1e-4)&(self.slack.min(-1).values>=-1e-6)
        status=ok if result.status_ok is None else result.status_ok&ok
        return replace(result,z=result.z[:,:n],status_ok=status,converged=result.converged&ok)

    def diagnostics(self, predicted_acc):
        forecast=(self.C@predicted_acc[...,None]).squeeze(-1)+self.offset
        return dict(com_soft_activation=self.activation,
                    com_soft_velocity_error=self.error,com_soft_forecast_error=forecast,
                    com_soft_bound=self.bound,com_soft_slack=self.slack,
                    com_soft_required_slack=(forecast.abs()-self.bound).clamp_min(0))
