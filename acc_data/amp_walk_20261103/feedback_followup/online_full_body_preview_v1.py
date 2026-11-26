"""Optional preview feedback, always reprojected through the existing physical WBC."""
import json,time
import numpy as np
import torch
from fabrics_sim.fabrics.task_request import TaskRequest
from full_body_preview import difference,integrate

class OnlineFullBodyPreview:
 def __init__(self,planner,path,period=.02):
  self.planner=planner;self.path=path;self.period=period;self.last_time=-1.;self.last_mode=None;self.plan=None;self.records=[]
 def solve(self,baseline_solver,owner,q,v,now,frame,rate,**kwargs):
  baseline=baseline_solver(**kwargs)
  if not bool(baseline.validated.all()):return baseline
  mode=int(kwargs['contact_mode'].item());qn=q.detach().cpu().numpy()[0].astype(float);vn=v.detach().cpu().numpy()[0].astype(float)
  if now-self.last_time>=self.period-1e-8 or mode!=self.last_mode:
   problem=owner.last_priority_problems[0];first={name:getattr(problem,name).detach().cpu().numpy()[0].astype(float) for name in ['A_eq','b_eq','G','b','lb','ub']};first['mode']=mode;started=time.monotonic();result=self.planner.solve(qn,vn,frame,rate,first);self.last_time=now;self.last_mode=mode
   self.records.append(dict(time=now,frame=frame,mode=mode,success=result['success'],status=result['status'],linear_violation=result['linear_violation'],tube_violation=result['tube_violation'],solve_seconds=time.monotonic()-started,objective=result['objective']))
   if result['success'] and not any(r.get('plan_saved',False) for r in self.records):
    np.savez_compressed(self.path.with_suffix('.first_plan.npz'),q=qn,v=vn,time=now,frame=frame,rate=rate,**result)
    self.records[-1]['plan_saved']=True
   self.plan=(qn,vn,result['first_control'][:35]) if result['success'] else None
   self.path.write_text(json.dumps(self.records,indent=2)+'\n')
  if self.plan is None:return baseline
  q0,v0,a0=self.plan;elapsed=now-self.last_time;predicted_q=integrate(q0,elapsed*v0+.5*elapsed*elapsed*a0);predicted_v=v0+elapsed*a0
  target=a0-20*difference(qn,predicted_q)-8*(vn-predicted_v)
  desired=q.new_tensor(target)[None];identity=torch.eye(35,device=q.device,dtype=q.dtype)[None]
  task=TaskRequest('full_body_preview',identity,torch.zeros_like(desired),desired,identity*10000.,torch.ones_like(desired,dtype=torch.bool),priority='high')
  modified=dict(kwargs);modified.update(task_requests=list(kwargs['task_requests'])+[task],priority_tasks=('full_body_preview',),priority_requests=None,priority_levels=None,priority_rows={},priority_projections=None)
  result=baseline_solver(**modified)
  if bool(result.validated.all()):
   self.records[-1]['applied']=True
   self.records[-1]['latest_acceleration_projection_max_error']=float((result.predicted_acc-desired).abs().max())
  else:self.records[-1]['reprojection_failed']=True
  self.path.write_text(json.dumps(self.records,indent=2)+'\n')
  return result
