"""Damp only the next linearization seed; never admit an interpolated control."""
import copy
import numpy as np

SCALES={'nonlinear_collision_barrier':2.,'nonlinear_support_gap':.001,
 'nonlinear_node_clearance':.001,'nonlinear_minimum_clearance':.002,
 'nonlinear_base_force_residual':20.,'nonlinear_base_moment_residual':5.,
 'nonlinear_joint_residual':5.,'nonlinear_impact_residual':.5,
 'nonlinear_support_speed':.15,'nonlinear_contact_acceleration':2.,
 'nonlinear_wrench_cone_violation':.2,'nonlinear_impulse_cone_violation':.02,
 'nonlinear_integration_orientation':.02,'nonlinear_landing_error':.04}

def merit(audit):
 values=[]
 for key,scale in SCALES.items():
  x=audit.get(key,0.)
  if key in ('nonlinear_node_clearance','nonlinear_minimum_clearance'):x=max(0.,-x)
  values.append((max(0.,x)/scale)**2)
 return sum(values)

def blend(previous,current,alpha):
 result=copy.deepcopy(current)
 for key in ('states','controls','impulses'):
  result[key]=(1-alpha)*previous[key]+alpha*current[key]
 for stage,sides in result.get('point_forces',{}).items():
  for side,block in sides.items():
   old=previous.get('point_forces',{}).get(stage,{}).get(side)
   if old is not None and old['indices']==block['indices']:
    block['forces']=((1-alpha)*np.array(old['forces'])+alpha*np.array(block['forces'])).tolist()
 return result

def next_nominal(planner,previous,current,rate):
 if previous is None:return current
 candidates=[]
 for alpha in (1.,.5,.25,.125):
  candidate=current if alpha==1. else blend(previous,current,alpha)
  audit=planner.nonlinear_audit(candidate,rate);candidates.append((merit(audit),alpha,candidate))
 score,alpha,result=min(candidates,key=lambda row:row[0])
 planner.sqp_trust_scale=max(.125,min(1.,getattr(planner,'sqp_trust_scale',1.)*(.5 if alpha<.5 else 1.2)))
 planner.last_damping=alpha
 return result
