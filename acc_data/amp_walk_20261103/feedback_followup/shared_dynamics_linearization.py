"""Reuse the same central-difference model states for dynamics and contacts."""
import numpy as np
from generalized_dynamics_linearization import perturb,residual

def derivatives(model,q,v,a,w,epsilon=1e-5):
 Dq=np.zeros((35,35));Dv=Dq.copy();position_states=[];velocity_states=[]
 for j in range(35):
  delta=np.zeros(35);delta[j]=epsilon
  qp=model.evaluate(perturb(q,delta),v,kinematic_bias=True)
  qm=model.evaluate(perturb(q,-delta),v,kinematic_bias=True)
  vp=model.evaluate(q,v+delta,kinematic_bias=True)
  vm=model.evaluate(q,v-delta,kinematic_bias=True)
  Dq[:,j]=(residual(qp,a,w)-residual(qm,a,w))/(2*epsilon)
  Dv[:,j]=(vp['bias']-vm['bias'])/(2*epsilon)
  position_states.append([qp,qm]);velocity_states.append([vp,vm])
 return Dq,Dv,position_states,velocity_states
