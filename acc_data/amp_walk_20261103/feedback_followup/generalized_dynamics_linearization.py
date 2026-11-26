"""Finite differences of the full inverse-dynamics residual in world tangents."""
import numpy as np
from scipy.spatial.transform import Rotation

def perturb(q,delta):
 x=q.copy();x[:3]+=delta[:3];x[7:]+=delta[6:];x[3:7]=(Rotation.from_rotvec(delta[3:6])*Rotation.from_quat(q[3:7])).as_quat();return x

def residual(state,a,w):
 return state['mass']@a+state['bias']-sum(state['jacobian'][j].T@w[6*j:6*j+6] for j in [0,1])

def derivatives(model,q,v,a,w,epsilon=1e-5):
 Dq=np.zeros((35,35));Dv=Dq.copy()
 for j in range(35):
  delta=np.zeros(35);delta[j]=epsilon
  plus=model.evaluate(perturb(q,delta),v);minus=model.evaluate(perturb(q,-delta),v);Dq[:,j]=(residual(plus,a,w)-residual(minus,a,w))/(2*epsilon)
  plus=model.evaluate(q,v+delta);minus=model.evaluate(q,v-delta);Dv[:,j]=(plus['bias']-minus['bias'])/(2*epsilon)
 return Dq,Dv
