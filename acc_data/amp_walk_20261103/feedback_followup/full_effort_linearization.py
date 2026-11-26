"""Single-support inverse-dynamics variations along a constrained pose curve."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

def inverse_output(state,foot):
 J=state['jacobian'][foot];f=state['generalized_force'];w=np.linalg.solve(J[:,:6].T,f[:6]);return np.r_[w,f[6:]-J[:,6:].T@w]
def perturb(q,d):
 out=q.copy();out[:3]+=d[:3];out[7:]+=d[6:];out[3:7]=(Rotation.from_rotvec(d[3:6])*Rotation.from_quat(q[3:7])).as_quat();return out

def effort_variations(model,qs,vs,accelerations,T,Td,times,indices,feet):
 Tdd=CubicSpline(times,T)(times,2);result={};eps=1e-5
 for k,foot in zip(indices,feet):
  q,v,a=qs[k],vs[k],accelerations[k];state=model.evaluate(q,v,a);raw=inverse_output(state,foot);P=[];V=[];A=[];velocity_map=Td[k].copy();velocity_map[3:6]+=np.cross(T[k,3:6].T,v[3:6]).T;acceleration_map=Tdd[k].copy();acceleration_map[3:6]+=np.cross(Td[k,3:6].T,v[3:6]).T+np.cross(T[k,3:6].T,a[3:6]).T;acceleration_velocity=2*Td[k].copy();acceleration_velocity[3:6]+=np.cross(T[k,3:6].T,v[3:6]).T
  for j in range(T.shape[-1]):
   plus=model.evaluate(perturb(q,eps*T[k,:,j]),v+eps*velocity_map[:,j],a+eps*acceleration_map[:,j]);minus=model.evaluate(perturb(q,-eps*T[k,:,j]),v-eps*velocity_map[:,j],a-eps*acceleration_map[:,j]);P.append((inverse_output(plus,foot)-inverse_output(minus,foot))/(2*eps))
   plus=model.evaluate(q,v+eps*T[k,:,j],a+eps*acceleration_velocity[:,j]);minus=model.evaluate(q,v-eps*T[k,:,j],a-eps*acceleration_velocity[:,j]);V.append((inverse_output(plus,foot)-inverse_output(minus,foot))/(2*eps))
   derivative=dict(state);derivative['generalized_force']=state['mass']@T[k,:,j];A.append(inverse_output(derivative,foot))
  result[k]=(raw,np.stack(P,1),np.stack(V,1),np.stack(A,1))
 return result
