"""Local COM/orientation variation with both reference foot poses preserved."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

def skew(v):
 x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def pose_variations(com,leg,poses,velocities,times,upper=False):
 dimension=23 if upper else 6
 maps=[]
 for q in poses:
  T=np.zeros((35,dimension));T[:6,:6]=np.eye(6)
  if upper:T[18:,6:]=np.eye(17)
  for side in [0,1]:
   pos,R,J=leg.evaluate(q,side);base=np.block([[np.eye(3),-skew(pos-q[:3])],[np.zeros((3,3)),np.eye(3)]]);T[6+6*side:12+6*side]=-np.linalg.solve(J,base@T[:6])
  S=np.stack([com.momentum(q,T[:,j],return_state=True)[1] for j in range(dimension)],axis=1)
  change=np.eye(dimension);change[:3,:3]=np.linalg.inv(S[:,:3]);change[:3,3:]=-np.linalg.solve(S[:,:3],S[:,3:]);maps.append(T@change)
 T=np.asarray(maps);Td=CubicSpline(times,T)(times,1);P=[];Q=[];eps=1e-5
 for k,(q,v) in enumerate(zip(poses,velocities)):
  position=[];velocity=[]
  for j in range(dimension):
   direction=T[k,:,j];plus=q.copy();minus=q.copy();plus[:3]+=eps*direction[:3];minus[:3]-=eps*direction[:3];plus[7:]+=eps*direction[6:];minus[7:]-=eps*direction[6:]
   R=Rotation.from_quat(q[3:7]);plus[3:7]=(Rotation.from_rotvec(eps*direction[3:6])*R).as_quat();minus[3:7]=(Rotation.from_rotvec(-eps*direction[3:6])*R).as_quat()
   tangent_rate=Td[k,:,j].copy();tangent_rate[3:6]+=np.cross(direction[3:6],v[3:6])
   position.append((com.momentum(plus,v)-com.momentum(minus,v))/(2*eps)+com.momentum(q,tangent_rate));velocity.append(com.momentum(q,direction))
  P.append(np.stack(position,axis=1));Q.append(np.stack(velocity,axis=1))
 P=np.asarray(P);Q=np.asarray(Q)
 return T,Td,P,Q,CubicSpline(times,P)(times,1),CubicSpline(times,Q)(times,1)
