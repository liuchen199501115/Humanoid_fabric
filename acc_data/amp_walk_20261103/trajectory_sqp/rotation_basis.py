"""Exact first variation of normalized quaternion spline q/omega/alpha."""
import numpy as np

def quaternion_kinematics(u,du,ddu):
 norm=np.linalg.norm(u,axis=-1,keepdims=True);q=u/norm
 nv=np.sum(q*du,axis=-1,keepdims=True);qv=(du-q*nv)/norm
 na=(np.sum(du*du+u*ddu,axis=-1,keepdims=True)-nv*nv)/norm;qa=(ddu-q*na-2*qv*nv)/norm
 def angular(d):return 2*(q[...,3:]*d[...,:3]-d[...,3:]*q[...,:3]+np.cross(q[...,:3],d[...,:3]))
 return q,angular(qv),angular(qa)

def rotation_variations(coeff,B,D,E,near):
 qi=B@coeff[:,3:7];vi=D@coeff[:,3:7];ai=E@coeff[:,3:7]
 index=np.array([(k,j) for k,row in enumerate(near) for j in row]);ks,js=index.T
 u,du,ddu=qi[ks],vi[ks],ai[ks];q=quaternion_kinematics(u,du,ddu)[0]
 maps=np.empty((len(index),3,3,3));eps=1e-6
 for axis in range(3):
  e=np.zeros(3);e[axis]=1.;c=coeff[js,3:7]
  delta=.5*np.c_[c[:,3:]*e+np.cross(e,c[:,:3]),-c[:,axis]]
  pu,pv,pa=quaternion_kinematics(u+eps*B[ks,js,None]*delta,du+eps*D[ks,js,None]*delta,ddu+eps*E[ks,js,None]*delta)
  mu,mv,ma=quaternion_kinematics(u-eps*B[ks,js,None]*delta,du-eps*D[ks,js,None]*delta,ddu-eps*E[ks,js,None]*delta)
  dq=(pu-mu)/(2*eps)
  maps[:,0,:,axis]=2*(q[:,3:]*dq[:,:3]-dq[:,3:]*q[:,:3]+np.cross(q[:,:3],dq[:,:3]))
  maps[:,1,:,axis]=(pv-mv)/(2*eps);maps[:,2,:,axis]=(pa-ma)/(2*eps)
 out=[{} for _ in near]
 for row,(k,j) in enumerate(index):out[k][j]=maps[row]
 return out
