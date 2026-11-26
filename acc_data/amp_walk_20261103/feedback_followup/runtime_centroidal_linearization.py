"""Differentiate raw fixed-foot IK through the executed quaternion/spline curve.

Applying a smoother to the coordinate basis and then multiplying by a pose
Jacobian is not equivalent to smoothing the reconstructed joint trajectory.
This module composes those operations in their actual execution order.
"""
import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.transform import Rotation

def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def rotation_left_jacobian(v):
    theta=np.linalg.norm(v);K=skew(v)
    if theta<1e-6:return np.eye(3)+.5*K+K@K/6
    return np.eye(3)+(1-np.cos(theta))/theta**2*K+(theta-np.sin(theta))/theta**3*(K@K)

def pose_map(com,leg,q,orientation_offset):
    T=np.zeros((35,23));T[:6,:6]=np.eye(6);T[18:,6:]=np.eye(17)
    for side in [0,1]:
        pos,_,J=leg.evaluate(q,side)
        base=np.block([[np.eye(3),-skew(pos-q[:3])],[np.zeros((3,3)),np.eye(3)]])
        T[6+6*side:12+6*side]=-np.linalg.solve(J,base@T[:6])
    J,_=com.matrices(q);S=J@T;change=np.eye(23)
    change[:3,:3]=np.linalg.inv(S[:,:3]);change[:3,3:]=-np.linalg.solve(S[:,:3],S[:,3:])
    T=T@change;T[:,3:6]=T[:,3:6]@rotation_left_jacobian(orientation_offset)
    return T

def angular(q,d):
    # q is xyzw; d may contain any number of column vectors.
    return 2*(q[3]*d[:3]-q[:3,None]*d[3]+np.cross(q[:3],d[:3].T).T)

def curve_maps(com,leg,curve,raw_basis,coefficients,frames,rate=240.):
    K=raw_basis.c.shape[1];nc=23*K;knots=curve.smooth.t
    nodes=np.array([np.mean(knots[i+1:i+4]) for i in range(len(curve.smooth.c))])
    start,end=raw_basis.t[3],raw_basis.t[-4]
    raw_maps=np.zeros((len(nodes),36,nc))
    for i,f in enumerate(nodes):
        if not start<=f<=end:continue
        if abs(f-round(f))>1e-10:raise ValueError('Fractional PCHIP coefficient nodes need their own derivative')
        q=curve.raw(float(f)).copy();q[3:7]/=np.linalg.norm(q[3:7])
        basis=raw_basis(f);T=pose_map(com,leg,q,basis@coefficients[:,3:6]);D=np.zeros((36,23))
        D[:3]=T[:3];D[7:]=T[6:]
        D[3:6]=.5*(q[6]*T[3:6]+np.cross(T[3:6].T,q[3:6]).T)
        D[6]=-.5*q[3:6]@T[3:6]
        raw_maps[i]=np.kron(basis,D)
    mapped=BSpline(knots,raw_maps,3)
    values=curve.smooth(frames);velocities=curve.smooth(frames,1)*rate
    delta=mapped(frames);delta_v=mapped(frames,1)*rate
    pose_maps=[];velocity_maps=[];poses=[];speeds=[]
    for value,velocity,D,E in zip(values,velocities,delta,delta_v):
        norm=np.linalg.norm(value[3:7]);q=value[3:7]/norm;u=velocity[3:7];nv=q@u
        dq=(D[3:7]-q[:,None]*(q@D[3:7]))/norm
        qv=(u-q*nv)/norm;dn=q@D[3:7]
        dqv=(E[3:7]-dq*nv-q[:,None]*(u@dq+q@E[3:7]))/norm-qv[:,None]*dn/norm
        pose_maps.append(np.vstack((D[:3],angular(q,dq),D[7:])))
        second=2*(dq[3]*qv[:3,None]-qv[3]*dq[:3]+np.cross(dq[:3].T,qv[:3]).T)
        velocity_maps.append(np.vstack((E[:3],angular(q,dqv)+second,E[7:])))
        poses.append(np.r_[value[:3],q,value[7:]])
        speeds.append(np.r_[velocity[:3],angular(q,qv[:,None]).ravel(),velocity[7:]])
    return np.array(poses),np.array(speeds),np.array(pose_maps),np.array(velocity_maps)

def linearize(com,leg,curve,raw_basis,coefficients,frames,rate=240.):
    poses,velocities,D,E=curve_maps(com,leg,curve,raw_basis,coefficients,frames,rate)
    cm=[];cvm=[];hm=[];eps=1e-5
    for q,v,d,e in zip(poses,velocities,D,E):
        J,A=com.matrices(q);P=np.empty((3,35));PV=np.empty((3,35))
        for j in range(35):
            plus=q.copy();minus=q.copy()
            if j<3:plus[j]+=eps;minus[j]-=eps
            elif j<6:
                axis=np.zeros(3);axis[j-3]=eps;R=Rotation.from_quat(q[3:7])
                plus[3:7]=(Rotation.from_rotvec(axis)*R).as_quat();minus[3:7]=(Rotation.from_rotvec(-axis)*R).as_quat()
            else:plus[j+1]+=eps;minus[j+1]-=eps
            positive=np.array(com.momentum(plus,v,True));negative=np.array(com.momentum(minus,v,True));difference=(positive-negative)/(2*eps)
            P[:,j]=difference[2];PV[:,j]=difference[1]
        cm.append(J@d);cvm.append(PV@d+J@e);hm.append(P@d+A@e)
    cm=np.array(cm);hm=np.array(hm);times=np.asarray(frames)/rate
    return dict(pose=D,joints=D[:,6:],com=cm,acceleration=CubicSpline(times,np.array(cvm))(times,1),momentum=hm,momentum_rate=CubicSpline(times,hm)(times,1))
