"""Vectorized offline RNEA with nominal foot-wrench allocation.

This accelerates reference linearization only. Double support uses equal
load sharing; resulting motor estimates are not feasibility certificates.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def mv(matrix,vector):
    return (matrix@np.asarray(vector)[...,None])[...,0]


def reference_outputs(model,q,v,a,contacts,armature):
    q,v,a=np.asarray(q),np.asarray(v),np.asarray(a)
    batch=len(q)
    if q.shape!=(batch,36) or v.shape!=(batch,35) or a.shape!=(batch,35):
        raise ValueError('Expected batches of 36 positions and 35 tangent coordinates')
    contacts=np.broadcast_to(np.asarray(contacts,dtype=bool),(batch,2))
    poses={model.root:(q[:,:3],Rotation.from_quat(q[:,3:7]).as_matrix())}
    omega={model.root:v[:,3:6]};alpha={model.root:a[:,3:6]};acc={model.root:a[:,:3]}
    axes={};origins={}
    for parent,child,xyz,local_rotation,axis,index in model.joints:
        p,R=poses[parent];r=mv(R,xyz);Rc=R@local_rotation
        w=omega[parent].copy();al=alpha[parent].copy()
        ac=acc[parent]+np.cross(al,r)+np.cross(w,np.cross(w,r))
        if index is not None:
            aw=mv(Rc,axis);axes[index]=aw;origins[index]=p+r
            al+=aw*a[:,6+index,None]+np.cross(w,aw*v[:,6+index,None])
            w+=aw*v[:,6+index,None]
            Rc=Rc@Rotation.from_rotvec(axis[None]*q[:,7+index,None]).as_matrix()
        poses[child]=(p+r,Rc);omega[child]=w;alpha[child]=al;acc[child]=ac
    forces={};moments={};com=np.zeros((batch,3))
    for name,(mass,local) in model.links.items():
        p,R=poses[name];r=mv(R,local);w=omega[name];al=alpha[name]
        inertia=R@model.inertias[name]@R.swapaxes(-1,-2)
        force=mass*(acc[name]+np.cross(al,r)+np.cross(w,np.cross(w,r))+[0,0,9.81])
        forces[name]=force;moments[name]=mv(inertia,al)+np.cross(w,mv(inertia,w))+np.cross(r,force)
        com+=mass*(p+r)
    com/=model.mass;result=np.zeros((batch,35))
    for parent,child,xyz,local_rotation,axis,index in reversed(model.joints):
        if index is not None:result[:,6+index]=(axes[index]*moments[child]).sum(-1)
        forces[parent]+=forces[child]
        moments[parent]+=moments[child]+np.cross(poses[child][0]-poses[parent][0],forces[child])
    F=forces[model.root];M=moments[model.root]
    result[:,:3]=F;result[:,3:6]=M-np.cross(com-q[:,:3],F)
    count=np.maximum(1,contacts.sum(-1))
    for side,label in enumerate(('left','right')):
        foot=poses[label+'_ankle_roll_link'][0]
        wrench=np.concatenate((F,M-np.cross(foot-q[:,:3],F)),axis=-1)/count[:,None]
        for index in range(6*side,6*side+6):
            column=np.concatenate((np.cross(axes[index],foot-origins[index]),axes[index]),axis=-1)
            result[:,6+index]-=contacts[:,side]*(column*wrench).sum(-1)
    result[:,6:]+=np.asarray(armature)*a[:,6:]
    return result
