"""Bounded C2 reference correction toward measured stance material points.

Only reference leg coordinates change. Source audits, physical state, contact
constraints and actuator limits are owned by the caller and remain unchanged.
"""
from dataclasses import dataclass
import numpy as np
import torch
from bounded_leg_ik import solve_leg_pose
from numpy_leg_kinematics import LegKinematics


def smooth(u):
    u=np.clip(u,0.,1.)
    return u**3*(10.-15.*u+6.*u*u)


@dataclass
class Anchor:
    begin: float
    end: float
    point: np.ndarray
    local: np.ndarray


class MeasuredStanceReference:
    def __init__(self, urdf, joint_names, modes, fps, *, ramp=.10, limit=.03):
        if not np.isfinite([fps,ramp,limit]).all() or min(fps,ramp,limit)<=0:
            raise ValueError('Positive finite reference parameters required')
        self.model=LegKinematics(urdf,joint_names)
        self.modes=np.asarray(modes,dtype=int).reshape(-1)
        self.fps,self.ramp,self.limit=float(fps),float(ramp)*fps,float(limit)
        self.anchors={};self.last_interval={};self.events=[]
        self.peak_position_residual=0.;self.peak_orientation_residual=0.
        self.peak_joint_correction=0.

    def observe(self, frame, q, forces, contact_data):
        index=min(len(self.modes)-1,max(0,int(frame)))
        normals,points,_,separations,counts,starts=map(np.asarray,contact_data)
        normals=normals.reshape(-1);separations=separations.reshape(-1)
        for side in range(2):
            bit=1<<side
            if not self.modes[index]&bit:continue
            end=index+1
            while end<len(self.modes) and self.modes[end]&bit:end+=1
            key=(side,end)
            if self.last_interval.get(side)==key or forces[side]<100.:continue
            start,count=int(starts.reshape(-1)[side]),int(counts.reshape(-1)[side]);sl=slice(start,start+count)
            mask=(normals[sl]>.5)&(separations[sl]<=.001)
            if not mask.any() or end-frame<self.ramp:continue
            world=np.asarray(points[sl])[mask].mean(0)
            pos,R,_=self.model.evaluate(q,side)
            local=R.T@(world-pos)
            self.anchors[side]=Anchor(float(frame),float(end),world.copy(),local)
            self.last_interval[side]=key
            self.events.append(dict(side=side,frame=float(frame),end=float(end),point=world.tolist(),local=local.tolist()))

    def correction(self, frame, q):
        out=np.array(q,dtype=float,copy=True)
        for side,anchor in self.anchors.items():
            if not anchor.begin<frame<anchor.end:continue
            weight=smooth((frame-anchor.begin)/self.ramp)*smooth((anchor.end-frame)/self.ramp)
            pos,R,_=self.model.evaluate(q,side)
            error=anchor.point-pos-R@anchor.local
            # Smooth radial saturation avoids a derivative corner at the cap.
            delta=weight*error*self.limit/np.sqrt(self.limit**2+error@error)
            out,pe,re=solve_leg_pose(self.model,out,side,pos+delta,R,max_iterations=80,position_tolerance=1e-10,orientation_tolerance=1e-9)
            self.peak_position_residual=max(self.peak_position_residual,pe)
            self.peak_orientation_residual=max(self.peak_orientation_residual,re)
            if pe>.002 or re>.01:
                raise RuntimeError(f'Measured stance reference IK residual at frame {frame:.3f}, foot {side}: {pe:.6f} m, {re:.6f} rad')
        result=out[7:]-np.asarray(q)[7:]
        self.peak_joint_correction=max(self.peak_joint_correction,float(abs(result).max()))
        return result

    def project(self, frame, rate, acceleration, sampler, reference):
        from fabrics_sim.reference.whole_body import WholeBodyReference
        if not hasattr(self,'double_sampler'):
            import copy
            self.double_sampler=copy.deepcopy(sampler)
            def make_double(curve):
                curve.trajectory=curve.trajectory.double()
                if getattr(curve,'coefficients',None) is not None:curve.coefficients=curve.coefficients.double()
                if hasattr(curve,'original'):make_double(curve.original)
            make_double(self.double_sampler)
        h=.001
        if frame-rate*h<0:times=np.array([0.,h,2*h])
        elif frame+rate*h>len(self.modes)-1:times=np.array([-2*h,-h,0.])
        else:times=np.array([-h,0.,h])
        offsets=[]
        for dt in times:
            f=frame+rate*dt+.5*acceleration*dt*dt
            f=min(len(self.modes)-1,max(0.,f))
            pose=self.double_sampler.kinematics(f,rate+acceleration*dt,acceleration)[0]
            offsets.append(self.correction(f,pose.detach().cpu().numpy()[0]))
        offsets=np.asarray(offsets)
        q0=offsets[np.flatnonzero(times==0)[0]]
        weights=np.linalg.inv(np.vstack([np.ones(3),times,times**2]))
        dv=weights[:,1]@offsets
        da=2*weights[:,2]@offsets
        q,v,a=(x.clone() for x in (reference.q,reference.v,reference.a))
        q[:,7:]+=q.new_tensor(q0)
        v[:,6:]+=v.new_tensor(dv)
        a[:,6:]+=a.new_tensor(da)
        return WholeBodyReference(q,v,a)
