"""Diagnostic smooth interpolation transition; source audit and physics unchanged."""
import math
import torch
import fabrics_sim.reference.interpolation as interpolation
Original = interpolation.MotionInterpolator


def qderivative(quat, omega):
    return .5*torch.cat((quat[:,3:]*omega+torch.linalg.cross(omega,quat[:,:3]),
                        -(omega*quat[:,:3]).sum(-1,keepdim=True)),dim=-1)


def pose_derivatives(q,v,a):
    qv=qderivative(q[:,3:7],v[:,3:6])
    qa=qderivative(q[:,3:7],a[:,3:6])+qderivative(qv,v[:,3:6])
    return torch.cat((v[:,:3],qv,v[:,6:]),-1),torch.cat((a[:,:3],qa,a[:,6:]),-1)


class BlendedInterpolator:
    def __init__(self,trajectory,*,pose_dimension=36,method='pchip'):
        self.original=Original(trajectory,pose_dimension=pose_dimension,method=method)
        self.smooth=Original(trajectory,pose_dimension=pose_dimension,method='bspline') if method=='pchip' and len(trajectory)>=1000 else None
        self.start,self.width=184.,20.

    def __getattr__(self,name):
        return getattr(self.original,name)

    def __call__(self,frame):
        return self.kinematics(frame,1.,0.)[0]

    def kinematics(self,frame,frame_rate,frame_acceleration=0.):
        if self.smooth is None or frame <= self.start:
            return self.original.kinematics(frame,frame_rate,frame_acceleration)
        if frame >= self.start+self.width:
            return self.smooth.kinematics(frame,frame_rate,frame_acceleration)
        q0,v0,a0=self.original.kinematics(frame,frame_rate,frame_acceleration)
        q1,v1,a1=self.smooth.kinematics(frame,frame_rate,frame_acceleration)
        dq0,ddq0=pose_derivatives(q0,v0,a0);dq1,ddq1=pose_derivatives(q1,v1,a1)
        if (q0[:,3:7]*q1[:,3:7]).sum()<0:
            q1[:,3:7]*=-1;dq1[:,3:7]*=-1;ddq1[:,3:7]*=-1
        u=(frame-self.start)/self.width
        s=10*u**3-15*u**4+6*u**5
        ds=(30*u**2-60*u**3+30*u**4)/self.width
        dds=(60*u-180*u**2+120*u**3)/self.width**2
        sd=ds*frame_rate;sdd=dds*frame_rate**2+ds*frame_acceleration
        q=(1-s)*q0+s*q1
        dq=(1-s)*dq0+s*dq1+sd*(q1-q0)
        ddq=(1-s)*ddq0+s*ddq1+2*sd*(dq1-dq0)+sdd*(q1-q0)
        raw=q[:,3:7].clone();du=dq[:,3:7];ddu=ddq[:,3:7]
        norm=raw.norm(dim=-1,keepdim=True);quat=raw/norm
        norm_v=(quat*du).sum(-1,keepdim=True)
        quat_v=(du-quat*norm_v)/norm
        norm_a=(du.square().sum(-1,keepdim=True)+(raw*ddu).sum(-1,keepdim=True)-norm_v.square())/norm
        quat_a=(ddu-quat*norm_a-2*quat_v*norm_v)/norm
        def angular(derivative):
            return 2*(quat[:,3:]*derivative[:,:3]-derivative[:,3:]*quat[:,:3]+torch.linalg.cross(quat[:,:3],derivative[:,:3]))
        q[:,3:7]=quat
        return q,torch.cat((dq[:,:3],angular(quat_v),dq[:,7:]),-1),torch.cat((ddq[:,:3],angular(quat_a),ddq[:,7:]),-1)

if __name__=='__main__':
    interpolation.MotionInterpolator=BlendedInterpolator
    import test_g1
    test_g1.main()
