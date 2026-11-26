"""Diagnostic receding-horizon centroidal feedback; never replaces source q/v/a.
Future flat support patches are a planning approximation. The live WBC retains
its measured contact patches, full-body dynamics and all actuator constraints.
"""
import numpy as np
from scipy import sparse
import osqp
import torch


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


class CentroidalPreview:
    def __init__(self,fabric,sampler,modes,fps,horizon=24,dt=.025):
        self.fabric,self.sampler,self.modes,self.fps=fabric,sampler,np.asarray(modes).reshape(-1),fps
        self.N,self.dt=horizon,dt
        self.mass=float(fabric.center_of_mass_taskmap.link_masses.sum())
        self.feet=[fabric.dynamics_tasks.model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        self.cache={};self.last_time=-1.;self.last_result=None;self.records=[]

    @torch.no_grad()
    def reference(self,frame):
        frame=min(max(float(frame),0.),len(self.modes)-1)
        key=round(frame,6)
        if key not in self.cache:
            q,v,a=self.sampler.kinematics(frame,self.fps,0.)
            st=self.fabric.dynamics_tasks.model.snapshot(q)
            cv=(st['com_jac']@v[...,None]).squeeze(-1)
            h=(st['momentum_matrix']@v[...,None]).squeeze(-1)
            state=torch.cat((st['com'],cv,h),dim=-1)[0].numpy().astype(float)
            feet=st['transforms'][0,self.feet,:3].numpy().astype(float).copy();feet[:,2]=0.
            self.cache[key]=(state,feet,int(self.modes[int(frame)]))
        return self.cache[key]

    def solve(self,state,feet,mode,frame,first_G,first_b):
        N,dt,m=self.N,self.dt,self.mass
        nx=9*(N+1);nu=12*N;n=nx+nu
        weights=np.array([4000.,4000.,4000.,150.,150.,150.,1000.,1000.,300.])
        control_weights=np.tile([1e-5,1e-5,1e-5,1e-3,1e-3,1e-3],2)
        diagonal=np.r_[np.tile(weights,N+1),np.tile(control_weights,N)]
        diagonal[9*N:9*(N+1)]*=4.
        target=np.zeros(n);refs=[self.reference(frame+k*dt*self.fps) for k in range(N+1)]
        for k in range(N+1):target[9*k:9*k+9]=refs[k][0]
        rows=[];lower=[];upper=[]
        def append(row,lo,hi):
            rows.append(row);lower.extend(np.atleast_1d(lo));upper.extend(np.atleast_1d(hi))
        row=np.zeros((9,n));row[:,:9]=np.eye(9);append(row,state,state)
        for k in range(N):
            ref,foot,planned=refs[k]
            if k==0:
                foot,planned=feet,mode
                ref=ref.copy();ref[:3]=state[:3]
            active=[bool(planned&1),bool(planned&2)];number=sum(active)
            ur=np.zeros(12)
            for side in range(2):
                if active[side]:ur[6*side+2]=m*9.81/number
            target[nx+12*k:nx+12*k+12]=ur
            A=np.eye(9);A[:3,3:6]=np.eye(3)*dt
            B=np.zeros((9,12));offset=np.zeros(9)
            offset[2]=-.5*dt*dt*9.81;offset[5]=-dt*9.81
            # Linearize -c x sum(F) around the source COM and nominal support load.
            sum_force=ur[:3]+ur[6:9]
            A[6:9,:3]=dt*skew(sum_force)
            offset[6:9]=-dt*skew(sum_force)@ref[:3]
            for side in range(2):
                j=6*side
                B[:3,j:j+3]=np.eye(3)*(.5*dt*dt/m)
                B[3:6,j:j+3]=np.eye(3)*(dt/m)
                B[6:9,j:j+3]=dt*skew(foot[side]-ref[:3])
                B[6:9,j+3:j+6]=np.eye(3)*dt
            row=np.zeros((9,n));row[:,9*(k+1):9*(k+2)]=np.eye(9);row[:,9*k:9*(k+1)]=-A;row[:,nx+12*k:nx+12*(k+1)]=-B
            append(row,offset,offset)
            if k==0:
                row=np.zeros((len(first_b),n));row[:,nx:nx+12]=first_G
                append(row,np.full(len(first_b),-np.inf),first_b)
            for side in range(2):
                base=nx+12*k+6*side
                if not active[side]:
                    row=np.zeros((6,n));row[:,base:base+6]=np.eye(6);append(row,np.zeros(6),np.zeros(6));continue
                row=np.zeros((1,n));row[0,base+2]=1.;append(row,0.,800.)
                if k==0:continue
                # Inscribed friction pyramid and conservative sole CoP/yaw bounds.
                G=[]
                for sx in [-1.,1.]:
                    for sy in [-1.,1.]:G.append([sx,sy,-.5,0.,0.,0.])
                G.extend([[0.,0.,-.025,1.,0.,0.],[0.,0.,-.025,-1.,0.,0.],
                          [0.,0.,-.05,0.,1.,0.],[0.,0.,-.12,0.,-1.,0.],
                          [0.,0.,-.015,0.,0.,1.],[0.,0.,-.015,0.,0.,-1.]])
                row=np.zeros((len(G),n));row[:,base:base+6]=np.asarray(G);append(row,np.full(len(G),-np.inf),np.zeros(len(G)))
        matrix=sparse.csc_matrix(np.vstack(rows));lo=np.asarray(lower);hi=np.asarray(upper)
        shift=matrix@target
        solver=osqp.OSQP();solver.setup(P=sparse.diags(2*diagonal,format='csc'),q=np.zeros(n),A=matrix,l=lo-shift,u=hi-shift,
            verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,polish=True)
        solution=solver.solve()
        if solution.x is None or solution.info.status_val not in (1,2):
            raise RuntimeError('Centroidal preview failed: '+solution.info.status)
        values=solution.x+target
        residual=matrix@values
        violation=float(max(np.max(lo-residual),np.max(residual-hi),0.))
        if violation>2e-3:raise RuntimeError(f'Centroidal preview constraint residual {violation:g}')
        u=values[nx:nx+12].reshape(2,6)
        acceleration=u[:,:3].sum(0)/m+np.array([0.,0.,-9.81])
        momentum_rate=np.cross(feet-state[:3],u[:,:3]).sum(0)+u[:,3:].sum(0)
        return acceleration,momentum_rate,dict(frame=float(frame),mode=int(mode),status=solution.info.status,
            residual=violation,acceleration=acceleration.tolist(),momentum_rate=momentum_rate.tolist(),wrench=u.tolist())

    @torch.no_grad()
    def update(self,time,frame,q,v,live_feet,mode,patches,G,b):
        if time-self.last_time>=.02-1e-8:
            st=self.fabric.dynamics_tasks.model.snapshot(q)
            state=torch.cat((st['com'],(st['com_jac']@v[...,None]).squeeze(-1),
                             (st['momentum_matrix']@v[...,None]).squeeze(-1)),-1)[0].numpy().astype(float)
            feet=live_feet[0,:,:3].numpy().astype(float).copy()
            transform=np.eye(12)
            for side,patch in patches.items():
                rot=st['rotations'][0,self.feet[side]].numpy().astype(float)
                feet[side]+=rot@patch.local_point[0].numpy().astype(float)
                axes=patch.frame[0].numpy().astype(float)
                transform[side*6:side*6+3,side*6:side*6+3]=axes
                transform[side*6+3:side*6+6,side*6+3:side*6+6]=axes
            first_G=G[0].numpy().astype(float)@transform.T;first_b=b[0].numpy().astype(float)
            acc,rate,record=self.solve(state,feet,int(mode),frame,first_G,first_b)
            self.last_result=(acc,rate);self.last_time=time;record['time']=float(time);self.records.append(record)
            self.fabric.dynamics_tasks.model.snapshot(q)
        return self.last_result
