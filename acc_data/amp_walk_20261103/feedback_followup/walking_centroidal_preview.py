"""Walking preview with a soft vertical tracking policy; source q/v/a stay unchanged.
Observed stance anchors and wrench cones persist until their planned lift-off.
New landings still use an approximate flat patch. The live WBC retains
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
        self.reference_clock=None;self.cache={};self.last_time=-1.;self.last_result=None;self.records=[]

    @torch.no_grad()
    def reference(self,frame,frame_rate=None,frame_acceleration=0.):
        frame=min(max(float(frame),0.),len(self.modes)-1)
        frame_rate=self.fps if frame_rate is None else float(frame_rate)
        key=(round(frame,6),round(frame_rate,6),round(frame_acceleration,6))
        if key not in self.cache:
            q,v,a=self.sampler.kinematics(frame,frame_rate,frame_acceleration)
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
        weights=np.array([4000.,4000.,4000.,150.,150.,150.,4.,4.,1.])
        control_weights=np.tile([1e-5,1e-5,1e-5,1e-3,1e-3,1e-3],2)
        diagonal=np.r_[np.tile(weights,N+1),np.tile(control_weights,N)]
        diagonal[9*N:9*(N+1)]*=4.
        target=np.zeros(n);refs=[self.reference(*(self.reference_clock(k*dt) if getattr(self,'reference_clock',None) is not None else (frame+k*dt*self.fps,self.fps,0.))) for k in range(N+1)]
        for k in range(N+1):target[9*k:9*k+9]=refs[k][0]
        rows=[];lower=[];upper=[];vertical_rows=[];vertical_targets=[]
        def append(row,lo,hi):
            rows.append(row);lower.extend(np.atleast_1d(lo));upper.extend(np.atleast_1d(hi))
        row=np.zeros((9,n));row[:,:9]=np.eye(9);append(row,state,state)
        continuing=np.array([bool(mode&1),bool(mode&2)])
        persistent_nodes=[]
        for k in range(N):
            ref,foot,planned=refs[k]
            foot=foot.copy()
            if k==0:
                foot,planned=feet,mode
                ref=ref.copy();ref[:3]=state[:3]
            active=[bool(planned&1),bool(planned&2)];number=sum(active)
            continuing &= np.asarray(active)
            persistent_nodes.append(continuing.tolist())
            for side in range(2):
                if continuing[side]:foot[side]=feet[side]
            ur=np.zeros(12)
            for side in range(2):
                if active[side]:ur[6*side+2]=m*9.81/number
            target[nx+12*k:nx+12*k+12]=ur
            if number:
                # Prevent the centroidal relaxation from postponing support by
                # free-falling now and demanding a large future landing impulse.
                # The live WBC keeps its unified vertical tracking request.
                kp_z,kd_z=50.,14.
                az_reference=float(np.clip((refs[k+1][0][5]-refs[k][0][5])/dt,-8.,8.))
                row=np.zeros((1,n));row[0,nx+12*k+2]=1/m;row[0,nx+12*k+8]=1/m
                row[0,9*k+2]=kp_z;row[0,9*k+5]=kd_z
                value=9.81+az_reference+kp_z*refs[k][0][2]+kd_z*refs[k][0][5]
                vertical_rows.append(row[0]);vertical_targets.append(value)
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
                if continuing[side]:
                    selected=np.any(abs(first_G[:,6*side:6*side+6])>1e-12,axis=1)
                    other=slice(6*(1-side),6*(1-side)+6)
                    if np.any(abs(first_G[selected,other])>1e-12):raise ValueError('Expected independent per-foot wrench rows')
                    row=np.zeros((int(selected.sum()),n));row[:,nx+12*k:nx+12*(k+1)]=first_G[selected]
                    append(row,np.full(int(selected.sum()),-np.inf),first_b[selected])
                    continue
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
        vertical=sparse.csc_matrix(np.asarray(vertical_rows));vertical_weight=10.
        objective=sparse.diags(2*diagonal,format='csc')+2*vertical_weight*(vertical.T@vertical)
        linear=2*vertical_weight*np.asarray(vertical.T@(vertical@target-np.asarray(vertical_targets))).reshape(-1)
        solver=osqp.OSQP();solver.setup(P=sparse.triu(objective,format='csc'),q=linear,A=matrix,l=lo-shift,u=hi-shift,
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
            residual=violation,persistent_stance_nodes=persistent_nodes,predicted_state=values[:nx].reshape(N+1,9).tolist(),acceleration=acceleration.tolist(),momentum_rate=momentum_rate.tolist(),wrench=u.tolist())

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
