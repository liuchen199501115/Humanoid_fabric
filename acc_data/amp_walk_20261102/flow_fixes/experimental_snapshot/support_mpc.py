"""Small planar COM preview with source footsteps and measured initial state.

LIPM approximation only: WBC still enforces full-body dynamics, actual contact
patches and motor limits. This planner cannot certify those constraints.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull
import osqp


class SupportMPC:
    def __init__(self, height, dt=.02, horizon=48):
        if not np.isfinite(height) or height<=0 or dt<=0 or horizon<2:
            raise ValueError('Invalid COM preview configuration')
        self.height=float(height);self.dt=float(dt);self.horizon=int(horizon)
        self.omega2=9.81/height
        n=self.horizon; d=self.dt
        # Eliminate states: p_k = p_0 + k*d*v_0 + B_k @ u.
        self.B=np.zeros((n,n));self.V=np.zeros((n,n))
        for k in range(n):
            self.B[k,:k+1]=d*d*(np.arange(k,-1,-1)+.5)
            self.V[k,:k+1]=d
        self.times=np.arange(1,n+1)*d
        self.last_report={}

    def solve(self, position, velocity, reference, polygons):
        n=self.horizon
        p,v,ref=map(lambda x:np.asarray(x,dtype=float),(position,velocity,reference))
        if p.shape!=(2,) or v.shape!=(2,) or ref.shape!=(n,2) or len(polygons)!=n:
            raise ValueError('Invalid preview states or support polygons')
        free=p[None]+self.times[:,None]*v[None]
        # Variables are ax[0:N], ay[0:N].
        H=40*self.B.T@self.B+np.eye(n)*.02
        H+=2*self.V.T@self.V
        rv=np.gradient(ref,self.dt,axis=0)
        capture=self.B[-1]+self.V[-1]/np.sqrt(self.omega2)
        capture_offset=free[-1]+v/np.sqrt(self.omega2)
        capture_target=ref[-1]+rv[-1]/np.sqrt(self.omega2)
        H+=20*np.outer(capture,capture)
        q=np.concatenate([40*self.B.T@(free[:,j]-ref[:,j])+
                          2*self.V.T@(v[j]-rv[:,j])+
                          20*capture*(capture_offset[j]-capture_target[j]) for j in range(2)])
        P=sparse.block_diag([H,H],format='csc')
        rows=[];bounds=[]
        for k,polygon in enumerate(polygons):
            hull=ConvexHull(np.asarray(polygon,dtype=float))
            cp=self.B[k].copy();cp[k]-=1/self.omega2
            for nx,ny,b in hull.equations:
                rows.append(np.r_[nx*cp,ny*cp])
                bounds.append(-b-np.dot([nx,ny],free[k]))
        # Continuous walking need not stop inside the last stance polygon.
        # Penalize the source terminal capture state above; only physical CoP
        # constraints are hard. Long-term viability is not certified here.
        A=sparse.csc_matrix(np.asarray(rows)); upper=np.asarray(bounds)
        solver=osqp.OSQP()
        solver.setup(P=P,q=q,A=A,l=np.full(len(bounds),-np.inf),u=upper,
                     verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=20000,polish=True)
        result=solver.solve()
        x=np.asarray(result.x,dtype=float) if result.info.status_val in (1,2) else None
        violation=float(np.max(A@x-upper)) if x is not None and np.isfinite(x).all() else float('inf')
        valid=result.info.status_val in (1,2) and violation<=1e-4
        self.last_report=dict(status=result.info.status,valid=valid,max_support_violation=violation,
                              model='constant-height LIPM; full dynamics checked by WBC')
        if not valid:raise RuntimeError('COM preview failed: '+str(self.last_report))
        return np.array([x[0],x[n]])
