"""Experimental finite-horizon rigid-body QP. No simulator state writes.
The first control retains the supplied, already validated WBC constraints.
Future constraints are linearized predictions, not a physical certificate.
"""
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
import clarabel
from mujoco_reference_dynamics import ReferenceDynamics
from centroidal_pose_linearization import skew

def difference(q,target):
 return np.r_[q[:3]-target[:3],(Rotation.from_quat(q[3:7])*Rotation.from_quat(target[3:7]).inv()).as_rotvec(),q[7:]-target[7:]]
def integrate(q,delta):
 result=q.copy();result[:3]+=delta[:3];result[3:7]=(Rotation.from_rotvec(delta[3:6])*Rotation.from_quat(q[3:7])).as_quat();result[7:]+=delta[6:];return result

class FullBodyPreview:
 def __init__(self,model,reference,source,modes,geometry,limits,effort,velocity,dt=.025,horizon=12):
  self.model=model;self.reference=reference;self.source=source;self.modes=modes;self.geometry=geometry;self.limits=limits;self.effort=effort;self.velocity=velocity;self.dt=dt;self.N=horizon
 def solve(self,q,v,frame,rate,first):
  N,dt=self.N,self.dt;nx=70*(N+1);n=nx+76*N;frames=np.minimum(frame+np.arange(N+1)*rate*dt,len(self.modes)-1)
  refs=[[a.numpy().ravel() for a in self.reference.kinematics(float(f),rate)] for f in frames]
  original=[self.source.kinematics(float(f),rate)[0].numpy().ravel() for f in frames]
  H=np.zeros(n);g=np.zeros(n);rows=[];lo=[];hi=[];soc=[]
  def add(parts,lower,upper):
   size=next(iter(parts.values())).shape[0];row=sparse.lil_matrix((size,n))
   for offset,value in parts.items():row[:,offset:offset+value.shape[1]]=value
   rows.append(row.tocsc());lo.extend(np.broadcast_to(lower,(size,)));hi.extend(np.broadcast_to(upper,(size,)))
  def tube(k,start,width,radius,offset):
   row=sparse.lil_matrix((width+1,n));row[1:,70*k+start:70*k+start+width]=-np.eye(width);soc.append((row.tocsc(),np.r_[radius,offset]))
  position_weights=np.r_[[100000.,100000.,30000.],[3000.]*3,[400.]*29]
  velocity_weights=np.r_[[1500.]*3,[60.]*3,[4.]*29]
  x0=np.r_[difference(q,refs[0][0]),v-refs[0][1]];add({0:np.eye(70)},x0,x0)
  anchors={};previous_active=set()
  for k,(rq,rv,ra) in enumerate(refs):
   factor=4. if k==N else 1.;H[70*k:70*k+35]=2*factor*position_weights;H[70*k+35:70*k+70]=2*factor*velocity_weights
   add({70*k+6:np.eye(29)},self.limits[:,0]+.005-rq[7:],self.limits[:,1]-.005-rq[7:]);add({70*k+41:np.eye(29)},-self.velocity-rv[6:],self.velocity-rv[6:])
   source_difference=difference(rq,original[k]);tube(k,0,3,.195,source_difference[:3]);tube(k,3,3,.34,source_difference[3:6]);tube(k,6,29,.195*np.sqrt(29),source_difference[6:])
   if k==N:break
   u=nx+76*k;H[u:u+35]=2e-4;g[u:u+35]=-2e-4*ra;H[u+35:u+47]=2e-6;H[u+47:u+76]=2e-4
   Aq=np.eye(35);Aq[3:6,3:6]+=dt*skew(rv[3:6]);drift=difference(integrate(rq,dt*rv),refs[k+1][0]);control=np.zeros((35,76));control[:,:35]=-.5*dt*dt*np.eye(35)
   add({70*(k+1):np.eye(35),70*k:-Aq,70*k+35:-dt*np.eye(35),u:control},drift,drift)
   control=np.zeros((35,76));control[:,:35]=-dt*np.eye(35);dv=rv-refs[k+1][1]
   add({70*(k+1)+35:np.eye(35),70*k+35:-np.eye(35),u:control},dv,dv)
   if k==0:
    add({u:first['A_eq']},first['b_eq'],first['b_eq']);add({u:first['G']},-np.inf,first['b']);add({u:np.eye(76)},first['lb'],first['ub'])
    state=self.model.evaluate(q,v)
    active=[j for j in [0,1] if abs(first['ub'][35+6*j+2])>1e-5]
    # Bounds alone do not encode inactive wrenches; use the supplied observed mode.
    active=[j for j in [0,1] if int(first['mode'])&(1<<j)]
    for side in active:
     points,radii=self.geometry[side];foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();z=foot[2]+(points@R.T)[:,2]-radii;index=int(z.argmin());local=points[index].copy();local-=R.T@np.array([0.,0.,radii[index]]);anchors[side]=(foot[:3]+R@local,local)
    previous_active=set(active);continue
   nominal_error=x0[:35]*np.exp(-k*dt/.3);nominal_velocity_error=x0[35:]*np.exp(-k*dt/.3)
   nq=integrate(rq,nominal_error);nv=rv+nominal_velocity_error;state=self.model.evaluate(nq,nv);mode=int(self.modes[int(frames[k])]);active=[j for j in [0,1] if mode&(1<<j)]
   dynamics=np.zeros((35,76));dynamics[:,:35]=state['mass'];dynamics[:,47:]=-np.r_[np.zeros((6,29)),np.eye(29)]
   for side in [0,1]:dynamics[:,35+6*side:41+6*side]=-state['jacobian'][side].T
   add({u:dynamics},-state['bias'],-state['bias']);add({u+47:np.eye(29)},-self.effort,self.effort)
   for side in [0,1]:
    if side not in active:add({u+35+6*side:np.eye(6)},0.,0.);continue
    foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=self.geometry[side];z=foot[2]+(points@R.T)[:,2]-radii;lowest=np.flatnonzero(z<=z.min()+.003);index=int(z.argmin());local=points[index].copy();local-=R.T@np.array([0.,0.,radii[index]])
    if side not in previous_active:
     anchor=foot[:3]+R@local;anchor[2]=0.;anchors[side]=(anchor,local)
    anchor,local=anchors[side];offset=R@local;J=state['jacobian'][side];Jp=J[:3]-skew(offset)@J[3:];point=foot[:3]+offset
    # Contact position/velocity tolerances cover discrete linearization error.
    tangent=nominal_error[3:6];theta=np.linalg.norm(tangent);W=skew(tangent);left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
    Jq=Jp.copy();Jq[:,3:6]=Jp[:,3:6]@left
    target=anchor-point+Jq@nominal_error
    add({70*k:Jq},target-.003,target+.003)
    add({70*k+35:Jp},-Jp@rv-.04,-Jp@rv+.04)
    yaw=Rotation.from_quat(foot[3:]).as_euler('xyz')[2];Y=Rotation.from_euler('z',yaw).as_matrix();contact_center=foot[:3]+offset;D=np.block([[Y.T,np.zeros((3,3))],[-Y.T@skew(contact_center-foot[:3]),Y.T]])
    xy=((points[lowest]@R.T)-offset)@Y;extent=np.maximum(abs(xy[:,:2]).max(0),.001);toe,halfwidth=min(.12,extent[0]),min(.03,extent[1])
    cone=[];bound=[]
    for sx in [-1,1]:
     for sy in [-1,1]:cone.append([sx,sy,-.5,0,0,0]);bound.append(0.)
    cone.extend([[0,0,-1,0,0,0],[0,0,1,0,0,0]]);bound.extend([0.,800.])
    for axis,width in [(3,halfwidth),(4,toe),(5,.02)]:
     for sign in [-1,1]:row=np.zeros(6);row[2]=-width;row[axis]=sign;cone.append(row);bound.append(0.)
    add({u+35+6*side:np.array(cone)@D},-np.inf,bound)
   previous_active=set(active)
  C=sparse.vstack(rows,format='csc');low=np.array(lo);high=np.array(hi);eq=np.isfinite(low)&np.isfinite(high)&(abs(high-low)<1e-12);up=np.isfinite(high)&~eq;down=np.isfinite(low)&~eq
  A=sparse.vstack([C[eq],C[up],-C[down]]+[s[0] for s in soc],format='csc');b=np.r_[high[eq],high[up],-low[down],*[s[1] for s in soc]];cones=[clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(up.sum()+down.sum()))]+[clarabel.SecondOrderConeT(len(s[1])) for s in soc]
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=150;settings.tol_gap_abs=1e-7;settings.tol_gap_rel=1e-7;settings.tol_feas=1e-7
  solution=clarabel.DefaultSolver(sparse.diags(H,format='csc'),g,A,b,cones,settings).solve();x=np.array(solution.x);linear_violation=float(max(np.max(low-C@x),np.max(C@x-high),0.));cone_violation=max(float(np.linalg.norm(s[1][1:]-s[0][1:]@x)-(s[1][0]-s[0][0]@x)[0]) for s in soc);ok=str(solution.status) in ('Solved','AlmostSolved') and linear_violation<1e-4 and cone_violation<1e-5
  return dict(success=ok,status=str(solution.status),linear_violation=linear_violation,tube_violation=cone_violation,first_control=x[nx:nx+76],controls=x[nx:].reshape(N,76),states=x[:nx].reshape(N+1,70),frames=frames,objective=float(solution.obj_val))
