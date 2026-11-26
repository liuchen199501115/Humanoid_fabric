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
 def __init__(self,model,reference,source,modes,geometry,limits,effort,velocity,dt=.025,horizon=12,guarded=False,contact_delay_frames=0):
  self.model=model;self.reference=reference;self.source=source;self.modes=modes;self.geometry=geometry;self.limits=limits;self.effort=effort;self.velocity=velocity;self.dt=dt;self.N=horizon;self.guarded=guarded;self.contact_delay_frames=int(contact_delay_frames)
 def planned_mode(self,frame):
  return int(self.modes[int(np.clip(frame-self.contact_delay_frames,0,len(self.modes)-1))])
 def solve(self,q,v,frame,rate,first,nominal=None):
  N,dt=self.N,self.dt;nx=70*(N+1);impulse_start=nx+76*N;n=impulse_start+12*N;impact_maps={};patch_impact_nodes={};contact_patch_indices={};impact_patch_indices={};frames=np.minimum(frame+np.arange(N+1)*rate*dt,len(self.modes)-1)
  refs=[[a.numpy().ravel() for a in self.reference.kinematics(float(f),rate)] for f in frames]
  original=[self.source.kinematics(float(f),rate)[0].numpy().ravel() for f in frames]
  H=np.zeros(n);g=np.zeros(n);rows=[];lo=[];hi=[];soc=[];hessian_blocks=[];elastic_blocks=[];contact_force_blocks=[];impulse_force_blocks=[]
  def add(parts,lower,upper,group=None):
   nonlocal n,H,g
   if group in getattr(self,"diagnostic_skip",set()):return
   size=next(iter(parts.values())).shape[0]
   if group in getattr(self,'elastic_groups',set()):
    slack=n;n+=size;parts=dict(parts);parts[slack]=np.eye(size)
    weight={'future_ground':1e6,'future_contact_position':1e6,'future_contact_velocity':1e4,'future_contact_acceleration':1e2}[group]*self.elastic_scale
    H=np.r_[H,np.full(size,2*weight)];g=np.r_[g,np.zeros(size)];elastic_blocks.append((group,slack,size))
   row=sparse.lil_matrix((size,n))
   for offset,value in parts.items():row[:,offset:offset+value.shape[1]]=value
   rows.append(row.tocsc());lo.extend(np.broadcast_to(lower,(size,)));hi.extend(np.broadcast_to(upper,(size,)))
  def tube(k,start,width,radius,offset):
   row=sparse.lil_matrix((width+1,n));row[1:,70*k+start:70*k+start+width]=-np.eye(width);soc.append((row.tocsc(),np.r_[radius,offset]))
  def feet_objective(k,rq,nominal_error,state):
   reference_state=self.model.evaluate(rq,refs[k][1]);mode=self.planned_mode(frames[k]);tangent=nominal_error[3:6];theta=np.linalg.norm(tangent);W=skew(tangent)
   left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
   block=np.zeros((35,35))
   for side in [0,1]:
    J=state['jacobian'][side].copy();J[:,3:6]=J[:,3:6]@left
    error=np.r_[state['feet'][side,:3]-reference_state['feet'][side,:3],(Rotation.from_quat(state['feet'][side,3:])*Rotation.from_quat(reference_state['feet'][side,3:]).inv()).as_rotvec()]
    weights=np.r_[[30000. if mode&(1<<side) else 200000.]*3,[3000.]*3]
    block+=2*J.T@(weights[:,None]*J);g[70*k:70*k+35]+=2*J.T@(weights*(error-J@nominal_error))
   hessian_blocks.append((70*k,block));return reference_state
  def swing_floor(reference_state,side,mode):
   if mode&(1<<side):return -.0005
   points,radii=self.geometry[side];foot=reference_state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();height=foot[2]+np.min((points@R.T)[:,2]-radii)
   return max(-.0005,min(.04,.8*height-.003))
  position_weights=np.r_[[100000.,100000.,30000.],[15000.]*3,[2000.]*12,([10000.]*17 if self.guarded else [2000.]*17)]
  velocity_weights=np.r_[[1500.]*3,[60.]*3,[4.]*12,([40.]*17 if self.guarded else [4.]*17)]
  x0=np.r_[difference(q,refs[0][0]),v-refs[0][1]];add({0:np.eye(70)},x0,x0)
  nominal_patches=[]
  for node,(nref_q,nref_v,_) in enumerate(refs):
   if node==0:patch_state=self.model.evaluate(q,v)
   else:
    error=x0[:35]*np.exp(-node*dt/.3) if nominal is None else nominal['states'][node,:35]
    patch_state=self.model.evaluate(integrate(nref_q,error),nref_v)
   patch=[]
   for side in [0,1]:
    foot=patch_state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=self.geometry[side];z=foot[2]+(points@R.T)[:,2]-radii
    patch.append(set(np.flatnonzero(z<=z.min()+.0005).tolist()))
   nominal_patches.append(patch)
  anchors={};previous_active=set()
  for k,(rq,rv,ra) in enumerate(refs):
   factor=4. if k==N else 1.;H[70*k:70*k+35]=2*factor*position_weights;H[70*k+35:70*k+70]=2*factor*velocity_weights
   add({70*k+6:np.eye(29)},self.limits[:,0]+.005-rq[7:],self.limits[:,1]-.005-rq[7:]);add({70*k+41:np.eye(29)},-self.velocity-rv[6:],self.velocity-rv[6:])
   source_difference=difference(rq,original[k]);tube(k,0,3,.195,source_difference[:3]);tube(k,3,3,(.28+.06*np.exp(-k*dt/.1)) if self.guarded else .34,source_difference[3:6]);tube(k,6,29,((.17+.025*np.exp(-k*dt/.1)) if self.guarded else .195)*np.sqrt(29),source_difference[6:])
   if k==N:
    nominal_error=x0[:35]*np.exp(-k*dt/.3) if nominal is None else nominal['states'][k,:35].copy()
    nq=integrate(rq,nominal_error);state=self.model.evaluate(nq,rv);reference_state=feet_objective(k,rq,nominal_error,state);mode=self.planned_mode(frames[k])
    tangent=nominal_error[3:6];theta=np.linalg.norm(tangent);W=skew(tangent);left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
    for side in [0,1]:
     foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=self.geometry[side];J=state['jacobian'][side]
     for point_local,radius in zip(points,radii):
      offset=R@point_local;Jpoint=J[:3]-skew(offset)@J[3:];Jpoint[:,3:6]=Jpoint[:,3:6]@left;height=foot[2]+offset[2]-radius
      add({70*k:Jpoint[2:3]},swing_floor(reference_state,side,mode)-height+Jpoint[2]@nominal_error,np.inf,group="future_ground")
    break
   u=nx+76*k;H[u:u+35]=2e-4;g[u:u+35]=-2e-4*ra;H[u+35:u+47]=2e-6;H[u+47:u+76]=2e-4
   Aq=np.eye(35);Aq[3:6,3:6]+=dt*skew(rv[3:6]);drift=difference(integrate(rq,dt*rv),refs[k+1][0]);control=np.zeros((35,76));control[:,:35]=-.5*dt*dt*np.eye(35)
   add({70*(k+1):np.eye(35),70*k:-Aq,70*k+35:-dt*np.eye(35),u:control},drift,drift)
   control=np.zeros((35,76));control[:,:35]=-dt*np.eye(35);dv=rv-refs[k+1][1]
   velocity_parts={70*(k+1)+35:np.eye(35),70*k+35:-np.eye(35),u:control}
   next_mode=self.planned_mode(frames[k+1]);current_mode=int(first['mode']) if k==0 else self.planned_mode(frames[k])
   new_sides=[side for side in [0,1] if k+1<N and next_mode&(1<<side) and not current_mode&(1<<side)]
   patch_gains=[side for side in [0,1] if k+1<N and current_mode&(1<<side) and next_mode&(1<<side) and nominal_patches[k+1][side]-nominal_patches[k][side]]
   impact_sides=[side for side in [0,1] if next_mode&(1<<side)] if new_sides or patch_gains else []
   if patch_gains:patch_impact_nodes[k]=patch_gains
   ip=impulse_start+12*k;H[ip:ip+12]=.02
   if impact_sides:
    nrq,nrv,_=refs[k+1];ne=x0[:35]*np.exp(-(k+1)*dt/.3) if nominal is None else nominal['states'][k+1,:35]
    impact_q=integrate(nrq,ne);impact_state=self.model.evaluate(impact_q,nrv);impulse_map=np.zeros((35,12))
    for side in impact_sides:
     impulse_map[:,side*6:side*6+6]=np.linalg.solve(impact_state['mass'],impact_state['jacobian'][side].T)
    velocity_parts[ip]=-impulse_map;impact_maps[k]=impact_sides
    # Linearize the configuration dependence of M(q)^-1 J(q)^T impulse.
    # Holding this map fixed drops the dominant touchdown geometry term.
    if nominal is not None:
     from generalized_dynamics_linearization import perturb
     impulse_nominal=nominal.get('impulses',np.zeros((N,12)))[k]
     derivative=np.zeros((35,35));epsilon=1e-5
     for coordinate in range(35):
      delta=np.zeros(35);delta[coordinate]=epsilon;values=[]
      for sign in [1.,-1.]:
       ds=self.model.evaluate(perturb(impact_q,sign*delta),nrv)
       impulse_rhs=sum((ds['jacobian'][side].T@impulse_nominal[6*side:6*side+6] for side in impact_sides),start=np.zeros(35))
       values.append(np.linalg.solve(ds['mass'],impulse_rhs))
      derivative[:,coordinate]=(values[0]-values[1])/(2*epsilon)
     theta=np.linalg.norm(ne[3:6]);W=skew(ne[3:6]);left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
     derivative[:,3:6]=derivative[:,3:6]@left
     velocity_parts[70*(k+1)]=-derivative
     dv=dv-derivative@ne
   for side in [0,1]:
    if side not in impact_sides:add({ip+6*side:np.eye(6)},0.,0.);continue
    foot=impact_state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=self.geometry[side];z=foot[2]+(points@R.T)[:,2]-radii;index=int(z.argmin());offset=R@points[index]-np.array([0.,0.,radii[index]])
    from native_patch_wrench import native_patch_wrench
    near=np.flatnonzero(z<=z.min()+.0005)
    impact_patch_indices.setdefault(k,{})[side]=near.tolist()
    patch_offsets=points[near]@R.T-radii[near,None]*np.array([0.,0.,1.])
    yaw=Rotation.from_quat(foot[3:]).as_euler('xyz')[2];Y=Rotation.from_euler('z',yaw).as_matrix()
    count=len(near);force_offset=n;n+=3*count;H=np.r_[H,np.full(3*count,2e-7)];g=np.r_[g,np.zeros(3*count)]
    impulse_force_blocks.append((k,side,near.tolist(),force_offset,count))
    nominal_impulse=np.zeros(6) if nominal is None else nominal.get('impulses',np.zeros((N,12)))[k,6*side:6*side+6]
    nominal_forces=np.tile(nominal_impulse[:3]/count,(count,1))
    if nominal is not None:
     old=nominal.get('impulse_point_forces',{}).get(k,{}).get(side)
     if old is not None:
      mapping=dict(zip(old['indices'],old['forces']))
      for local_index,point_index in enumerate(near):
       if int(point_index) in mapping:nominal_forces[local_index]=mapping[int(point_index)]
    theta=np.linalg.norm(ne[3:6]);W=skew(ne[3:6]);left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
    B=np.zeros((6,3*count));Dq_force=np.zeros((6,35));angular_map=impact_state['jacobian'][side][3:].copy();angular_map[:,3:6]=angular_map[:,3:6]@left
    for local_index,point_index in enumerate(near):
     rotated=R@points[point_index];point_offset=rotated-np.array([0.,0.,radii[point_index]])
     B[:3,3*local_index:3*local_index+3]=np.eye(3);B[3:,3*local_index:3*local_index+3]=skew(point_offset)
     Dq_force[3:]+=skew(nominal_forces[local_index])@skew(rotated)@angular_map
    add({ip+6*side:np.eye(6),force_offset:-B,70*(k+1):-Dq_force},-Dq_force@ne,-Dq_force@ne)
    column=R[:,0];yaw_gradient=np.array([-column[0]*column[2],-column[1]*column[2],column[0]**2+column[1]**2])/max(column[0]**2+column[1]**2,1e-8)
    yaw_dq=yaw_gradient@angular_map;cone=np.array([[sx,sy,-.5] for sx in [-1.,1.] for sy in [-1.,1.]])
    for local_index in range(count):
     friction_dq=(cone@(-skew(np.array([0.,0.,1.])))@Y.T@nominal_forces[local_index])[:,None]*yaw_dq[None,:]
     add({force_offset+3*local_index:cone@Y.T,70*(k+1):friction_dq},-np.inf,friction_dq@ne)
     add({force_offset+3*local_index:np.array([[0.,0.,1.]])},0.,np.inf)
    total_normal=np.zeros((1,3*count));total_normal[0,2::3]=1.
    add({force_offset:total_normal},0.,80.)
   add(velocity_parts,dv,dv)
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
   if nominal is not None:
    nominal_error=nominal['states'][k,:35].copy();nominal_velocity_error=nominal['states'][k,35:].copy()
   nq=integrate(rq,nominal_error);nv=rv+nominal_velocity_error;state=self.model.evaluate(nq,nv,kinematic_bias=True);mode=self.planned_mode(frames[k]);active=[j for j in [0,1] if mode&(1<<j)];reference_state=feet_objective(k,rq,nominal_error,state)
   dynamics=np.zeros((35,76));dynamics[:,:35]=state['mass'];dynamics[:,47:]=-np.r_[np.zeros((6,29)),np.eye(29)]
   for side in [0,1]:dynamics[:,35+6*side:41+6*side]=-state['jacobian'][side].T
   from generalized_dynamics_linearization import derivatives
   nominal_acc=ra if nominal is None else nominal['controls'][k,:35]
   nominal_wrench=np.zeros(12) if nominal is None else nominal['controls'][k,35:47].copy()
   if nominal is None:
    for side in active:nominal_wrench[6*side+2]=sum(self.model.model.body_mass)*9.81/max(1,len(active))
   position_regularization=np.r_[[100000.]*3,[10000.]*3,[2000.]*29]
   velocity_regularization=np.r_[[1000.]*3,[100.]*3,[10.]*29]
   H[70*k:70*k+35]+=2*position_regularization;g[70*k:70*k+35]-=2*position_regularization*nominal_error
   H[70*k+35:70*k+70]+=2*velocity_regularization;g[70*k+35:70*k+70]-=2*velocity_regularization*nominal_velocity_error
   H[u:u+35]+=.04;g[u:u+35]-=.04*nominal_acc
   velocity_trust=np.r_[[.7]*3,[3.]*3,[6.]*29]
   add({70*k+35:np.eye(35)},nominal_velocity_error-velocity_trust,nominal_velocity_error+velocity_trust,group="future_velocity_trust")
   Dq,Dv=derivatives(self.model,nq,nv,nominal_acc,nominal_wrench)
   tangent=nominal_error[3:6];theta=np.linalg.norm(tangent);W=skew(tangent);left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
   Dq[:,3:6]=Dq[:,3:6]@left
   rhs=-state['bias']+Dq@nominal_error+Dv@nominal_velocity_error
   add({u:dynamics,70*k:Dq,70*k+35:Dv},rhs,rhs);add({u+47:np.eye(29)},-self.effort,self.effort,group="future_effort")
   trust=np.r_[[.04]*3,[.12]*3,[.2]*29]
   add({70*k:np.eye(35)},nominal_error-trust,nominal_error+trust,group="future_trust")
   for side in [0,1]:
    foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=self.geometry[side];J=state['jacobian'][side]
    for point_local,radius in zip(points,radii):
     offset=R@point_local;Jpoint=J[:3]-skew(offset)@J[3:];Jpoint[:,3:6]=Jpoint[:,3:6]@left
     height=foot[2]+offset[2]-radius
     add({70*k:Jpoint[2:3]},swing_floor(reference_state,side,mode)-height+Jpoint[2]@nominal_error,np.inf,group="future_ground")

   for side in [0,1]:
    if side not in active:add({u+35+6*side:np.eye(6)},0.,0.);continue
    foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=self.geometry[side];z=foot[2]+(points@R.T)[:,2]-radii;lowest=np.flatnonzero(z<=z.min()+.0005);index=int(z.argmin());local=points[index].copy();local-=R.T@np.array([0.,0.,radii[index]])
    contact_patch_indices.setdefault(k,{})[side]=lowest.tolist()
    if side not in previous_active:
     reference_state=self.model.evaluate(rq,rv);reference_foot=reference_state['feet'][side];reference_R=Rotation.from_quat(reference_foot[3:]).as_matrix()
     anchor=reference_foot[:3]+reference_R@local;anchor[2]=0.;anchors[side]=(anchor,local)
    # The native foot may roll from heel to toe. Use the current lowest
    # surface for both wrench and no-slip velocity, rather than pinning a
    # material point that can cease to touch the floor.
    offset=R@local;J=state['jacobian'][side];Jp=J[:3]-skew(offset)@J[3:];point=foot[:3]+offset
    # Contact position/velocity tolerances cover discrete linearization error.
    tangent=nominal_error[3:6];theta=np.linalg.norm(tangent);W=skew(tangent);left=np.eye(3)+(1-np.cos(theta))/max(theta**2,1e-16)*W+(theta-np.sin(theta))/max(theta**3,1e-16)*(W@W)
    Jq=Jp.copy();Jq[:,3:6]=Jp[:,3:6]@left
    target=-point[2]+Jq[2]@nominal_error
    add({70*k:Jq[2:3]},target-.0005,target+.0005,group="future_contact_position")
    # Every point used to transmit force must stay on the floor. Without
    # these upper bounds a point can leave contact while its moment capacity
    # remains in the frozen linearized wrench cone.
    for contact_index in lowest:
     contact_offset=R@points[contact_index]-np.array([0.,0.,radii[contact_index]])
     Jheight=(J[:3]-skew(contact_offset)@J[3:])[2:3].copy();Jheight[:,3:6]=Jheight[:,3:6]@left
     height=foot[2]+contact_offset[2]
     add({70*k:Jheight},-np.inf,.0005-height+Jheight@nominal_error,group="future_contact_position")
    # Velocity also changes with configuration, especially during rolling.
    from generalized_dynamics_linearization import perturb
    patch_locals=points[lowest]-(np.array([0.,0.,1.])@R)[None,:]*radii[lowest,None]
    def patch_jacobian(rotation,jacobian):
     return np.vstack([jacobian[:3]-skew(rotation@patch_local)@jacobian[3:] for patch_local in patch_locals])
    def patch_bias(sample,rotation,jacobian,velocity):
     omega=jacobian[3:]@velocity;bias=sample['foot_bias'][side]
     return np.concatenate([bias[:3]-skew(rotation@point)@bias[3:]+np.cross(omega,np.cross(omega,rotation@point)) for point in patch_locals])
    Jpatch=patch_jacobian(R,J);velocity_dq=np.zeros((len(Jpatch),35));acceleration_dq=np.zeros_like(velocity_dq);acceleration_dv=np.zeros_like(velocity_dq);epsilon=1e-5
    for coordinate in range(35):
     delta=np.zeros(35);delta[coordinate]=epsilon;values=[];acceleration_values=[];velocity_acceleration_values=[]
     for sign in [1.,-1.]:
      ds=self.model.evaluate(perturb(nq,sign*delta),nv,kinematic_bias=True);df=ds['feet'][side];dR=Rotation.from_quat(df[3:]).as_matrix();dJ=ds['jacobian'][side]
      values.append(patch_jacobian(dR,dJ)@nv)
      acceleration_values.append(patch_jacobian(dR,dJ)@nominal_acc+patch_bias(ds,dR,dJ,nv))
      dv=nv+sign*delta;vs=self.model.evaluate(nq,dv,kinematic_bias=True)
      velocity_acceleration_values.append(Jpatch@nominal_acc+patch_bias(vs,R,J,dv))
     velocity_dq[:,coordinate]=(values[0]-values[1])/(2*epsilon)
     acceleration_dq[:,coordinate]=(acceleration_values[0]-acceleration_values[1])/(2*epsilon)
     acceleration_dv[:,coordinate]=(velocity_acceleration_values[0]-velocity_acceleration_values[1])/(2*epsilon)
    acceleration_dq[:,3:6]=acceleration_dq[:,3:6]@left
    acceleration_target=-patch_bias(state,R,J,nv)+acceleration_dq@nominal_error+acceleration_dv@nominal_velocity_error
    add({u:Jpatch,70*k:acceleration_dq,70*k+35:acceleration_dv},acceleration_target-.5,acceleration_target+.5,group="future_contact_acceleration")
    velocity_dq[:,3:6]=velocity_dq[:,3:6]@left
    velocity_target=-Jpatch@rv+velocity_dq@nominal_error
    add({70*k+35:Jpatch,70*k:velocity_dq},velocity_target-.025,velocity_target+.025,group="future_contact_velocity")
    # Use point forces so the derivative of r(q) x f is represented in
    # the SQP. A frozen aggregate wrench cone misses this geometry coupling.
    count=len(lowest);force_offset=n;n+=3*count;H=np.r_[H,np.full(3*count,2e-7)];g=np.r_[g,np.zeros(3*count)]
    contact_force_blocks.append((k,side,lowest.tolist(),force_offset,count))
    nominal_forces=np.tile(nominal_wrench[6*side:6*side+3]/count,(count,1))
    if nominal is not None:
     old=nominal.get('point_forces',{}).get(k,{}).get(side)
     if old is not None:
      mapping=dict(zip(old['indices'],old['forces']))
      for local_index,point_index in enumerate(lowest):
       if int(point_index) in mapping:nominal_forces[local_index]=mapping[int(point_index)]
    B=np.zeros((6,3*count));Dq_force=np.zeros((6,35));angular_map=J[3:].copy();angular_map[:,3:6]=angular_map[:,3:6]@left
    for local_index,point_index in enumerate(lowest):
     rotated=R@points[point_index];point_offset=rotated-np.array([0.,0.,radii[point_index]])
     B[:3,3*local_index:3*local_index+3]=np.eye(3);B[3:,3*local_index:3*local_index+3]=skew(point_offset)
     Dq_force[3:]+=skew(nominal_forces[local_index])@skew(rotated)@angular_map
    rhs=-Dq_force@nominal_error
    add({u+35+6*side:np.eye(6),force_offset:-B,70*k:-Dq_force},rhs,rhs,group="future_wrench")
    yaw=Rotation.from_quat(foot[3:]).as_euler('xyz')[2];Y=Rotation.from_euler('z',yaw).as_matrix();column=R[:,0]
    yaw_gradient=np.array([-column[0]*column[2],-column[1]*column[2],column[0]**2+column[1]**2])/max(column[0]**2+column[1]**2,1e-8)
    yaw_dq=yaw_gradient@angular_map;cone=np.array([[sx,sy,-.5] for sx in [-1.,1.] for sy in [-1.,1.]])
    for local_index in range(count):
     friction_dq=(cone@(-skew(np.array([0.,0.,1.])))@Y.T@nominal_forces[local_index])[:,None]*yaw_dq[None,:]
     add({force_offset+3*local_index:cone@Y.T,70*k:friction_dq},-np.inf,friction_dq@nominal_error,group="future_wrench")
     add({force_offset+3*local_index:np.array([[0.,0.,1.]])},0.,np.inf,group="future_wrench")
    total_normal=np.zeros((1,3*count));total_normal[0,2::3]=1.
    add({force_offset:total_normal},0.,800.,group="future_wrench")
   previous_active=set(active)
  def padded(row):return sparse.hstack([row,sparse.csc_matrix((row.shape[0],n-row.shape[1]))],format='csc')
  rows=[padded(row) for row in rows];soc=[(padded(row),bound) for row,bound in soc]
  C=sparse.vstack(rows,format='csc');low=np.array(lo);high=np.array(hi);eq=np.isfinite(low)&np.isfinite(high)&(abs(high-low)<1e-12);up=np.isfinite(high)&~eq;down=np.isfinite(low)&~eq
  A=sparse.vstack([C[eq],C[up],-C[down]]+[s[0] for s in soc],format='csc');b=np.r_[high[eq],high[up],-low[down],*[s[1] for s in soc]];cones=[clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(up.sum()+down.sum()))]+[clarabel.SecondOrderConeT(len(s[1])) for s in soc]
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=150;settings.tol_gap_abs=1e-7;settings.tol_gap_rel=1e-7;settings.tol_feas=1e-7
  objective_hessian=sparse.diags(H,format='lil')
  for offset,block in hessian_blocks:objective_hessian[offset:offset+35,offset:offset+35]+=block
  solution=clarabel.DefaultSolver(sparse.triu(objective_hessian.tocsc(),format='csc'),g,A,b,cones,settings).solve();x=np.array(solution.x);linear_violation=float(max(np.max(low-C@x),np.max(C@x-high),0.));cone_violation=max(float(np.linalg.norm(s[1][1:]-s[0][1:]@x)-(s[1][0]-s[0][0]@x)[0]) for s in soc);ok=str(solution.status) in ('Solved','AlmostSolved') and linear_violation<1e-4 and cone_violation<1e-5
  point_forces={}
  for stage,side,indices,offset,count in contact_force_blocks:
   point_forces.setdefault(stage,{})[side]={'indices':indices,'forces':x[offset:offset+3*count].reshape(count,3).tolist()}
  impulse_point_forces={}
  for stage,side,indices,offset,count in impulse_force_blocks:
   impulse_point_forces.setdefault(stage,{})[side]={'indices':indices,'forces':x[offset:offset+3*count].reshape(count,3).tolist()}
  return dict(impulse_point_forces=impulse_point_forces,point_forces=point_forces,elastic_slacks={group:float(max(np.max(abs(x[offset:offset+size])) for label,offset,size in elastic_blocks if label==group)) for group in set(row[0] for row in elastic_blocks)},success=ok,status=str(solution.status),linear_violation=linear_violation,tube_violation=cone_violation,first_control=x[nx:nx+76],controls=x[nx:impulse_start].reshape(N,76),impulses=x[impulse_start:impulse_start+12*N].reshape(N,12),impact_maps=impact_maps,patch_impact_nodes=patch_impact_nodes,contact_patch_indices=contact_patch_indices,impact_patch_indices=impact_patch_indices,states=x[:nx].reshape(N+1,70),frames=frames,objective=float(solution.obj_val))
 def nonlinear_audit(self,result,rate):
  force=moment=joint=contact_acceleration_error=wrench_error=impulse_wrench_error=support_gap=impact_residual=energy_increase=support_speed=integration_position=integration_orientation=integration_joint=0.;clearance=float('inf')
  for k,frame in enumerate(result['frames']):
   if k==0:continue
   rq,rv,_=[x.numpy().ravel() for x in self.reference.kinematics(float(frame),rate)]
   q=integrate(rq,result['states'][k,:35]);v=rv+result['states'][k,35:];state=self.model.evaluate(q,v,kinematic_bias=True)
   prq,prv,_=[x.numpy().ravel() for x in self.reference.kinematics(float(result['frames'][k-1]),rate)]
   previous_q=integrate(prq,result['states'][k-1,:35]);previous_v=prv+result['states'][k-1,35:]
   integrated=integrate(previous_q,self.dt*previous_v+.5*self.dt**2*result['controls'][k-1,:35]);integration_error=difference(q,integrated)
   integration_position=max(integration_position,float(np.linalg.norm(integration_error[:3])));integration_orientation=max(integration_orientation,float(np.linalg.norm(integration_error[3:6])));integration_joint=max(integration_joint,float(abs(integration_error[6:]).max()))
   for fraction in [.25,.5,.75]:
    elapsed=self.dt*fraction;midq=integrate(previous_q,elapsed*previous_v+.5*elapsed**2*result['controls'][k-1,:35]);midstate=self.model.evaluate(midq,previous_v)
    for side in [0,1]:
     points,radii=self.geometry[side];foot=midstate['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix()
     clearance=min(clearance,float(foot[2]+np.min((points@R.T)[:,2]-radii)))
   for side in [0,1]:
    points,radii=self.geometry[side];foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();clearance=min(clearance,float(foot[2]+np.min((points@R.T)[:,2]-radii)))
    from native_patch_wrench import native_patch_wrench
    heights=(points@R.T)[:,2]-radii;near=np.asarray(result.get('contact_patch_indices',{}).get(k,{}).get(side,np.flatnonzero(heights<=heights.min()+.0015)),dtype=int);patch_offsets=points[near]@R.T-radii[near,None]*np.array([0.,0.,1.])
    yaw=Rotation.from_quat(foot[3:]).as_euler('xyz')[2];Y=Rotation.from_euler('z',yaw).as_matrix()
    if k<self.N:
     E,G,b=native_patch_wrench(patch_offsets,.5,800.,Y);w=result['controls'][k,35+6*side:41+6*side]
     wrench_error=max(wrench_error,float(np.max(abs(E@w),initial=0.)),float(np.max(G@w-b,initial=0.)))
    if k-1 in result['impact_maps']:
     impact_indices=np.asarray(result.get('impact_patch_indices',{}).get(k-1,{}).get(side,near),dtype=int)
     impact_offsets=points[impact_indices]@R.T-radii[impact_indices,None]*np.array([0.,0.,1.])
     E,G,b=native_patch_wrench(impact_offsets,.5,80.,Y);w=result['impulses'][k-1,6*side:6*side+6]
     if w[2]>1e-3:support_gap=max(support_gap,float(foot[2]+heights[impact_indices].max()))
     impulse_wrench_error=max(impulse_wrench_error,float(np.max(abs(E@w),initial=0.)),float(np.max(G@w-b,initial=0.)))
    if k<self.N and result['controls'][k,37+6*side]>20:
     support_gap=max(support_gap,float(foot[2]+heights[near].max()))
     J=state['jacobian'][side]
     for index in near:
      offset=R@points[index]-np.array([0.,0.,radii[index]])
      support_speed=max(support_speed,float(np.linalg.norm((J[:3]-skew(offset)@J[3:])@v)))
      omega=J[3:]@v;bias=state['foot_bias'][side];point_bias=bias[:3]-skew(offset)@bias[3:]+np.cross(omega,np.cross(omega,offset))
      contact_acceleration_error=max(contact_acceleration_error,float(np.linalg.norm((J[:3]-skew(offset)@J[3:])@result['controls'][k,:35]+point_bias)))
   if k-1 in result['impact_maps']:
    previous_frame=result['frames'][k-1];prq,prv,_=[x.numpy().ravel() for x in self.reference.kinematics(float(previous_frame),rate)]
    pre_v=prv+result['states'][k-1,35:]+self.dt*result['controls'][k-1,:35]
    impulse=result['impulses'][k-1];momentum=state['mass']@(v-pre_v)
    for side in [0,1]:momentum-=state['jacobian'][side].T@impulse[6*side:6*side+6]
    impact_residual=max(impact_residual,float(np.linalg.norm(momentum)))
    energy_increase=max(energy_increase,float(.5*(v@state['mass']@v-pre_v@state['mass']@pre_v)))
   if k>=self.N:continue
   control=result['controls'][k];res=state['mass']@control[:35]+state['bias'];res[6:]-=control[47:]
   for side in [0,1]:res-=state['jacobian'][side].T@control[35+6*side:41+6*side]
   force=max(force,float(np.linalg.norm(res[:3])));moment=max(moment,float(np.linalg.norm(res[3:6])));joint=max(joint,float(abs(res[6:]).max()))
  return dict(nonlinear_base_force_residual=force,nonlinear_base_moment_residual=moment,nonlinear_joint_residual=joint,nonlinear_minimum_clearance=clearance,nonlinear_support_gap=support_gap,nonlinear_impact_residual=impact_residual,nonlinear_impact_energy_increase=energy_increase,nonlinear_support_speed=support_speed,nonlinear_contact_acceleration=contact_acceleration_error,nonlinear_wrench_cone_violation=wrench_error,nonlinear_impulse_cone_violation=impulse_wrench_error,nonlinear_integration_position=integration_position,nonlinear_integration_orientation=integration_orientation,nonlinear_integration_joint=integration_joint,nonlinear_verified=contact_acceleration_error<2. and wrench_error<.2 and impulse_wrench_error<.02 and support_speed<.15 and integration_orientation<.02 and integration_position<1e-4 and integration_joint<1e-4 and impact_residual<.5 and energy_increase<.1 and force<20 and moment<5 and joint<5 and clearance>-.002 and support_gap<.0015)
 def refine(self,q,v,frame,rate,first,iterations=20):
  nominal=None;history=[];numerical_recovery=set()
  for iteration in range(iterations):
   # The guessed decay trajectory is not a feasible SQP seed. Bootstrap
   # without its position trust box, then use the previous feasible solution.
   saved_skip=getattr(self,'diagnostic_skip',set()).copy()
   self.diagnostic_skip=saved_skip|numerical_recovery|({'future_trust'} if iteration==0 else set())
   self.elastic_groups=set()
   try:
    result=self.solve(q,v,frame,rate,first,nominal=nominal)
    if not result['success'] and 'future_velocity_trust' not in self.diagnostic_skip:
     numerical_recovery.add('future_velocity_trust');self.diagnostic_skip.add('future_velocity_trust')
     result=self.solve(q,v,frame,rate,first,nominal=nominal)
    if not result['success']:
     self.elastic_groups={'future_ground','future_contact_position','future_contact_velocity','future_contact_acceleration'}
     self.elastic_scale=10.**min(iteration,4)
     result=self.solve(q,v,frame,rate,first,nominal=nominal)
   finally:
    self.diagnostic_skip=saved_skip;self.elastic_groups=set()
   result['numerical_velocity_trust_recovered']=bool(numerical_recovery)
   if not result['success']:
    history.append(dict(iteration=iteration,status=result['status']));result['refinement_history']=history;return result
   audit=self.nonlinear_audit(result,rate);result.update(audit);history.append(dict(iteration=iteration,status=result['status'],elastic_slacks=result.get('elastic_slacks',{}),**audit));result['refinement_history']=history
   if audit['nonlinear_verified'] and not result['elastic_slacks']:return result
   nominal=result
  result['success']=False;result['status']='NonlinearPredictionRejected';return result
