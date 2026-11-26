def contact_velocity_maps(state,velocity):
 out=np.zeros((N,2,len(points[0]),3))
 for side,link in enumerate(ids):
  R=state['rotations'][:,link];offsets=np.einsum('nij,pj->npi',R,points[side]);offsets[:,:,2]-=radii[side][None]
  omega=(state['angular_jac'][:,link]@velocity[...,None])[...,0];linear=(state['origin_jac'][:,link]@velocity[...,None])[...,0]
  out[:,side]=linear[:,None]+np.cross(omega[:,None],offsets)
 return out

def bias_only(state,pose,velocity):
 def adv(sign):
  eps=sign*.001;out=pose.copy();out[:,:3]+=eps*velocity[:,:3];out[:,7:]+=eps*velocity[:,6:];out[:,3:7]=(Rotation.from_rotvec(eps*velocity[:,3:6])*Rotation.from_quat(pose[:,3:7])).as_quat();return out
 plus=snapshot(adv(1));minus=snapshot(adv(-1));jv=state['link_com_jac'];jw=state['angular_jac'];R=state['rotations'];I=R@model.inertias.numpy()[None]@R.transpose(0,1,3,2);mass_links=model.com_task.link_masses.numpy()
 jb=((plus['link_com_jac']-minus['link_com_jac'])/.002@velocity[:,None,:,None])[...,0];wb=((plus['angular_jac']-minus['angular_jac'])/.002@velocity[:,None,:,None])[...,0];omega=(jw@velocity[:,None,:,None])[...,0]
 forces=mass_links[None,:,None]*(jb+[0,0,9.81]);moments=(I@wb[...,None])[...,0]+np.cross(omega,(I@omega[...,None])[...,0])
 return (jv.transpose(0,1,3,2)@forces[...,None]+jw.transpose(0,1,3,2)@moments[...,None]).sum(1)[...,0]

def full_derivatives(state,pose,velocity,acceleration,wrenches):
 Rq=np.zeros((N,35,35));Rv=np.zeros_like(Rq);Cq=np.zeros((N,2,len(points[0]),3,35));barrier_q=np.zeros((N,collision_selection.shape[1],35));eps=.005
 for column in range(35):
  values=[];contact_values=[];barrier_values=[]
  for sign in (1,-1):
   trial=pose.copy()
   if column<3:trial[:,column]+=sign*eps
   elif column<6:
    rot=np.zeros((N,3));rot[:,column-3]=sign*eps;trial[:,3:7]=(Rotation.from_rotvec(rot)*Rotation.from_quat(pose[:,3:7])).as_quat()
   else:trial[:,column+1]+=sign*eps
   ts=snapshot(trial);update_support(ts);tm,th,tj,_=dynamics(ts,trial,velocity,acceleration)
   values.append((tm@acceleration[...,None])[...,0]+th-(tj.transpose(0,2,1)@wrenches.reshape(N,12,1))[...,0]);contact_values.append(contact_velocity_maps(ts,velocity))
   cd,cA=collision_geometry_numpy(ts,collision_selection);_,cp=collision_geometry_numpy(last_bias_states[0],collision_selection);_,cm=collision_geometry_numpy(last_bias_states[1],collision_selection);cb=np.einsum('npj,nj->np',(cp-cm)/.002,velocity);barrier_values.append(np.einsum('npj,nj->np',cA,acceleration+16*velocity)+cb+64*(cd-safe_distance))
  barrier_q[:,:,column]=(barrier_values[0]-barrier_values[1])/(2*eps)
  Rq[:,:,column]=(values[0]-values[1])/(2*eps);Cq[:,:,:,:,column]=(contact_values[0]-contact_values[1])/(2*eps)
 update_support(state)
 eps=.05
 for column in range(3,35):
  vp=velocity.copy();vm=velocity.copy();vp[:,column]+=eps;vm[:,column]-=eps
  Rv[:,:,column]=(bias_only(state,pose,vp)-bias_only(state,pose,vm))/(2*eps)
 barrier_q[:,:,:6]=0.;Cq[:,:,:,:,:3]=0.
 return Rq,Rv,Cq,barrier_q
