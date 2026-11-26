def generalized_dynamics(st,q,v,acc):
 global last_bias_states
 def adv(eps):
  out=q.copy();out[:,:3]+=eps*v[:,:3];out[:,7:]+=eps*v[:,6:];out[:,3:7]=(Rotation.from_rotvec(eps*v[:,3:6])*Rotation.from_quat(q[:,3:7])).as_quat();return out
 plus=snapshot(adv(.001));minus=snapshot(adv(-.001));last_bias_states=(plus,minus);jv=st['link_com_jac'];jw=st['angular_jac'];R=st['rotations'];I=R@model.inertias.numpy()[None]@R.transpose(0,1,3,2);m=model.com_task.link_masses.numpy()
 jb=((plus['link_com_jac']-minus['link_com_jac'])/.002@v[:,None,:,None])[...,0];wb=((plus['angular_jac']-minus['angular_jac'])/.002@v[:,None,:,None])[...,0];omega=(jw@v[:,None,:,None])[...,0]
 force=m[None,:,None]*((jv@acc[:,None,:,None])[...,0]+jb+[0,0,9.81]);moment=(I@((jw@acc[:,None,:,None])[...,0]+wb)[...,None])[...,0]+np.cross(omega,(I@omega[...,None])[...,0])
 h=(jv.transpose(0,1,3,2)@force[...,None]+jw.transpose(0,1,3,2)@moment[...,None]).sum(1)[...,0]
 h[:,6:]+=armature*acc[:,6:]
 J=np.zeros((N,12,35))
 for side,link in enumerate(ids):
  jw=st['angular_jac'][:,link];offset=support[:,side]-st['transforms'][:,link,:3]
  J[:,6*side:6*side+3]=st['origin_jac'][:,link]+np.cross(jw.transpose(0,2,1),offset[:,None,:]).transpose(0,2,1)
  J[:,6*side+3:6*side+6]=jw
 return h,J
