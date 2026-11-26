f.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True)
geometry_task=f.dynamics_tasks
sphere_ids=np.array(geometry_task.sphere_indices);sphere_centers=geometry_task.sphere_centers.numpy().astype(float);sphere_radii=geometry_task.sphere_radii.numpy().astype(float)
pair_ids=geometry_task.pairs.numpy();first_pair=geometry_task.original_collision_pairs;safe_distance=float(geometry_task.config['collision']['safe_distance'])
collision_cut_sets=[set() for _ in range(N)]
barrier_sides=np.array([0 if geometry_task.sphere_names[int(pair[0])].startswith('left_') else 1 for pair in pair_ids])

def collision_geometry_numpy(state,selection=None):
 R=state['rotations'][:,sphere_ids];offset=(R@sphere_centers[None,:,:,None])[...,0]
 positions=state['transforms'][:,sphere_ids,:3]+offset
 jac=state['origin_jac'][:,sphere_ids].copy();angular=state['angular_jac'][:,sphere_ids]
 # omega cross offset, with generalized velocities represented by columns.
 jac+=np.cross(angular.transpose(0,1,3,2),offset[:,:,None,:]).transpose(0,1,3,2)
 if selection is None:
  first,second=pair_ids.T;delta=positions[:,first]-positions[:,second];J=jac[:,first]-jac[:,second];radius=sphere_radii[first]+sphere_radii[second]
 else:
  pair=pair_ids[selection];batch=np.arange(N)[:,None];first,second=pair[:,:,0],pair[:,:,1]
  delta=positions[batch,first]-positions[batch,second];J=jac[batch,first]-jac[batch,second];radius=sphere_radii[first]+sphere_radii[second]
 norm=np.linalg.norm(delta,axis=-1).clip(1e-8);rows=np.einsum('npi,npij->npj',delta/norm[...,None],J)
 # All self distances are invariant to rigid global root motion.
 rows[:,:,:6]=0.
 return norm-radius,rows

def collision_linearization(state,pose,velocity,acceleration):
 def advance(sign):
  eps=sign*.001;out=pose.copy();out[:,:3]+=eps*velocity[:,:3];out[:,7:]+=eps*velocity[:,6:];out[:,3:7]=(Rotation.from_rotvec(eps*velocity[:,3:6])*Rotation.from_quat(pose[:,3:7])).as_quat();return out
 distance,A=collision_geometry_numpy(state);_,plus=collision_geometry_numpy(snapshot(advance(1)));_,minus=collision_geometry_numpy(snapshot(advance(-1)));derivative=(plus-minus)/.002
 speed=np.einsum('npj,nj->np',A,velocity);bias=np.einsum('npj,nj->np',derivative,velocity);residual=np.einsum('npj,nj->np',A,acceleration)+bias+16*speed+64*(distance-safe_distance)
 active=distance<.4;active[:,:first_pair]=False
 scored=np.where(active,residual,np.inf);selected=[]
 for side in range(2):
  choices=np.flatnonzero((barrier_sides==side)&(np.arange(len(pair_ids))>=first_pair));selected.append(choices[np.argsort(scored[:,choices],axis=1)[:,:12]]);selected.append(choices[np.argsort(distance[:,choices],axis=1)[:,:6]])
 proposed=np.concatenate(selected,axis=1)
 for k in range(N):
  collision_cut_sets[k].update(proposed[k].tolist());collision_cut_sets[k].update(np.flatnonzero(active[k]&(residual[k]<.5)).tolist())
 width=max(map(len,collision_cut_sets));selection=np.zeros((N,width),dtype=int);valid=np.zeros((N,width),dtype=bool)
 for k,cuts in enumerate(collision_cut_sets):
  indices=sorted(cuts);selection[k,:len(indices)]=indices;valid[k,:len(indices)]=True
 batch=np.arange(N)[:,None]
 return dict(distance=distance,A=A,bias=bias,derivative=derivative,residual=residual,active=active,selection=selection,selected_A=A[batch,selection],selected_Rv=(2*derivative+16*A)[batch,selection],selected_residual=residual[batch,selection],selected_active=active[batch,selection]&valid)
