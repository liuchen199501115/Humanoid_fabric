"""Experimental C2 full-body collocation. Never changes source data or audit gates."""
from pathlib import Path
import argparse,json
import numpy as np,torch,warp as wp,clarabel
from scipy import sparse
from rotation_basis import rotation_variations
from scipy.spatial.transform import Rotation
from scipy.interpolate import BSpline,make_interp_spline,PchipInterpolator
from test_g1_two_tasks import load_motion_trajectory,G1_JOINT_NAMES,unitree_g1_effort_limits
from g1_armature import unitree_armatures
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=480);p.add_argument('--iterations',type=int,default=18);args=p.parse_args()
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();root=Path(__file__).parent
raw,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'));N=min(args.frames,len(raw));frames=np.arange(N);dt=.01
f=G1PoseFabric(N,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');model=f.dynamics_tasks.model
ids=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
def snapshot(q):
 with torch.no_grad():return {k:v.numpy().astype(float) for k,v in model.snapshot(torch.tensor(q,dtype=torch.float32)).items()}
def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
source=raw[:N].numpy().astype(float).copy();st=snapshot(source);corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
shift=-np.min(st['transforms'][0,ids,2,None]+(st['rotations'][0,ids]@corners.T)[:,2,:]);source[:,2]+=shift
for k in range(1,N):
 if source[k-1,3:7]@source[k,3:7]<0:source[k,3:7]*=-1
source_R=Rotation.from_quat(source[:,3:7]);initial_derivative=PchipInterpolator(frames,source)(0,1)
nodes=np.unique(np.r_[np.arange(0,N,2),N-1]);spline=make_interp_spline(nodes,source[nodes],k=3,bc_type=([(1,initial_derivative)],[(2,np.zeros(36))]));coeff=spline.c.copy();knots=spline.t;K=len(coeff)
B=BSpline(knots,np.eye(K),3)(frames);D=BSpline(knots,np.eye(K),3)(frames,1)*fps;E=BSpline(knots,np.eye(K),3)(frames,2)*fps**2
near=[np.flatnonzero((np.abs(B[k])+np.abs(D[k])+np.abs(E[k]))>1e-10) for k in frames]
def sample(c):
 value=B@c;first=D@c;second=E@c;u=value[:,3:7];du=first[:,3:7];ddu=second[:,3:7];norm=np.linalg.norm(u,axis=1,keepdims=True);quat=u/norm
 nv=np.sum(quat*du,axis=1,keepdims=True);qv=(du-quat*nv)/norm;na=(np.sum(du*du+u*ddu,axis=1,keepdims=True)-nv*nv)/norm;qa=(ddu-quat*na-2*qv*nv)/norm
 def angular(d):return 2*(quat[:,3:]*d[:,:3]-d[:,3:]*quat[:,:3]+np.cross(quat[:,:3],d[:,:3]))
 value[:,3:7]=quat;return value,np.c_[first[:,:3],angular(qv),first[:,7:]],np.c_[second[:,:3],angular(qa),second[:,7:]]
goal=np.load(root/'retimed.foot_targets.npz');contact=goal['contact'][:N];support=goal['support'][:N];cones=goal['cones'][:N];bounds=goal['cone_bounds'][:N];foot_goal=goal['position'][:N].copy();rot_goal=goal['rotation'][:N]
geometry=json.loads(Path('acc_data/amp_walk_20261102/contact_clock_followup/vertical_feedback_verified.steps_plan.json').read_text());points=np.array(geometry['collision_points']);radii=np.array(geometry['collision_radii'])
for side in range(2):
 active=contact[:,side];cuts=np.r_[0,np.flatnonzero(active[1:]!=active[:-1])+1,N]
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if active[lo] or lo==0:continue
  for k in range(lo,hi):
   u=(k-lo)/max(1,hi-lo);bottom=np.min((rot_goal[k,side]@points[side].T)[2]-radii[side]);foot_goal[k,side,2]=max(foot_goal[k,side,2],.056*16*u*u*(1-u)**2-bottom)
mass=float(f.center_of_mass_taskmap.link_masses.sum());mg=mass*9.81;armature=unitree_armatures(G1_JOINT_NAMES)[0];effort=.6*unitree_g1_effort_limits()
exec((root/'safe_spline_dynamics.inc').read_text())
import xml.etree.ElementTree as ET
limits={e.attrib['name']:e.find('limit') for e in ET.parse(f.urdf_path).getroot().findall('joint')};lo_joint=np.array([float(limits[k].attrib['lower']) for k in G1_JOINT_NAMES]);hi_joint=np.array([float(limits[k].attrib['upper']) for k in G1_JOINT_NAMES]);speed=np.array([min(30.,float(limits[k].attrib['velocity'])) for k in G1_JOINT_NAMES])
nx=35*K;nu=12*N;n=nx+nu+6*N;rows=[]
wrench=np.zeros((N,2,6));wrench[:,:,2]=contact/contact.sum(1,keepdims=True)*mg
initial_velocity=sample(coeff)[1][0].copy()
warm=root/'safe_spline_candidate.npz'
if warm.exists():
 with np.load(warm) as data:
  if data['q'].shape[0]==N and np.allclose(data['knots'],knots):coeff=data['coefficients'].copy();wrench=data['wrench'].copy()
touch=[]
for k in range(N):
 touch.append([])
 for side in range(2):
  height=(rot_goal[k,side]@points[side].T)[2]-radii[side];touch[-1].append(np.flatnonzero(height<=height.min()+1e-6))
def update_support(st):
 for k in range(N):
  for side in range(2):
   if not contact[k,side]:continue
   link=ids[side]
   if goal['kind'][k,side]==0:support[k,side]=st['transforms'][k,link,:3];support[k,side,2]=0.
   else:
    offsets=(st['rotations'][k,link]@points[side,touch[k][side]].T).T
    support[k,side]=st['transforms'][k,link,:3]+offsets.mean(0)-np.array([0,0,radii[side,touch[k][side]].mean()])
exec((root/'persistent_collision_planner.inc').read_text())
exec((root/'safe_collision_derivatives.inc').read_text())
for iteration in range(args.iterations):
 rotation_maps=rotation_variations(coeff,B,D,E,near)
 q,v,acc=sample(coeff);st=snapshot(q);update_support(st);M,h,J,_=dynamics(st,q,v,acc);g=(M@acc[...,None])[...,0]+h
 collision=collision_linearization(st,q,v,acc)
 collision_selection=collision["selection"]
 Rq,Rv,Cq,collision_Rq=full_derivatives(st,q,v,acc,wrench)
 print('Full dynamics and contact linearization complete',iteration,flush=True)
 configuration_offset=np.c_[q[:,:3]-source[:,:3],(Rotation.from_quat(q[:,3:7])*source_R.inv()).as_rotvec(),q[:,7:]-source[:,7:]]
 oi=[];oj=[];ov=[];ot=[];oc=0;ci=[];cj=[];cv=[];lower=[];upper=[];cc=0
 def append(blocks,target=None,weight=1.,lo=None,hi=None):
  global oc,cc
  m=next(iter(blocks.values())).shape[0]
  if target is not None:
   for col,block in blocks.items():
    a,b=np.nonzero(block);oi.extend((a+oc).tolist());oj.extend((b+col).tolist());ov.extend((block[a,b]*np.broadcast_to(weight,(m,))[a]).tolist())
   ot.extend((np.broadcast_to(target,(m,))*weight).tolist());oc+=m
  else:
   for col,block in blocks.items():
    a,b=np.nonzero(block);ci.extend((a+cc).tolist());cj.extend((b+col).tolist());cv.extend(block[a,b].tolist())
   lower.extend(np.broadcast_to(lo,(m,)).tolist());upper.extend(np.broadcast_to(hi,(m,)).tolist());cc+=m
 def blocks(k,A,derivative=0):
  basis=(B,D,E)[derivative];result={}
  for j in near[k]:
   matrix=basis[k,j]*A.copy();matrix[:,3:6]=A[:,3:6]@rotation_maps[k][j][derivative]
   if np.any(matrix):result[35*j]=matrix
  return result
 def merge_blocks(*groups):
  result={}
  for group in groups:
   for col,block in group.items():result[col]=result.get(col,0.)+block
  return result
 for j in range(K):
  trust=np.r_[np.full(3,.015),np.full(3,.06),np.full(29,.12)];append({35*j:np.eye(35)},lo=-trust,hi=trust)
 for k in range(N):
  u=nx+12*k
  velocity_trust=np.r_[np.full(3,.2),np.full(3,1.),np.full(29,3.)];append(blocks(k,np.eye(35),1),lo=-velocity_trust,hi=velocity_trust)
  for pair in range(collision_selection.shape[1]):
   if not collision['selected_active'][k,pair]:continue
   cb=merge_blocks(blocks(k,collision['selected_A'][k,pair:pair+1],2),blocks(k,collision['selected_Rv'][k,pair:pair+1],1),blocks(k,collision_Rq[k,pair:pair+1]))
   append(cb,lo=.2-collision['selected_residual'][k,pair],hi=np.inf)
   append(blocks(k,collision['selected_A'][k,pair:pair+1]),lo=.025-collision['distance'][k,collision_selection[k,pair]],hi=np.inf)
  append(blocks(k,np.eye(35)),-configuration_offset[k],np.r_[np.full(3,30.),np.full(3,20.),np.full(12,5.),np.full(17,10.)])
  append(blocks(k,np.eye(35),2),-acc[k],.001)
  append({u:np.eye(12)},wrench[k].ravel()/mg,.03)
  pe=(foot_goal[k]-st['transforms'][k,ids,:3]).ravel();re=(Rotation.from_matrix(rot_goal[k])*Rotation.from_matrix(st['rotations'][k,ids]).inv()).as_rotvec().ravel()
  append(blocks(k,st['origin_jac'][k,ids].reshape(6,35)),pe,1000.);append(blocks(k,st['angular_jac'][k,ids].reshape(6,35)),re,100.)
  low=np.r_[np.full(3,-.15),np.full(3,-.25),np.maximum(lo_joint-source[k,7:],-.6)]-configuration_offset[k];high=np.r_[np.full(3,.15),np.full(3,.25),np.minimum(hi_joint-source[k,7:],.6)]-configuration_offset[k]
  if k==0:
   initial_band=np.r_[np.full(3,.003),np.full(3,.01),np.full(29,.1)];low=-initial_band-configuration_offset[k];high=initial_band-configuration_offset[k]
  append(blocks(k,np.eye(35)),lo=low,hi=high)
  append(blocks(k,np.eye(35)[6:],1),lo=-speed-v[k,6:],hi=speed-v[k,6:])
  if k==0:
   velocity_band=np.r_[np.full(3,.1),np.full(3,.2),np.full(29,1.)];append(blocks(k,np.eye(35),1),lo=initial_velocity-v[k]-velocity_band,hi=initial_velocity-v[k]+velocity_band)
   append(blocks(k,np.eye(35),1),initial_velocity-v[k],10.)
  for side in range(2):
   if contact[k,side]:
    append({u+6*side:cones[k,side]},lo=-np.inf,hi=bounds[k,side]/mg)
    link=ids[side];R=st['rotations'][k,link];origin=st['transforms'][k,link,:3];jv=st['origin_jac'][k,link];jw=st['angular_jac'][k,link]
    for point in range(len(points[side])):
     center=R@points[side,point];height=origin[2]+center[2]-radii[side,point];height_jac=(jv-skew(center)@jw)[2:3]
     append(blocks(k,height_jac),lo=-.001-height,hi=np.inf)
     if point in touch[k][side]:
      append(blocks(k,height_jac),lo=-.0002-height,hi=.0002-height)
      offset=center-np.array([0,0,radii[side,point]]);contact_jac=jv-skew(offset)@jw;speed_now=contact_jac@v[k]
      append(merge_blocks(blocks(k,contact_jac,1),blocks(k,Cq[k,side,point])),lo=-.01-speed_now,hi=.01-speed_now)
      append(merge_blocks(blocks(k,contact_jac,1),blocks(k,Cq[k,side,point])),-speed_now,100.)
   else:append({u+6*side:np.eye(6)},lo=0.,hi=0.)
  db=merge_blocks(blocks(k,M[k,:6]*dt**2,2),blocks(k,Rq[k,:6]*dt**2),blocks(k,Rv[k,:6]*dt**2,1));db[u]=-dt**2*mg*J[k,:,:6].T;db[nx+nu+6*k]=np.eye(6);append(db,lo=-dt**2*g[k,:6],hi=-dt**2*g[k,:6])
  append({nx+nu+6*k:np.eye(6)},0.,10000.*min(10.,1.+iteration))
  tb=merge_blocks(blocks(k,M[k,6:]*dt**2,2),blocks(k,Rq[k,6:]*dt**2),blocks(k,Rv[k,6:]*dt**2,1));tb[u]=-dt**2*mg*J[k,:,6:].T;append(tb,lo=dt**2*(-effort-g[k,6:]),hi=dt**2*(effort-g[k,6:]))
 O=sparse.csc_matrix((ov,(oi,oj)),shape=(oc,n));C=sparse.csc_matrix((cv,(ci,cj)),shape=(cc,n));P=2*(O.T@O)+sparse.eye(n,format='csc')*1e-7;linear=-2*O.T@np.array(ot)
 lo=np.asarray(lower);hi=np.asarray(upper);eq=np.isfinite(lo)&np.isfinite(hi)&(lo==hi);up=np.isfinite(hi)&~eq;low=np.isfinite(lo)&~eq
 matrix=sparse.vstack((C[eq],C[up],-C[low]),format='csc');rhs=np.r_[hi[eq],hi[up],-lo[low]]
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.tol_gap_abs=1e-7;settings.tol_gap_rel=1e-7;settings.tol_feas=1e-8
 solver=clarabel.DefaultSolver(sparse.triu(P,format='csc'),np.asarray(linear),matrix,rhs,[clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(up.sum()+low.sum()))],settings);result=solver.solve()
 if str(result.status) not in ('Solved','AlmostSolved'):raise RuntimeError('Spline SQP: '+str(result.status))
 solution=np.asarray(result.x)
 alpha=.5;delta=alpha*solution[:nx].reshape(K,35);coeff[:,:3]+=delta[:,:3];coeff[:,7:]+=delta[:,6:];norm=np.linalg.norm(coeff[:,3:7],axis=1,keepdims=True);coeff[:,3:7]=(Rotation.from_rotvec(delta[:,3:6])*Rotation.from_quat(coeff[:,3:7])).as_quat()*norm
 wrench=(1-alpha)*wrench+alpha*solution[nx:nx+nu].reshape(N,2,6)*mg;q,v,acc=sample(coeff);new=snapshot(q);update_support(new);new_M,new_h,new_J,_=dynamics(new,q,v,acc);required=(new_M@acc[...,None])[...,0]+new_h-(new_J.transpose(0,2,1)@wrench.reshape(N,12,1))[...,0]
 contact_speeds=[];contact_heights=[]
 for k in range(N):
  for side in range(2):
   if not contact[k,side]:continue
   link=ids[side]
   for point in touch[k][side]:
    center=new['rotations'][k,link]@points[side,point];offset=center-np.array([0,0,radii[side,point]])
    contact_speeds.append((new['origin_jac'][k,link]-skew(offset)@new['angular_jac'][k,link])@v[k]);contact_heights.append(new['transforms'][k,link,2]+center[2]-radii[side,point])
 collision_audit=collision_linearization(new,q,v,acc)
 report=dict(iteration=iteration,collision_cut_count=sum(map(len,collision_cut_sets)),peak_collision_barrier_violation=float(max(0.,-collision_audit['residual'][collision_audit['active']].min())),minimum_collision_cover_distance=float(collision_audit['distance'][:,first_pair:].min()),peak_contact_speed=float(np.max(np.abs(contact_speeds))),peak_contact_height_error=float(np.max(np.abs(contact_heights))),qp_status=str(result.status),qp_iterations=int(result.iterations),peak_base_force_residual=float(abs(required[:,:3]).max()),peak_base_moment_residual=float(abs(required[:,3:6]).max()),rms_base_residual=float(np.sqrt(np.mean(required[:,:6]**2))),peak_torque_budget_ratio=float(np.max(abs(required[:,6:])/effort)),peak_foot_position_residual=float(np.linalg.norm(new['transforms'][:,ids,:3]-foot_goal,axis=-1).max()),peak_reference_joint_rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=-1)).max()),peak_joint_speed=float(abs(v[:,6:]).max()),peak_joint_acceleration=float(abs(acc[:,6:]).max()),max_update=float(abs(delta).max()))
 rows.append(report);print(json.dumps(report),flush=True);np.savez_compressed(root/'persistent_spline_candidate.npz',q=q,v=v,a=acc,coefficients=coeff,knots=knots,degree=3,frames=frames,source=source,wrench=wrench,required=required,foot_target=foot_goal,support_points=support.copy(),collision_barrier_residual=collision_audit["residual"])
 (root/'persistent_spline_candidate.json').write_text(json.dumps(dict(iterations=rows,optimized_source_frames=N,full_tracking_passed=False),indent=2)+'\n')
 if iteration>5 and abs(delta).max()<2e-5:break
