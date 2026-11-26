"""Experimental simultaneous centroidal/kinematic trajectory projection.
This edits only a candidate control reference. Original source audit is immutable.
No claim of physical tracking follows from an offline solution.
"""
from pathlib import Path
import argparse,json
import numpy as np,torch,warp as wp,osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from test_g1_two_tasks import load_motion_trajectory,G1_JOINT_NAMES
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=480);p.add_argument('--iterations',type=int,default=15);args=p.parse_args()
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
root=Path(__file__).parent
raw,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'))
goal=np.load(root.parent/'rolling_reference/rolling.foot_targets.npz')
frames=np.arange(0,min(args.frames,len(raw)),2);N=len(frames);dt=2/fps
f=G1PoseFabric(N,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');model=f.dynamics_tasks.model
ids=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
with torch.no_grad():
 st=model.snapshot(raw[frames]);corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
 shift=-np.min(st['transforms'][0,ids,2,None].numpy()+(st['rotations'][0,ids].numpy()@corners.T)[:,2,:])
source=raw[frames].numpy().copy();source[:,2]+=shift;q=source.copy();source_R=Rotation.from_quat(source[:,3:7])
contact=goal['contact'][frames];support=goal['support'][frames];cones=goal['cones'][frames];bounds=goal['cone_bounds'][frames]
foot_goal=goal['position'][frames];rot_goal=goal['rotation'][frames]
mass=float(f.center_of_mass_taskmap.link_masses.sum());mg=mass*9.81
nx=35*N;nu=12*N;ns=6*(N-2);n=nx+nu+ns
wrench=np.zeros((N,2,6));wrench[:,:,2]=contact/contact.sum(1,keepdims=True)*mg
import xml.etree.ElementTree as ET
limits={e.attrib['name']:e.find('limit') for e in ET.parse(f.urdf_path).getroot().findall('joint')}
lo_joint=np.array([float(limits[k].attrib['lower']) for k in G1_JOINT_NAMES]);hi_joint=np.array([float(limits[k].attrib['upper']) for k in G1_JOINT_NAMES])
rows=[]
def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def snapshot(q):
 with torch.no_grad():return {k:v.numpy().astype(float) for k,v in model.snapshot(torch.tensor(q,dtype=torch.float32)).items()}
def velocities(q):
 v=np.zeros((N-1,35));v[:,:3]=np.diff(q[:,:3],axis=0)/dt;v[:,6:]=np.diff(q[:,7:],axis=0)/dt
 v[:,3:6]=(Rotation.from_quat(q[1:,3:7])*Rotation.from_quat(q[:-1,3:7]).inv()).as_rotvec()/dt
 return v
def residual(st,q,w):
 c=st['com'];A=st['momentum_matrix'];H=.5*(A[1:]+A[:-1])@velocities(q)[...,None];H=H[...,0]
 force=w[1:-1,:,:3].sum(1);moment=np.cross(support[1:-1]-c[1:-1,None],w[1:-1,:,:3]).sum(1)+w[1:-1,:,3:].sum(1)
 return np.c_[np.diff(c,n=2,axis=0)/dt**2-force/mass-[0,0,-9.81],np.diff(H,axis=0)/dt-moment]
for iteration in range(args.iterations):
 st=snapshot(q);c=st['com'];Jc=st['com_jac'];A=st['momentum_matrix'];Am=.5*(A[1:]+A[:-1]);H=(Am@velocities(q)[...,None])[...,0]
 offset=np.c_[q[:,:3]-source[:,:3],(Rotation.from_quat(q[:,3:7])*source_R.inv()).as_rotvec(),q[:,7:]-source[:,7:]]
 # Sparse least-squares objective; all variables are local configuration increments,
 # world wrenches divided by body weight, and elastic dynamics residuals.
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
 for k in range(N):
  j=35*k;u=nx+12*k
  append({j:np.eye(35)},-offset[k],np.r_[np.full(3,30.),np.full(3,20.),np.full(12,5.),np.full(17,10.)])
  append({u:np.eye(12)},wrench[k].ravel()/mg,.03)
  pe=(foot_goal[k]-st['transforms'][k,ids,:3]).ravel()
  re=(Rotation.from_matrix(rot_goal[k])*Rotation.from_matrix(st['rotations'][k,ids]).inv()).as_rotvec().ravel()
  append({j:st['origin_jac'][k,ids].reshape(6,35)},pe,1000.)
  append({j:st['angular_jac'][k,ids].reshape(6,35)},re,100.)
  trust=np.r_[np.full(3,.02),np.full(3,.08),np.full(29,.15)]
  low=np.maximum(-trust,np.r_[np.full(3,-.15),np.full(3,-.25),np.maximum(lo_joint-source[k,7:],-.6)]-offset[k])
  high=np.minimum(trust,np.r_[np.full(3,.15),np.full(3,.25),np.minimum(hi_joint-source[k,7:],.6)]-offset[k])
  if k<2:low=-offset[k];high=-offset[k]
  append({j:np.eye(35)},lo=low,hi=high)
  for side in range(2):
   if contact[k,side]:append({u+6*side:cones[k,side]},lo=-np.inf,hi=bounds[k,side]/mg)
   else:append({u+6*side:np.eye(6)},lo=0.,hi=0.)
  if k and k<N-1:
   # Smooth corrections rather than suppressing source acceleration itself.
   append({j-35:np.eye(35),j:-2*np.eye(35),j+35:np.eye(35)},-(offset[k-1]-2*offset[k]+offset[k+1]),10.)
   d=np.zeros(6);d[:3]=np.array([0,0,-9.81])*dt**2-(c[k-1]-2*c[k]+c[k+1]);d[3:]=-dt*(H[k]-H[k-1])
   B=np.zeros((6,12));B[:3,:3]=B[:3,6:9]=-dt**2*mg/mass*np.eye(3)
   for side in range(2):B[3:,6*side:6*side+3]=-dt**2*mg*skew(support[k,side]-c[k]);B[3:,6*side+3:6*side+6]=-dt**2*mg*np.eye(3)
   middle=np.vstack((-2*Jc[k],-Am[k]-Am[k-1]-dt**2*skew(wrench[k,:,:3].sum(0))@Jc[k]))
   append({j-35:np.vstack((Jc[k-1],Am[k-1])),j:middle,j+35:np.vstack((Jc[k+1],Am[k])),u:B,nx+nu+6*(k-1):np.eye(6)},lo=d,hi=d)
   append({nx+nu+6*(k-1):np.eye(6)},0.,10000.*min(10.,1.+iteration))
 O=sparse.csc_matrix((ov,(oi,oj)),shape=(oc,n));C=sparse.csc_matrix((cv,(ci,cj)),shape=(cc,n))
 P=2*(O.T@O)+sparse.eye(n,format='csc')*1e-7;h=-2*(O.T@np.array(ot))
 solver=osqp.OSQP();solver.setup(P=sparse.triu(P,format='csc'),q=h,A=C,l=np.array(lower),u=np.array(upper),eps_abs=1e-5,eps_rel=1e-5,max_iter=40000,polish=True,verbose=False)
 result=solver.solve()
 if result.x is None or result.info.status_val not in (1,2):raise RuntimeError('SQP subproblem '+result.info.status)
 delta=result.x[:nx].reshape(N,35);q[:,:3]+=delta[:,:3];q[:,7:]+=delta[:,6:];q[:,3:7]=(Rotation.from_rotvec(delta[:,3:6])*Rotation.from_quat(q[:,3:7])).as_quat();wrench=result.x[nx:nx+nu].reshape(N,2,6)*mg
 new=snapshot(q);r=residual(new,q,wrench);pe=np.linalg.norm(new['transforms'][:,ids,:3]-foot_goal,axis=-1)
 report=dict(iteration=iteration,qp_status=result.info.status,peak_com_acc_residual=float(np.max(np.abs(r[:,:3]))),peak_momentum_rate_residual=float(np.max(np.abs(r[:,3:]))),rms_com_acc_residual=float(np.sqrt(np.mean(r[:,:3]**2))),rms_momentum_rate_residual=float(np.sqrt(np.mean(r[:,3:]**2))),peak_foot_position_residual=float(pe.max()),peak_reference_joint_rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=-1)).max()),max_update=float(np.max(np.abs(delta))))
 rows.append(report);print(json.dumps(report),flush=True)
 np.savez_compressed(root/'candidate.npz',q=q,frames=frames,source=source,wrench=wrench,dynamics_residual=r,foot_target=foot_goal)
 (root/'candidate.json').write_text(json.dumps(dict(iterations=rows,source_frames=len(raw),optimized_frames=frames.tolist(),full_tracking_passed=False),indent=2)+'\n')
