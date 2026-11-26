"""One offline whole-remaining-clip COM fit; runtime remains direct WBC.

Reduced centroidal feasibility is only a reference-preparation diagnostic.
Native contact, full-body torque and original PKL fidelity require simulation.
"""
from pathlib import Path
import json,pickle
import numpy as np,torch,osqp
from scipy import sparse
from scipy.interpolate import BSpline,CubicSpline
from scipy.spatial import ConvexHull
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent
parent=np.load(p/'wbc_convex_catch15_25_clear35.npz');base=parent['control_q']
saved=np.load(p/'wbc_twelve_reproduced.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
com=CenterOfMassModel(urdf,saved['final_joint_order']);leg=LegKinematics(urdf,saved['final_joint_order'])
with open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb') as f:contacts=np.asarray(pickle.load(f)['contact_sequence'])
frames=np.unique(np.r_[np.arange(700.,1999.,4.),1999.]);times=(frames-frames[0])/240.;N=len(frames)-1
q=torch.tensor(base);curve=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
qs=[];vs=[];states=[]
for f in frames:
 qi,vi,_=curve.kinematics(float(f),240.);qi=qi.numpy()[0];vi=vi.numpy()[0];qs.append(qi);vs.append(vi);states.append(com.momentum(qi,vi,return_state=True))
qs=np.array(qs);states=np.array(states);position,velocity,momentum=states[:,0],states[:,1],states[:,2]
height_ref=CubicSpline(times,position[:,2]);hdot=CubicSpline(times,momentum)(times,1)
knots=np.r_[np.full(4,0.),np.arange(.04,times[-1]-.001,.04),np.full(4,times[-1])];K=len(knots)-4
spline=BSpline(knots,np.eye(K),3);B=spline(times);D=spline(times,1);A=spline(times,2);J=spline(times,3)
H=2*(1e5*B.T@B+20*D.T@D+.0005*J.T@J)+np.eye(K)*1e-7;g=-2*(1e5*B.T@position[:,2]+20*D.T@velocity[:,2])
initial=np.array([position[0,2],velocity[0,2],height_ref(0.,2)])
print('vertical initial',initial.tolist(),flush=True)
C=sparse.csc_matrix(np.vstack((B,A,B[0],D[0],A[0])));lo=np.r_[position[:,2]-.045,np.full(len(times),-5.),initial];hi=np.r_[position[:,2]+.001,np.full(len(times),8.),initial]
solver=osqp.OSQP();solver.setup(P=sparse.triu(sparse.csc_matrix(H),format='csc'),q=g,A=C,l=lo,u=hi,eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polish=True,verbose=False);answer=solver.solve()
report=dict(source_frames=[700,1999],rate=1.2,vertical_status=answer.info.status,physical_tracking_certified=False,reduced_preflight_passed=False,planning='single offline reduced centroidal fit; no runtime predictive control')
if answer.info.status_val not in (1,2):
 (p/'wbc_remaining_com_plan.json').write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError(answer.info.status)
height_coefficients=answer.x.copy();z=B@answer.x;zd=D@answer.x;za=A@answer.x
report.update(vertical_constraint_residual=float(max(0.,(lo-C@answer.x).max(),(C@answer.x-hi).max())),height_offset_range=(z-position[:,2]).tolist()[::100])
print('vertical',report,flush=True)
nx=4*(N+1);n=nx+2*N;rows=[];low=[];high=[]
def add(indices,values,lo,hi):rows.append((indices,values));low.append(lo);high.append(hi)
initial_xy=np.r_[position[0,:2],velocity[0,:2]]
for j in range(4):add([j],[1.],initial_xy[j],initial_xy[j])
for k,dt in enumerate(np.diff(times)):
 omega2=(9.81+za[k])/z[k];omega=np.sqrt(omega2);ch=np.cosh(omega*dt);sh=np.sinh(omega*dt);AA=np.array([[ch,sh/omega],[omega*sh,ch]]);BB=np.array([1-ch,-omega*sh]);forcing=np.array([-hdot[k,1],hdot[k,0]])/(com.mass*z[k])
 for axis in range(2):
  for j in range(2):
   value=-BB[j]*forcing[axis]/omega2
   add([4*(k+1)+2*j+axis,4*k+axis,4*k+2+axis,nx+2*k+axis],[1.,-AA[j,0],-AA[j,1],-BB[j]],value,value)
 vertices=[]
 for side in np.flatnonzero(contacts[int(frames[k])]):
  pos,R,_=leg.evaluate(qs[k],int(side));yaw=np.arctan2(R[1,0],R[0,0]);rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
  vertices.extend(pos[:2]+np.array([[x,y]for x in [-.025,.085]for y in [-.02,.02]])@rotation.T)
 if len(vertices)<3:raise ValueError('Unsupported flight in reduced offline guide')
 for a,b,c in ConvexHull(np.asarray(vertices)).equations:add([nx+2*k,nx+2*k+1],[a,b],-np.inf,-c)
 for angle in np.arange(8)*np.pi/4:
  normal=np.array([np.cos(angle),np.sin(angle)])
  add([4*k,4*k+1,nx+2*k,nx+2*k+1],np.r_[omega2*normal,-omega2*normal],-np.inf,.50*(9.81+za[k])-normal@forcing)
for k in range(N+1):
 for angle in np.arange(8)*np.pi/4:
  normal=np.array([np.cos(angle),np.sin(angle)])
  add([4*k,4*k+1],normal,-np.inf,float(normal@position[k,:2]+.12*np.cos(np.pi/8)))
target=np.zeros(n);weights=np.ones(n)
for k in range(N+1):target[4*k:4*k+4]=np.r_[position[k,:2],velocity[k,:2]];weights[4*k:4*k+4]=[4000,4000,80,80]
weights[4*N:4*(N+1)]*=4
rr=[];cc=[];values=[]
for r,(indices,row) in enumerate(rows):rr.extend([r]*len(indices));cc.extend(indices);values.extend(row)
C=sparse.csc_matrix((values,(rr,cc)),shape=(len(rows),n));shift=C@target
row_scale=np.maximum(np.sqrt(C.multiply(C).sum(axis=1)).A.ravel(),1e-8)
solver=osqp.OSQP();solver.setup(P=sparse.diags(2*weights/4000,format='csc'),q=np.zeros(n),A=sparse.diags(1/row_scale)@C,l=(np.array(low)-shift)/row_scale,u=(np.array(high)-shift)/row_scale,eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polish=True,verbose=False);answer=solver.solve();report['horizontal_status']=answer.info.status
report['horizontal_solver_residuals']=[float(answer.info.pri_res),float(answer.info.dua_res)]
if answer.info.status_val in (1,2):
 x=answer.x+target;state=x[:nx].reshape(-1,4);cop=x[nx:].reshape(-1,2);residual=float(max(0.,(np.array(low)-C@x).max(),(C@x-np.array(high)).max()))
 report.update(reduced_constraint_residual=residual,peak_horizontal_offset=float(np.linalg.norm(state[:,:2]-position[:,:2],axis=1).max()),maximum_height_lowering=float((position[:,2]-z).max()),reduced_preflight_passed=residual<1e-5)
 np.savez_compressed(p/'wbc_remaining_com_plan.npz',frames=frames,times=times,state=state,cop=cop,height=z,height_velocity=zd,height_acceleration=za,height_knots=knots,height_coefficients=height_coefficients,reference_com=position,reference_velocity=velocity)
(p/'wbc_remaining_com_plan.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if not report.get('reduced_preflight_passed',False):
 raise RuntimeError('Reduced offline COM plan did not pass feasibility checks; no candidate is approved')
