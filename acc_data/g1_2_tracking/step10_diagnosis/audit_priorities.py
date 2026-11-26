"""Counterfactual objectives under identical saved physical constraints."""
from pathlib import Path
import sys,json,numpy as np
from scipy import sparse
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'acc_data/amp_walk_20261103/feedback_followup'))
from solve_upper_fit_qp import solve
reports=[]
for path in sorted(out.glob('baseline_probe_v2.qp_*.npz')):
 with np.load(path,allow_pickle=False) as d:
  get=lambda k:d['priority_level_0_'+k][0].astype(float)
  H,h,E,e,G,b,lo,hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h)
  options=json.loads(str(d['controller_options']));assert options['direct_wbc']
  offset=-d['correction'][0].astype(float)
  C=sparse.csc_matrix(np.vstack((E,G,np.eye(n))));low=np.r_[e,np.full(len(G),-np.inf),lo];high=np.r_[e,b,hi]
  def task(name,selected=None):
   A=d['task_'+name+'_A'][0].astype(float);M=d['task_'+name+'_metric'][0].astype(float)
   active=d['task_'+name+'_active'][0].copy()
   if selected is not None:
    mask=np.zeros_like(active);mask[selected]=True;active &=mask
   M=M*active[:,None]*active[None,:]
   J=np.c_[A,np.zeros((len(A),n-35))]
   r=A@offset+d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0]
   return J.T@M@J,J.T@M@r,J,r
  CH,ch,CJ,cr=task('center_of_mass')
  mode=int(d['contact_mode'].item());stance=(list(range(6)) if mode&1 else [])+(list(range(6,12)) if mode&2 else [])
  PH,ph,_,_=task('joint_posture',stance)
  cases=[('baseline',H,h),('com_x4',H+3*CH,h+3*ch),('com_x16',H+15*CH,h+15*ch),
         ('stance_posture_x025',H-.75*PH,h-.75*ph),('stance_posture_x0',H-PH,h-ph),
         ('com_only',CH+np.eye(n)*1e-6,ch)]
  for name,Q,c in cases:
   Q=(Q+Q.T)/2
   x,status=solve(sparse.csc_matrix(Q),c,C,low,high)
   row=dict(snapshot=path.name,time_s=float(d['time']),case=name,solver=status)
   if status['accepted']:
    acc=x[:35]+offset
    row['com_acceleration']=(d['task_center_of_mass_A'][0]@acc+d['task_center_of_mass_bias'][0]).tolist()
    row['com_residual']=(CJ@x+cr).tolist()
    row['baseline_predicted_difference']=float(np.max(abs(acc-d['predicted_acc'][0])))
    row['waist_acceleration']=acc[18:21].tolist()
    row['joint_posture_residual_rms']=float(np.sqrt(np.mean((d['task_joint_posture_A'][0]@acc+d['task_joint_posture_bias'][0]-d['task_joint_posture_desired'][0])**2)))
   reports.append(row)
   print(row,flush=True)
  exact=sparse.vstack((C,sparse.csc_matrix(CJ)),format='csc')
  _,status=solve(sparse.eye(n,format='csc')*1e-6,np.zeros(n),exact,np.r_[low,-cr],np.r_[high,-cr])
  reports.append(dict(snapshot=path.name,time_s=float(d['time']),case='exact_requested_com',solver=status))
(out/'priority_audit.json').write_text(json.dumps(dict(scope='Offline counterfactuals; physical constraints unchanged; no tracking claim.',cases=reports),indent=2)+'\n')
