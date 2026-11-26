from pathlib import Path
import numpy as np,json
from scipy.optimize import linprog
p=Path(__file__).parent;d=np.load(p/'fixed_clock120_plant7_balance_priority.snapshot_2.600.npz');n=d['problem_h'].shape[-1];A=d['problem_A_eq'][0].astype(float);b=d['problem_b_eq'][0].astype(float);G=d['problem_G'][0].astype(float);h=d['problem_b'][0].astype(float);bounds=list(zip(d['problem_lb'][0],d['problem_ub'][0]));J=np.pad(d['task_center_of_mass_A'][0],((0,0),(0,n-35)));bias=d['task_center_of_mass_bias'][0];target=d['task_center_of_mass_desired'][0];out={}
for scope in ['original','xy_exact','xyz_exact']:
 count={'original':0,'xy_exact':2,'xyz_exact':3}[scope];Ae=np.vstack((A,J[:count]));be=np.r_[b,target[:count]-bias[:count]];out[scope]={}
 for axis in range(3):
  limits=[]
  for sign in [1,-1]:
   r=linprog(sign*J[axis],A_ub=G,b_ub=h,A_eq=Ae,b_eq=be,bounds=bounds,method='highs');limits.append(float(J[axis]@r.x+bias[axis]) if r.success else r.message)
  out[scope][str(axis)]=limits
out['requested_acceleration']=target.tolist();out['predicted_acceleration']=(J[:,:35]@d['predicted_acc'][0]+bias).tolist();(p/'late_com_feasibility.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
