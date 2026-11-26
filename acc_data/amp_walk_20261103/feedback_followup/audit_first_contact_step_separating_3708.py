from pathlib import Path
import ast,math,pickle,json,numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from preview_integration import advance
from centroidal_pose_linearization import skew
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_separating28'
c=pickle.load(open(p/(run+'.planner_worker.configuration.pkl'),'rb'));c['native_contact_timing']=False
ns=dict(ast=ast,math=math,Path=Path,SimpleNamespace=SimpleNamespace);fn=next(n for n in ast.parse((p/'preview_worker.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='initialize');exec(compile(ast.Module(body=[fn],type_ignores=[]),'<init>','exec'),ns);planner=ns['initialize'](c)
z=np.load(p/(run+'.replan_states/3.708.npz'),allow_pickle=True);first={k:z['problem_'+k] for k in ['A_eq','b_eq','G','b','lb','ub']};q=z['q'];v=z['v'];state=planner.model.evaluate(q,v,kinematic_bias=True);bounds=np.c_[first['lb'],first['ub']];rows=[]
for side,indices in z['problem_contact_patch_indices'].item().items():
 foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=planner.geometry[side];J=state['jacobian'][side]
 for index in indices:
  offset=R@points[index]-np.array([0.,0.,radii[index]]);Jz=(J[:3]-skew(offset)@J[3:])[2]
  row=dict(side=side,point=index,height=float(foot[2]+offset[2]),vertical_velocity=float(Jz@v),acceleration_extrema=[])
  for sign in [1.,-1.]:
   cost=np.r_[sign*Jz,np.zeros(41)]
   result=linprog(cost,A_ub=first['G'],b_ub=first['b'],A_eq=first['A_eq'],b_eq=first['b_eq'],bounds=bounds,method='highs')
   item=dict(status=result.message,success=result.success)
   if result.success:
    nq,nv=advance(q,v,result.x[:35],.002);next_state=planner.model.evaluate(nq,nv);nf=next_state['feet'][side];nr=Rotation.from_quat(nf[3:]).as_matrix()
    item.update(jacobian_acceleration=float(Jz@result.x[:35]),predicted_height_after_2ms=float(nf[2]+(nr@points[index])[2]-radii[index]))
   row['acceleration_extrema'].append(item)
  rows.append(row)
(p/'first_contact_step_separating_3708_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
