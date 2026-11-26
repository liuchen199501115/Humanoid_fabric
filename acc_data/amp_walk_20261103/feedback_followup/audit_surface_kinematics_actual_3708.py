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
from preview_surface_kinematics import surface_jacobian,surface_bias
from full_body_preview_collision_fine_release import integrate
h=1e-5
plus=planner.model.evaluate(integrate(q,h*v),v)
minus=planner.model.evaluate(integrate(q,-h*v),v)
rows=[]
for side,indices in z['problem_contact_patch_indices'].item().items():
 points,radii=planner.geometry[side];points=points[indices];radii=radii[indices]
 R=Rotation.from_quat(state['feet'][side,3:]).as_matrix();J=state['jacobian'][side];b=state['foot_bias'][side]
 fd=(surface_jacobian(Rotation.from_quat(plus['feet'][side,3:]).as_matrix(),plus['jacobian'][side],points,radii)@v-surface_jacobian(Rotation.from_quat(minus['feet'][side,3:]).as_matrix(),minus['jacobian'][side],points,radii)@v)/(2*h)
 revised=surface_bias(R,J,b,v,points,radii)
 omega=J[3:]@v
 old=np.concatenate([b[:3]-skew(offset)@b[3:]+np.cross(omega,np.cross(omega,offset)) for offset in points@R.T-radii[:,None]*np.array([0.,0.,1.])])
 rows.append(dict(side=side,points=indices,old_max_error=float(abs(old-fd).max()),corrected_max_error=float(abs(revised-fd).max()),finite_difference=fd.tolist(),corrected=revised.tolist()))
(p/'surface_kinematics_actual_3708_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
