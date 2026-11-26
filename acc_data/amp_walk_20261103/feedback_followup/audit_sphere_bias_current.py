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
for side in [0,1]:
 omega=state['jacobian'][side,3:]@v
 print(dict(side=side,omega=omega.tolist(),sphere_vertical_bias_error=float(.005*(omega[0]**2+omega[1]**2))))
