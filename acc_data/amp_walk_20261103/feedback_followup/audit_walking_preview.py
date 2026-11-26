from pathlib import Path
import json,numpy as np
from walking_centroidal_preview import CentroidalPreview
p=Path(__file__).parent
preview=object.__new__(CentroidalPreview);preview.N=8;preview.dt=.025;preview.mass=33.4;preview.fps=200.
state=np.array([0.,0.,.7,0.,0.,0.,0.,0.,0.]);feet=np.array([[0.,0.,0.],[.2,0.,0.]])
# A measured left point contact has zero moment. Right foot remains airborne.
rows=[]
for axis in range(3,6):
 for sign in [-1.,1.]:
  row=np.zeros(12);row[axis]=sign;rows.append(row)
for axis in range(2):
 for sign in [-1.,1.]:
  row=np.zeros(12);row[axis]=sign;row[2]=-.5;rows.append(row)
row=np.zeros(12);row[2]=-1;rows.append(row)
G=np.array(rows);b=np.zeros(len(G))
preview.reference=lambda frame,*clock_derivatives:(state.copy(),feet.copy(),1)
a,h,record=preview.solve(state,feet,1,0.,G,b)
assert np.linalg.norm(a)<.001 and np.linalg.norm(h)<.001
assert record['persistent_stance_nodes']==[[True,False]]*8
preview.reference=lambda frame,*clock_derivatives:(state.copy(),feet.copy(),1 if frame<10 else 3 if frame<20 else 2)
a,h,transition=preview.solve(state,feet,1,0.,G,b)
assert transition['persistent_stance_nodes']==[[True,False]]*4+[[False,False]]*4
sampled=[]
preview.reference_clock=lambda offset:(100.+240.*offset,240.,0.)
def sample_clock(frame,rate,acceleration):
 sampled.append((frame,rate,acceleration));return state.copy(),feet.copy(),1
preview.reference=sample_clock
preview.solve(state,feet,1,100.,G,b)
assert len(sampled)==9 and all(abs(f-(100.+6*k))<1e-9 and r==240. and a==0. for k,(f,r,a) in enumerate(sampled))
report=dict(static_acceleration=record['acceleration'],static_constraint_residual=record['residual'],static_persistent_nodes=record['persistent_stance_nodes'],transition_persistent_nodes=transition['persistent_stance_nodes'],checks_passed=True,retimed_reference_clock_verified=True,physical_tracking_certified=False)
(p/'walking_preview_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
