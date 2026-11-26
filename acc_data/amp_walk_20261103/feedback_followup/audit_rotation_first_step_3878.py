from pathlib import Path
# Reuse the saved-state/model setup without its LP loop or artifact writes.
setup=(Path(__file__).parent/'audit_first_contact_step_3878.py').read_text().split('for side,indices')[0]
exec(compile(setup,'<saved_actual_setup>','exec'))
from preview_manifold import difference,integrate
from preview_integration import position_factor
from preview_rotation_transition import linearize_rotation
frame=float(z['frame']);rate=float(z['rate']);dt=.002
rq,rv,ra=[a.numpy().ravel() for a in planner.reference.kinematics(frame,rate)];nextq=planner.reference.kinematics(frame+dt*rate,rate)[0].numpy().ravel();e=difference(q,rq);dv=v-rv
result=linprog(np.zeros(76),A_ub=first['G'],b_ub=first['b'],A_eq=first['A_eq'],b_eq=first['b_eq'],bounds=bounds,method='highs');assert result.success
acc=result.x[:35];actual,_=advance(q,v,acc,dt)
A=np.eye(35);A[3:6,3:6]+=dt*skew(rv[3:6]);old_error=A@e+dt*dv+position_factor(dt)*acc+difference(integrate(rq,dt*rv),nextq);old=integrate(nextq,old_error)
new_error=old_error.copy();offset,A,V,B=linearize_rotation(e[3:6],dv[3:6],ra[3:6],rq[3:7],rv[3:6],nextq[3:7],dt);new_error[3:6]=offset+A@e[3:6]+V@dv[3:6]+B@acc[3:6];new=integrate(nextq,new_error)
rows=[]
for label,pose in [('semiimplicit_physical',actual),('old_rotation_surrogate',old),('exact_rotation_linearization',new)]:
 state=planner.model.evaluate(pose,v);foot=state['feet'][1];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=planner.geometry[1]
 rows.append(dict(method=label,root_orientation_error=float(np.linalg.norm(difference(pose,actual)[3:6])),support_heel_height=float(foot[2]+(R@points[9])[2]-radii[9])))
(p/'rotation_first_step_3878_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
