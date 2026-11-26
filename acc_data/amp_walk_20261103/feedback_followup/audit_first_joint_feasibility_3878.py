from pathlib import Path
setup=(Path(__file__).parent/'audit_first_contact_step_3878.py').read_text().split('for side,indices')[0]
exec(compile(setup,'<saved_actual_setup>','exec'))
dt=.002;select=np.zeros((29,76));select[:,6:35]=np.eye(29);records=[]
for reserve in [0.,.0002,.005]:
 low=(c['limits'][:,0]+reserve-q[7:]-dt*v[6:])/dt**2
 high=(c['limits'][:,1]-reserve-q[7:]-dt*v[6:])/dt**2
 G=np.vstack([first['G'],select,-select]);b=np.r_[first['b'],high,-low]
 r=linprog(np.zeros(76),A_ub=G,b_ub=b,A_eq=first['A_eq'],b_eq=first['b_eq'],bounds=bounds,method='highs')
 item=dict(future_reserve=reserve,success=r.success,status=r.message)
 if r.success:item['right_ankle_roll_acceleration']=float(r.x[6+c['names'].index('right_ankle_roll_joint')])
 records.append(item)
(p/'first_joint_feasibility_3878_audit.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
