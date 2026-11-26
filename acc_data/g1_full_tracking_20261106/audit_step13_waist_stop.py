"""Locate measured waist-stop impulses and replay auxiliary-row feasibility."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
p=Path(__file__).parent
cases=[]
for name in ['waist_braking_before12','thirteenth_early_waist_braking']:
    with np.load(p/(name+'.failure.npz')) as d:
        prefix='priority_problem_' if 'priority_problem_G' in d else 'problem_'
        get=lambda k:d[prefix+k][0].astype(float)
        E,e,G,b,lo,hi=[get(k) for k in ['A_eq','b_eq','G','b','lb','ub']]
        # Only the final auxiliary orientation row may be removed in this
        # counterfactual. No torque, friction, joint or contact row is removed.
        row=G[-1].copy();assert np.count_nonzero(row[3:6])>0
        row[3:6]=0;assert not np.any(row)
        result=dict(run=name,failed_time_s=float(d['control_time']),cases=[])
        for removed in [False,True]:
            GG,bb=(G[:-1],b[:-1]) if removed else (G,b)
            solution=linprog(np.zeros(len(lo)),A_ub=GG,b_ub=bb,A_eq=E,b_eq=e,
                             bounds=list(zip(lo,hi)),method='highs')
            record=dict(auxiliary_orientation_row_removed=removed,status=int(solution.status),message=solution.message)
            if solution.success:
                x=solution.x
                record['retained_constraint_residual']=float(max(np.max(abs(E@x-e)),np.max(GG@x-bb),np.max(lo-x),np.max(x-hi),0.))
            result['cases'].append(record)
        cases.append(result)
with np.load(p/'thirteenth_support_knee20.npz') as d:
    t=d['history_time'].ravel();q=d['history_q'].reshape(-1,36);v=d['history_qd'].reshape(-1,35);a=d['history_qdd_predicted'].reshape(-1,35)
    upper=float(d['joint_position_limits'].reshape(29,2)[13,1])
    near=np.flatnonzero((t>4.2)&(q[:,20]>=upper-1e-4));i=int(near[0])-1
    impulse=dict(run='thirteenth_support_knee20',interval_s=[float(t[i]),float(t[i+1])],
        waist_roll_limit_rad=upper,waist_roll_position_rad=q[i:i+2,20].tolist(),
        waist_roll_speed_rad_s=v[i:i+2,19].tolist(),
        root_angular_acceleration_prediction=a[i,3:6].tolist(),
        root_angular_acceleration_observed=((v[i+1,3:6]-v[i,3:6])/(t[i+1]-t[i])).tolist(),
        self_contact_events=d['self_contact_events'].tolist())
report=dict(scope='Read-only saved-state diagnostics. No counterfactual command is applied; raw-source fidelity remains independent.',
            infeasibility_replays=cases,waist_stop_observation=impulse)
(p/'step13_waist_stop.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
