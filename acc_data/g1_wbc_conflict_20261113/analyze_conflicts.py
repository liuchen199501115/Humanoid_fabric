"""Fixed-state WBC posture-allocation experiments; no physical constraints removed."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples'))
from analyze_g1_shoulder_roll import get_problem,metric_delta,solve_qp,solve_two_stage

OUT=Path(__file__).resolve().parent


def rms(values, active):
    return float(np.sqrt(np.mean(values[active]**2))) if active.any() else 0.


def metrics(d,x,names):
    acceleration=x[:35]-d['correction'][0]
    result={}
    for task in ('center_of_mass','feet_pose','body_orientation','source_pelvis_orientation','joint_posture'):
        A,bias,desired,active=(d['task_'+task+'_'+k][0] for k in ('A','bias','desired','active'))
        residual=A@acceleration+bias-desired
        result[task]=dict(residual=residual.tolist(),active=active.tolist(),rms=rms(residual,active))
        if task=='feet_pose':
            for side,start in (('left',0),('right',6)):
                for kind,inds in (('position',np.arange(start,start+3)),('orientation',np.arange(start+3,start+6))):
                    result[side+'_foot_'+kind]=dict(rms=rms(residual[inds],active[inds]),active_rows=int(active[inds].sum()))
        if task=='joint_posture':
            for group,indices in groups(names).items():
                result[group+'_posture']=dict(rms=rms(residual[indices],active[indices]))
        if task=='center_of_mass':
            result['com_acceleration']=(A@acceleration+bias).tolist()
    lo,hi=d['priority_level_0_lb'][0,47:],d['priority_level_0_ub'][0,47:]
    result['torques']=[dict(joint=name,torque_nm=float(x[47+i]),lower_nm=float(lo[i]),upper_nm=float(hi[i]),
        margin_nm=float(min(x[47+i]-lo[i],hi[i]-x[47+i])),
        utilization=float(abs(x[47+i])/max(abs(lo[i]),abs(hi[i])))) for i,name in enumerate(names)]
    return result


def groups(names):
    waist=[i for i,n in enumerate(names) if n.startswith('waist_')]
    arms=[i for i,n in enumerate(names) if any(s in n for s in ('shoulder','elbow','wrist'))]
    legs=[i for i,n in enumerate(names) if any(s in n for s in ('hip','knee','ankle'))]
    return dict(waist=waist,arms=arms,legs=legs,waist_arms=waist+arms)


def screen(base,case,group):
    """Exploratory acceptance screen, not a stability or joint-angle guarantee."""
    protected=[('body_orientation',.25),('source_pelvis_orientation',.25),('legs_posture',.5),
               ('left_foot_position',.05),('right_foot_position',.05),
               ('left_foot_orientation',.25),('right_foot_orientation',.25)]
    if group=='waist':protected.append(('arms_posture',.5))
    if group=='arms':protected.append(('waist_posture',.5))
    violations=[key for key,floor in protected if case[key]['rms']>1.05*base[key]['rms']+floor]
    selected=group+'_posture'
    if case[selected]['rms']>1.2*base[selected]['rms']+.5:violations.append(selected)
    benefit=case['center_of_mass']['rms']<=.95*base['center_of_mass']['rms']
    return dict(com_improves_at_least_5_percent=bool(benefit),protected_task_violations=violations,
                passes=bool(benefit and not violations))


def feasible_com(d,first,names):
    _,h,E,e,G,b,lo,hi=first
    A=d['task_center_of_mass_A'][0].astype(float)
    bias=d['task_center_of_mass_bias'][0]-A@d['correction'][0]
    J=np.pad(A,((0,0),(0,len(h)-35)))
    desired=d['task_center_of_mass_desired'][0]
    result={}
    def solve(c,E=E,e=e):
        lp=linprog(c,A_ub=G,b_ub=b,A_eq=E,b_eq=e,bounds=np.c_[lo,hi],method='highs')
        row=dict(status=int(lp.status),message=lp.message)
        if lp.success:
            row['constraint_residual']=float(max(0,np.max(G@lp.x-b),np.max(abs(E@lp.x-e)),
                                                  np.max(lo-lp.x),np.max(lp.x-hi)))
            stationarity=c-G.T@lp.ineqlin.marginals-E.T@lp.eqlin.marginals-lp.lower.marginals-lp.upper.marginals
            finite_lo,finite_hi=np.isfinite(lo),np.isfinite(hi)
            dual=(b@lp.ineqlin.marginals+e@lp.eqlin.marginals+
                  lo[finite_lo]@lp.lower.marginals[finite_lo]+hi[finite_hi]@lp.upper.marginals[finite_hi])
            row['stationarity_residual']=float(abs(stationarity).max())
            row['duality_gap']=float(abs(c@lp.x-dual))
            row['dual_sign_violation']=float(max(0,lp.ineqlin.marginals.max(),
                -lp.lower.marginals.min(),lp.upper.marginals.max()))
            row['validated']=max(row['constraint_residual'],row['stationarity_residual'],
                                 row['duality_gap'],row['dual_sign_violation'])<1e-5
            row['torque_budget_sensitivity']=[dict(joint=name,
                objective_improvement_per_nm=float(lp.lower.marginals[47+i]-lp.upper.marginals[47+i]))
                for i,name in enumerate(names) if lp.lower.marginals[47+i]-lp.upper.marginals[47+i]>1e-6]
            row['limiting_inequality_rows']=[dict(row=int(i),multiplier=float(-v),
                family='wrench' if np.any(abs(G[i,35:47])>1e-10) else 'acceleration',
                coefficients=[[int(j),float(G[i,j])] for j in np.flatnonzero(abs(G[i])>1e-9)],rhs=float(b[i]))
                for i,v in enumerate(lp.ineqlin.marginals) if abs(v)>1e-5]
        return lp,row
    for axis in range(3):
        bounds=[]
        for sign in (1.,-1.):
            lp,row=solve(sign*J[axis])
            if row.get('validated'):row['acceleration_mps2']=float(J[axis]@lp.x+bias[axis])
            bounds.append(row)
        result['xyz'[axis]]=dict(requested_mps2=float(desired[axis]),minimum=bounds[0],maximum=bounds[1])
    _,result['exact_com_request']=solve(np.zeros_like(h),np.vstack((E,J)),np.r_[e,desired-bias])
    if result['exact_com_request']['status']==4:
        EE,ee=np.vstack((E,J)),np.r_[e,desired-bias]
        gs=np.maximum(np.linalg.norm(G,axis=1),1e-8);es=np.maximum(np.linalg.norm(EE,axis=1),1e-8)
        scaled=linprog(np.zeros_like(h),A_ub=G/gs[:,None],b_ub=b/gs,A_eq=EE/es[:,None],
                       b_eq=ee/es,bounds=np.c_[lo,hi],method='highs')
        _,independent=solve_qp(np.eye(len(h)),np.zeros_like(h),EE,ee,G,b,lo,hi)
        result['exact_com_request']['scaled_highs_status']=int(scaled.status)
        result['exact_com_request']['independent_clarabel_status']=independent['status']
        # A failed candidate's residual is not an executed physical quantity.
    if result['exact_com_request'].get('validated'):
        x,status=solve_qp(first[0],h,np.vstack((E,J)),np.r_[e,desired-bias],G,b,lo,hi)
        result['exact_com_tradeoff']=dict(solver=status)
        if status['accepted']:result['exact_com_tradeoff']['metrics']=metrics(d,x,names)
    return result


def main():
    with np.load(OUT/'baseline_probe.npz') as history:
        names=history['final_joint_order'].tolist()
        times=history['history_time'].ravel()
        history_q=history['history_q'].reshape(-1,36)
        history_v=history['history_qd'].reshape(-1,35)
        velocity_error=(history['history_com_velocity']-history['history_com_target_velocity']).reshape(-1,3)
        with np.load(ROOT/'acc_data/g1_shoulder_roll_20261112/guarded/baseline_live.npz') as old:
            for key in ('history_q','history_qd','history_frame'):np.testing.assert_array_equal(history[key],old[key])
        group_indices=groups(names)
        cases=[('baseline',[],1.)]+[(g,group_indices[g],scale) for g in ('waist','arms','waist_arms') for scale in (.9,.75,.5)]
        rows=[];solutions={}
        for path in sorted(OUT.glob('baseline_probe.qp_*.npz')):
            with np.load(path) as f:d={k:f[k] for k in f.files if not k.startswith(('history_','servo_'))}
            sec=float(d['time']);i=int(np.argmin(abs(times-sec)))
            np.testing.assert_allclose(times[i],sec,atol=1e-9)
            np.testing.assert_array_equal(history_q[i],d['q'][0])
            np.testing.assert_array_equal(history_v[i],d['qd'][0])
            np.testing.assert_allclose(d['problem_solution'][0,47:],d['total_torque'][0],atol=1e-5)
            first,final=get_problem(d,'priority_level_0_'),get_problem(d,'problem_')
            assert 'priority_level_1_H' not in d
            A=d['task_joint_posture_A'][0].astype(float)
            np.testing.assert_array_equal(A,np.eye(35)[6:])
            J=np.pad(A,((0,0),(0,len(first[1])-35)))
            offset=d['task_joint_posture_bias'][0]-d['task_joint_posture_desired'][0]-A@d['correction'][0]
            with np.load(path.with_name(path.name.replace('.qp_','.response_'))) as response:
                observed=(response['qd'][0].astype(float)-d['qd'][0])/float(response['dt'])
                applied_error=float(abs(response['applied_torque']-d['total_torque']).max())
                np.testing.assert_allclose(response['qd'][0],history_v[i+1],atol=1e-6)
            row=dict(time_s=sec,source_frame=float(d['motion_frame_position']),
                contact_mode=int(d['contact_mode'][0]),snapshot=str(path),
                com_velocity_error_mps=velocity_error[i].tolist(),applied_torque_max_error_nm=applied_error,
                predicted_observed_joint_acceleration_rms=float(np.sqrt(np.mean((observed[6:]-d['predicted_acc'][0,6:])**2))),
                cases=[])
            xs=[]
            for group,indices,scale in cases:
                delta=metric_delta(d['task_joint_posture_metric'][0],d['task_joint_posture_active'][0],indices,scale)
                x,status=solve_two_stage(first,final,J.T@delta@J,J.T@delta@offset)
                case=dict(group=group,scale=scale,solver=status)
                if status['accepted']:
                    case.update(metrics(d,x,names));xs.append(x)
                else:xs.append(np.full(len(first[1]),np.nan))
                row['cases'].append(case)
            base=row['cases'][0]
            if not base['solver']['accepted']:raise RuntimeError(f'Baseline replay failed: {path}')
            difference=float(abs(xs[0][:35]-d['problem_solution'][0,:35]).max())
            row['baseline_acceleration_replay_max_difference']=difference
            if difference>1e-3:raise RuntimeError(f'Baseline acceleration mismatch {sec}: {difference}')
            for case in row['cases'][1:]:
                if case['solver']['accepted']:
                    case['screen']=screen(base,case,case['group'])
                    case['com_residual_ratio']=case['center_of_mass']['rms']/max(base['center_of_mass']['rms'],1e-12)
            row['com_feasibility_without_priority_locks']=feasible_com(d,first,names)
            rows.append(row);solutions[path.stem]=np.asarray(xs)
            print(sec,'baseline replay',difference,'passing candidates',
                  [(c['group'],c['scale']) for c in row['cases'][1:] if c.get('screen',{}).get('passes')],flush=True)
    aggregates=[]
    for j,(group,_,scale) in enumerate(cases[1:],1):
        valid=[r['cases'][j] for r in rows if r['cases'][j]['solver']['accepted']]
        safe=all(not c['screen']['protected_task_violations'] for c in valid)
        benefits=sum(c['screen']['passes'] for c in valid)
        aggregates.append(dict(group=group,scale=scale,valid_states=len(valid),
            beneficial_states=benefits,protected_at_all_states=safe,
            eligible_for_closed_loop=len(valid)==len(rows) and safe and benefits>=3,
            com_residual_ratios=[c['com_residual_ratio'] for c in valid]))
    report=dict(scope='Fixed-state two-stage WBC replay; unchanged physical/auxiliary constraints, rebuilt priority locks; no MPC.',
        baseline_history_exact_match=True,controller_sha256=hashlib.sha256((ROOT/'examples/test_g1.py').read_bytes()).hexdigest(),
        screen='COM residual improves >=5%; other task residual RMS <= baseline*1.05+floor (COM/foot position units m/s², rotations rad/s²); selected posture <= baseline*1.2+0.5. Floors: foot position 0.05, rotation 0.25, other joints 0.5. Eligible only if all states protected and >=3 beneficial. This is an experimental screen, not a stability guarantee.',
        candidates=aggregates,snapshots=rows)
    (OUT/'conflicts.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'counterfactual_solutions.npz',**solutions)
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
