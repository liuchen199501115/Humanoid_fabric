import importlib.util,json,sys
from pathlib import Path
import numpy as np
root=Path('/home/sky/Humanoid_fabric')
out=root/'acc_data/g1_2_tracking/formula_verification'
spec=importlib.util.spec_from_file_location('prefix_audit',root/'acc_data/amp_walk_20261102/contact_clock_followup/summarize_trials.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=out
rows=[]
for path in sorted(out.glob('*.command.json')):
    name=path.name.removesuffix('.command.json')
    command=json.loads(path.read_text())
    if command.get('status')!='finished':continue
    try:r=m.summarize(name)
    except (FileNotFoundError,KeyError,ValueError) as exc:
        r=dict(run=name,audit_unavailable=str(exc),full_motion_tracking_passed=False)
    npz=out/(name+'.npz')
    if not npz.exists():npz=out/(name+'.failure.npz')
    if npz.exists():
        with np.load(npz) as d:
            opts=json.loads(d['controller_options'].item())
            r['execution']=dict(direct_wbc=opts['direct_wbc'],tracking_servo=opts['tracking_servo'],profile=opts['tracking_profile'],scope=opts['formula_scope'],source_reference=opts['formula_source_reference'],sync_frequency=opts['mit_sync_frequency'],servo_bandwidth=opts['servo_bandwidth'])
            if 'history_joint_position_desired' in d:
                p=d['history_joint_position_desired'];v=d['history_joint_velocity_desired'];sp=d['history_integration_seed_position'];sv=d['history_integration_seed_velocity']
                r['integrator']=dict(position_step_error=float(np.abs(p-sp-.002*v).max()),position_continuity_error=float(np.abs(sp[1:]-p[:-1]).max()),velocity_continuity_error=float(np.abs(sv[1:]-v[:-1]).max()))
            for k in ('servo_valid','servo_saturation_fraction','qp_validated'):
                if k in d:r[k]=d[k].tolist()
            if 'history_formula_weights' in d:
                r['weight_range']=[float(d['history_formula_weights'].min()),float(d['history_formula_weights'].max())]
            if 'history_torque_pd_estimate' in d:r['peak_pd_torque_norm']=float(np.linalg.norm(d['history_torque_pd_estimate'],axis=-1).max())
            if 'history_q' in d:
                q=d['history_q'].reshape(-1,36);ref=d['history_reference_q'].reshape(-1,36)
                r['peak_joint_rmse']=float(np.sqrt(np.mean((q[:,7:]-ref[:,7:])**2,axis=-1)).max())
                r['peak_root_position']=float(np.linalg.norm(q[:,:3]-ref[:,:3],axis=-1).max())
    r['returncode']=command.get('returncode')
    (out/(name+'.audit.json')).write_text(json.dumps(r,indent=2)+'\n');rows.append(r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('landings','next_step_diagnostics','recording')}))
(out/'results.json').write_text(json.dumps(dict(full_motion_tracking_passed=any(r['full_motion_tracking_passed'] for r in rows),trials=rows),indent=2)+'\n')
