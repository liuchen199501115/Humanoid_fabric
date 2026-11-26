"""Compare pre-change trajectories and ordered landing errors with the baseline."""
from pathlib import Path
import numpy as np,json
out=Path(__file__).resolve().parent;rows=[]
with np.load(out/'baseline_probe_v2.npz',allow_pickle=False) as base:
 bt=base['history_time'].ravel();bq=base['history_q'];bf=base['history_frame'].ravel()
 for manifest in sorted(out.glob('*.command.json')):
  name=manifest.name.removesuffix('.command.json');path=out/(name+'.npz')
  if name=='baseline_probe_v2' or not path.exists():continue
  with np.load(path,allow_pickle=False) as d:
   n=min(len(bt),len(d['history_time']));same=np.all(d['history_q'][:n]==bq[:n],axis=(1,2))&(d['history_frame'][:n].ravel()==bf[:n])
   changed=np.flatnonzero(~same);first=None if not len(changed) else int(changed[0])
   row=dict(run=name,first_state_or_phase_difference_s=None if first is None else float(bt[first]),
     exact_prefix_samples=n if first is None else first)
   if 'history_state_com_scale' in d:
    active=np.flatnonzero(d['history_state_com_scale'].ravel()>1.)
    row['first_state_com_activation_s']=None if not len(active) else float(d['history_time'].ravel()[active[0]])
   audit=out/(name+'.audit.json')
   if audit.exists():
    a=json.loads(audit.read_text());row['prefix_landings']=a['raw_source_audit']['prefix_landings']
    row['landing_errors']=[dict(step=e['step'],time_s=e['time_s'],root_position_m=e['root_position_error_m'],root_orientation_rad=e['root_orientation_error_rad'],joint_rmse_rad=e['joint_rmse_rad']) for e in a['landings'] if e['step']>=10]
   rows.append(row)
(out/'prefix_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
