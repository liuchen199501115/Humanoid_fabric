import json,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent
reports=[]
for stem in ['pelvis_handover','convex_collision']:
 d=np.load(p/(stem+'.failure.npz'));t=d['history_time'];feet=d['history_feet'].reshape(-1,2,7);force=d['history_ground_foot_forces'].reshape(-1,2,3)[:,:,2]
 plan=json.loads((p/(stem+'.steps_plan.json')).read_text());rows=[]
 for ev,goal in zip(plan['events'][:6],plan['landing_positions_world']):
  foot=1-ev['support'];end=ev['end_frame']/200.;m=(t>=end-.05)&(t<end+.18)&(force[:,foot]>=20)
  if not m.any():continue
  ids=np.where(m)[0];err=np.linalg.norm(feet[ids,foot,:3]-goal,axis=1);i=ids[err.argmin()]
  rows.append(dict(event=ev,time=float(t[i]),goal=goal,closest_loaded_error=float(err.min()),offset=(feet[i,foot,:3]-goal).tolist(),force=float(force[i,foot])))
 reports.append(dict(run=stem,landings=rows))
print(json.dumps(reports,indent=2));(p/'landing_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
