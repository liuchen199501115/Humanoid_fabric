import json,numpy as np
from pathlib import Path
from analyze_contact_transfer import sole_clearances
p=Path(__file__).resolve().parent;reports=[]
for stem in ['pelvis_handover','convex_collision','initial_crouch']:
 path=p/(stem+'.failure.npz')
 if not path.exists():continue
 d=np.load(path);t=d['history_time'];frame=d['history_frame'].ravel();feet=d['history_feet'].reshape(-1,2,7);force=d['history_ground_foot_forces'].reshape(-1,2,3)[:,:,2];mode=d['history_measured_mode'].ravel()
 plan=json.loads((p/(stem+'.steps_plan.json')).read_text());report=json.loads((p/(stem+'.steps.json')).read_text());start=report['landings'][-1]['time'] if report['landings'] else 0
 n=report['measured_steps'];ev=plan['events'][n];s=ev['support'];sw=1-s;clear=sole_clearances(feet,plan['collision_points'],plan['collision_radii']);window=(t>start)&(frame>=ev['takeoff_frame']-1)&(frame<ev['landing_window_end_frame']);flight=window&(mode==s+1)&(force[:,s]>=100)&(force[:,sw]<10)
 error=np.linalg.norm(feet[:,sw,:3]-plan['landing_positions_world'][n],axis=1);near=window&(frame>=ev['end_frame']-1)&np.isin(mode,[3,sw+1])&(force[:,sw]>=20)&(error<=.04)
 longest=running=0
 for yes in near:
  running=running+1 if yes else 0;longest=max(longest,running)
 rows=dict(run=stem,first_unconfirmed_step=n+1,flight_evidence=bool((flight&(clear[:,sw]>=.02)).any()),peak_unloaded_clearance_m=float(clear[flight,sw].max()) if flight.any() else None,longest_loaded_goal_hold_s=longest*float(np.median(np.diff(t))),required_hold_s=.05,required_clearance_m=.02)
 reports.append(rows)
print(json.dumps(reports,indent=2));(p/'step_gate_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
