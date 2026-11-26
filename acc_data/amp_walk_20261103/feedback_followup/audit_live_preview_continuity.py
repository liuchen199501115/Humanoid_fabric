from pathlib import Path
import time,json,numpy as np
from preview_manifold import difference
from preview_integration import advance
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_fine_process28';path=p/(run+'.full_body_preview.latest_successful_plan.npz')
def read():
 with np.load(path,allow_pickle=True) as raw:return {k:(raw[k].item() if raw[k].ndim==0 else raw[k].copy()) for k in raw.files}
old=read();np.savez_compressed(p/'live_continuity_previous.npz',**old)
for _ in range(100):
 try:new=read()
 except (OSError,ValueError):time.sleep(.5);continue
 if new['time']>old['time']+1e-8:break
 time.sleep(.5)
else:raise SystemExit('No new plan within the bounded observation window')
np.savez_compressed(p/'live_continuity_current.npz',**new)
frame=new['frame'];stage=min(np.searchsorted(old['frames'],frame+1e-8,side='right')-1,len(old['controls'])-1)
q,v=advance(old['physical_q'][stage],old['physical_v'][stage],old['controls'][stage,:35],max(0.,(frame-old['frames'][stage])/new['rate']))
e=difference(new['q'],q);dv=new['v']-v
records=json.loads((p/(run+'.full_body_preview.json')).read_text());previous=next(r for r in records if abs(r['time']-old['time'])<1e-8)
result=dict(previous_time=old['time'],current_time=new['time'],interval=new['time']-old['time'],stage=int(stage),joint_pose_rmse=float(np.sqrt(np.mean(e[6:]**2))),joint_pose_maximum=float(abs(e[6:]).max()),root_position_error=float(np.linalg.norm(e[:3])),root_orientation_error=float(np.linalg.norm(e[3:6])),joint_velocity_maximum=float(abs(dv[6:]).max()),base_velocity_error=float(np.linalg.norm(dv[:3])),angular_velocity_error=float(np.linalg.norm(dv[3:6])),previous_acceleration_projection_normalized_error=previous.get('latest_acceleration_projection_normalized_error'))
print(result);(p/'live_preview_continuity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
