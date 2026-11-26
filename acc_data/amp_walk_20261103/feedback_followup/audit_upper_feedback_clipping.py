from pathlib import Path
import json,numpy as np
p=Path(__file__).parent;d=np.load(p/'fixed_clock120_body10_turn_recovery40.npz');q=d['history_q'][:,0];v=d['history_qd'][:,0];r=d['history_whole_body_reference_q'][:,0];rv=d['history_whole_body_reference_v'][:,0];ra=d['history_whole_body_reference_a'][:,0];frames=d['history_frame'];rows=[]
for frame in [840,880,900,920,930,938]:
 i=np.argmin(abs(frames-frame));feedback=-180*(q[i,7:]-r[i,7:])-24*(v[i,6:]-rv[i,6:]);index=[13,14,23]
 rows.append(dict(frame=float(frames[i]),joints=d['final_joint_order'][index].tolist(),position_error=(q[i,7:]-r[i,7:])[index].tolist(),velocity_error=(v[i,6:]-rv[i,6:])[index].tolist(),feedforward=ra[i,6:][index].tolist(),raw_feedback=feedback[index].tolist(),desired_cap150=(ra[i,6:]+np.clip(feedback,-150,150))[index].tolist(),desired_cap500=(ra[i,6:]+np.clip(feedback,-500,500))[index].tolist()))
(p/'upper_feedback_clipping_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-3:],indent=2))
