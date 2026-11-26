from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parent
record=Path('acc_data/amp_walk_20261102/contact_clock_followup/vertical_feedback_verified.failure.npz')
with np.load(record) as d:
 t=d['history_time'].ravel();q=d['history_q'].reshape(-1,36);v=d['history_qd'].reshape(-1,35)
 refq=d['history_control_reference_q'].reshape(-1,36);refv=d['history_whole_body_reference_v'].reshape(-1,35);refa=d['history_whole_body_reference_a'].reshape(-1,35)
 names=list(d['joint_names']);j=names.index('left_ankle_pitch_joint')
 feedback=180*(refq[:,7:]-q[:,7:])+24*(refv[:,6:]-v[:,6:])
 desired=refa[:,6:]+np.clip(feedback,-30,30)
 residual=d['history_task_residual_joint_posture'].reshape(-1,29)
 rows=[]
 for at in [.9,1.,1.4,1.5,1.56,2.5]:
  i=abs(t-at).argmin();rows.append(dict(time_s=float(t[i]),joint=names[j],q=float(q[i,7+j]),reference_q=float(refq[i,7+j]),
   velocity=float(v[i,6+j]),reference_velocity=float(refv[i,6+j]),reference_acceleration=float(refa[i,6+j]),
   unclipped_feedback=float(feedback[i,j]),clipped_feedback=float(np.clip(feedback[i,j],-30,30)),
   requested_acceleration=float(desired[i,j]),predicted_acceleration=float(desired[i,j]+residual[i,j])))
 report=dict(recording=str(record),feedback_gain=dict(stiffness=180,damping=24,cap=30),samples=rows)
(p/'feedback_saturation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(r)
