"""Offline sensitivity at an identical saved state; never a runtime controller.

Reweight only saved task quadratics, retaining every original QP constraint.
Report also the fixed-torque mass response to the measured waist-roll stop.
This local response is not a prediction of closed-loop tracking success.
"""
import json
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend

OUT=Path(__file__).resolve().parent
a=np.load(OUT/'ten_no_stop_verify.snapshot_3.800.npz')
def task(name,metric=None):
    p='composed_task_'+name+'_';A=a[p+'A'][0].astype(float);mask=a[p+'active'][0]
    W=a[p+'metric'][0].astype(float) if metric is None else metric
    W=W*mask[:,None]*mask[None,:]
    return A.T@W@A,A.T@W@(a[p+'bias'][0]-a[p+'desired'][0])
footH,footh=task('feet_pose');rollH,rollh=task('source_pelvis_orientation',np.diag([40.,0.,0.]))
fields=['H','h','G','b','lb','ub','A_eq','b_eq'];original={k:a['soft_problem_'+k].astype(float) for k in fields}
E=original['A_eq'][0];M=E[:35,:35];J=E[35:,:35];J=J[np.linalg.norm(J,axis=1)>1e-8]
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();coordinate=6+names.index('waist_roll_joint')
C=np.r_[J,np.eye(35)[coordinate:coordinate+1]];K=np.block([[M,C.T],[C,np.zeros((len(C),len(C)))]])
target=a['composed_task_source_pelvis_orientation_desired'][0];records=[]
for foot_scale in [16.,8.,4.,1.]:
    for pelvis_roll_scale in [0.,1.,2.,4.]:
        p={k:v.copy() for k,v in original.items()}
        p['H'][0,:35,:35]+=(foot_scale/16-1)*footH+pelvis_roll_scale*rollH
        p['h'][0,:35]+=(foot_scale/16-1)*footh+pelvis_roll_scale*rollh
        backend=DAQPBackend(single_solve=True,single_proximal=.0001,equilibration_cap=10.)
        solution=backend.solve(QPData(**{k:torch.tensor(v) for k,v in p.items()}))
        ok=bool(solution.status_ok.all());record=dict(foot_scale=foot_scale,pelvis_roll_scale=pelvis_roll_scale,certificate_passed=ok)
        if ok:
            qdd=solution.z.numpy()[0,:35]
            response=np.linalg.solve(K,np.r_[np.zeros(35+len(J)),-qdd[coordinate]])
            predicted=qdd+response[:35]
            # Positive upper-stop multiplier is necessary for an inward
            # reaction. A negative value would require the joint to release.
            record.update(qp_root_acceleration=qdd[3:6].tolist(),
                root_acceleration_with_stop=predicted[3:6].tolist(),stop_multiplier=float(response[-1]),
                stop_prediction_unilateral_valid=bool(response[-1]>=0),
                requested_pelvis_acceleration=target.tolist(),
                root_acceleration_error_norm=float(np.linalg.norm(predicted[3:6]-target)),
                leg_acceleration_norm=float(np.linalg.norm(predicted[6:18])))
        records.append(record)
(OUT/'frozen_weight_sensitivity.json').write_text(json.dumps(dict(time_s=3.8,
    scope='Local frozen-state QP and passive-stop response only; closed-loop verification required',cases=records),indent=2)+'\n')
for r in records:print(r,flush=True)
