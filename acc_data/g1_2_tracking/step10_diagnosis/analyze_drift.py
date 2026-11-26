"""Separate reference error, QP task compromise, and measured response error."""
from pathlib import Path
import numpy as np,json,sys
out=Path(__file__).resolve().parent
source=Path(sys.argv[1]) if len(sys.argv)>1 else out.parent/'response_formula/response_zero.npz'
with np.load(source,allow_pickle=False) as z:
 t=z['history_time'].ravel();n=len(t)
 get=lambda k:z['history_'+k].reshape(n,-1)
 q=get('q');ref=get('reference_q');com=get('com');target=get('com_target')
 ve=get('com_velocity')-get('com_target_velocity')
 request=get('com_acc_requested');pred=get('com_acc_qp_predicted');actual=get('com_acc_observed')
 mode=get('measured_mode')[:,0];command=get('commanded_mode')[:,0]
 rows=[]
 for lo,hi in [(3.3,3.48),(3.48,3.58),(3.58,3.65),(3.65,3.8),(3.8,4.0),(4.0,4.2)]:
  ix=(t>=lo)&(t<hi)&np.isfinite(actual).all(-1)
  if not ix.any():continue
  rms=lambda a:np.sqrt(np.mean(a[ix]**2,axis=0)).tolist()
  rows.append(dict(interval=[lo,hi],count=int(ix.sum()),root_position_rms=rms(q[:,:3]-ref[:,:3]),
    com_position_rms=rms(com-target),com_velocity_rms=rms(ve),
    qp_vs_requested_acceleration_rms=rms(pred-request),observed_vs_qp_acceleration_rms=rms(actual-pred),
    contact_mode_disagreement_fraction=float(np.mean(command[ix]!=mode[ix]))))
 samples=[]
 for sec in [3.48,3.58,3.65,3.75,3.96,4.216]:
  if sec>t[-1]:continue
  i=np.argmin(abs(t-sec));samples.append(dict(time_s=float(t[i]),root_error=(q[i,:3]-ref[i,:3]).tolist(),
   com_velocity_error=ve[i].tolist(),com_requested=request[i].tolist(),com_predicted=pred[i].tolist(),com_observed=actual[i].tolist()))
report=dict(recording=str(source),scope='Observed accelerations use the completed interval at the same stored row; interval RMS includes contact impulses.',intervals=rows,samples=samples)
dest=out/(source.stem+'.drift.json');dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
