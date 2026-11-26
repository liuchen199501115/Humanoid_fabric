"""Compose the validated free-leg fit with the independent waist-roll margin."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;root=p.parents[1]
parent=p/'prepared_freeleg_dynamics_r10_l10_u05_scale80.npz';margin=p/'prepared_stop_margin_waist10.npz';basepath=p/'prepared_combined_root_y50.npz'
with np.load(parent) as d:base={k:d[k] for k in d.files}
factor=float(sys.argv[1]) if len(sys.argv)>1 else 1.;assert 0<factor<=3
q=base['control_q'].copy();q0=np.load(basepath)['control_q'];qm=np.load(margin)['control_q']
change=qm-q0;assert np.count_nonzero(np.max(abs(change),axis=0))==1
q[:,20]+=factor*change[:,20]
source=np.load(root/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=1)).max())
assert rmse<.19;assert np.all(abs(q[:,20])<=.52+1e-7)
np.testing.assert_array_equal(q[:,:20],base['control_q'][:,:20]);np.testing.assert_array_equal(q[:,21:],base['control_q'][:,21:])
target=p/f'prepared_freeleg80_waist{round(factor*10)}.npz';assert not target.exists()
with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=q))
report=dict(method='Compose verified independent waist roll offset with free-leg scale80; feet unchanged from free-leg candidate',waist_margin_multiplier=factor,peak_source_joint_rmse_rad=rmse,preflight_passed=True,full_tracking_passed=False,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),margin_sha256=hashlib.sha256(margin.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item())
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
