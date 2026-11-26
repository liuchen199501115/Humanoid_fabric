"""Restore baseline native foot clearances after independent-leg fitting.

Exact full-body IK preserves candidate COM, horizontal foot positions and foot
rotations. The original source and acceptance remain unchanged.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
name=sys.argv[1];parent=OUT/(name+'.npz');target=OUT/(name+'_clearance.npz');assert not target.exists()
with np.load(parent) as a:base={k:a[k] for k in a.files}
original=np.load(OUT/'prepared_combined_root_y50.npz')['control_q']
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf');legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
geometry=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.array(geometry['collision_points']);radii=np.array(geometry['collision_radii'])
def clearance(q,side):
 p,R,_=legs.evaluate(q,side);return np.min((points[side]@R.T)[:,2]+p[2]-radii[side])
offsets=np.zeros((len(source),2,3))
for k in range(641,1980):
 if np.array_equal(base['control_q'][k],original[k]):continue
 for side in (0,1):offsets[k,side,2]=clearance(original[k],side)-clearance(base['control_q'][k],side)
out,report=fit_com_reference(base,source,np.zeros((len(source),3)),legs,model,foot_offsets=offsets,max_iterations=250)
max_clearance=max(abs(clearance(out[k],s)-clearance(original[k],s)) for k in range(641,1980) for s in (0,1))
report.update(method='Native collision clearance restoration with COM-preserving full-body IK',maximum_native_clearance_residual_m=float(max_clearance),maximum_vertical_foot_correction_m=float(np.max(abs(offsets))),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
report['preflight_passed']=bool(report['preflight_passed'] and max_clearance<2e-5)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(report,flush=True)
