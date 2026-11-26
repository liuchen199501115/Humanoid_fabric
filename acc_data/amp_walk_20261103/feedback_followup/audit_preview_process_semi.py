from pathlib import Path
import json,numpy as np,pickle,time
from preview_process import ProcessPreview
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_landing24';saved=np.load(p/'wrench_fixed_ten.npz');history=np.load(p/'fixed_clock120_body10_overdue.npz');plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
configuration=dict(urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names=saved['final_joint_order'].tolist(),reference_path=str(p/(run+'.control_reference.npz')),modes=modes,geometry=geometry,limits=saved['joint_position_limits'][0],effort=saved['effort_limits'].ravel(),velocity=history['history_joint_velocity_limits'][0].ravel(),horizon=24,guarded=True,warm_start=False)
d=np.load(p/(run+'.full_body_preview.failure_plan.npz'));first={k:d['problem_'+k] for k in ['A_eq','b_eq','G','b','lb','ub']};first.update(mode=int(d['problem_mode']),contact_patch_indices={1:[9,11]})
worker=ProcessPreview(configuration,p/'preview_process_semi_test_worker.log');start=time.monotonic()
try:result=worker.refine(d['q'],d['v'],float(d['frame']),float(d['rate']),first)
finally:worker.close()
assert result['success'],result['status'];expected=np.load(p/'full_body_preview_semiimplicit_30.npz');differences={k:float(np.max(abs(result[k]-expected[k]))) for k in ['states','controls','impulses']};assert max(differences.values())<1e-9,differences
report=dict(success=True,maximum_differences=differences,elapsed_seconds=time.monotonic()-start);print(report);(p/'preview_process_semi_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
