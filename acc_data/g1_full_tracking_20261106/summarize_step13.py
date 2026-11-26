"""Report this step-13 investigation without promoting unsuccessful profiles."""
from pathlib import Path
import hashlib,json
p=Path(__file__).parent
names=['waist_braking_root_margin13','thirteenth_clock160','thirteenth_waist_metric4',
       'thirteenth_waist16_root_margin','thirteenth_flat_roll_support','thirteenth_clock080',
       'thirteenth_full_effort','thirteenth_force_fit150_physics','thirteenth_com_feet_first',
       'thirteenth_fast_full_effort','thirteenth_upper_posture_first','thirteenth_support_knee20','thirteenth_knee_orientation30','thirteenth_early_waist_braking','thirteenth_waist_braking_physical','thirteenth_soft_upper_braking200','thirteenth_late_upper_braking200','thirteenth_yaw_priority_turn','thirteenth_restore_torso_metric','thirteenth_yaw_torso_feedback','thirteenth_late_yaw_waist','thirteenth_smooth12_turn','thirteenth_feasible_reference','thirteenth_feasible_root_braking','thirteenth_feasible_root_braking_v2','thirteenth_feasible_all_tasks','thirteenth_grounded_landing','thirteenth_landing_bridge','thirteenth_phase_lead12','thirteenth_contact_velocity8']
rows=[]
for name in names:
 command_path=p/(name+'.command.json')
 if not command_path.exists():continue
 command=json.loads(command_path.read_text())
 row=dict(run=name,status=command['status'],returncode=command.get('returncode'),step13_passed=False)
 for suffix in ['tracking.json','audit.json','raw_source.json']:
  path=p/(name+'.'+suffix)
  if not path.exists():continue
  d=json.loads(path.read_text())
  for key in ['measured_landings','last_source_frame','stop_reason','fidelity_stop',
              'prefix_fidelity_passing_landings','raw_source_prefix_landings','full_motion_tracking_passed']:
   if key in d:row[key]=d[key]
 row['step13_passed']=bool(row.get('raw_source_prefix_landings',0)>=13 and row.get('prefix_fidelity_passing_landings',0)>=13)
 rows.append(row)
rejected=[]
for name in ['twelfth_flat_support','twelfth_roll_flat','twelfth_roll_flat_crouch8',
             'twelfth_flat_roll_heading','twelfth_flat_roll_heading_crouch12']:
 path=p/(name+'.json')
 if path.exists():rejected.append(dict(candidate=name,report=json.loads(path.read_text()),physical_trial_run=False))
offline=[]
for name in ['step13_smooth5_upper_wrench_full_effort12','step13_smooth5_contact_wrench_fit',
             'step13_smooth5_contact_wrench_source_cost','step13_root_smooth6_feet_fixed',
             'step13_whole_body_fit','step13_whole_body_boundary','step13_whole_body_collision',
             'step13_whole_body_hard','step13_whole_body_geometric','step13_whole_body_proximal','step13_whole_body_dense','step13_turn_reference','step13_turn_reference_refine','step12_settled_support','step12_grounded_landing','step12_landing_bridge','step13_early_landing_bridge','step13_phase_lead12','step13_phase_lead12_clearance','step13_turn_boundary','step13_turn_wide','step13_turn_roll','turn_ninth_landing_bridge']:
 candidate_runs=[r['run'] for r in rows if (p/(r['run']+'.command.json')).exists() and any(str(arg).endswith('/'+name+'.npz') for arg in json.loads((p/(r['run']+'.command.json')).read_text())['command'])]
 row=dict(candidate=name,physical_trial_run=bool(candidate_runs),physical_trials=candidate_runs)
 for suffix in ['json','candidate_audit.json','collision_audit.json','contact_kinematics.json','interrupted.json','failure.json']:
  path=p/(name+'.'+suffix)
  if path.exists():row[suffix]=json.loads(path.read_text())
 if len(row)>3:offline.append(row)
report=dict(offline_candidates=offline,target='First 13 ordered landings with unchanged original PKL fidelity; every physical trial begins at frame 0.',
            entry='examples/test_g1.py',default_profile_unchanged=True,
            source_sha256=hashlib.sha256(Path('retarget/amp_reference/amp_walk_1ms.pkl').read_bytes()).hexdigest(),
            step13_passed=any(r['step13_passed'] for r in rows),
            trials=rows,rejected_reference_candidates=rejected,
            regression_tests=json.loads((p/'step13_tests.json').read_text()))
(p/'step13_progress.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(r['run'],r.get('measured_landings'),r.get('last_source_frame'),r['status'],r['step13_passed'])
