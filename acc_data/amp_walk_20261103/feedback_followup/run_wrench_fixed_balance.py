import subprocess,os,json,importlib.util
from pathlib import Path
p=Path(__file__).parent;env=os.environ.copy();env['G1_PYTHON']='/home/sky/anaconda3/envs/humanoid_fabric/bin/python'
spec=importlib.util.spec_from_file_location('summarizer',p.parent.parent/'amp_walk_20261102/contact_clock_followup/summarize_trials.py');summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary);summary.ROOT=p.resolve()
common=['--global-posture-feedback-start','.70','--global-posture-feedback-limit','150']
common+=['--smooth-start','.65','--smooth-ramp','.1','--foot-feedback-joints','leg','leg','--foot-feedback-limits','150','150','--relax-joint-tube']
base=common+['--reference-pose-candidate',str(p/'early_flat_lower15.npz'),'--landing-gated-unload-start','1.48','--gate-contact-transitions']
xy=['--horizontal-feedback-start','1.4','--horizontal-feedback-limit','80','--horizontal-feedback-metric-scale','4','--vertical-feedback-start','1.4']
base+=xy+['--full-swing-pose-start','1.4']
base+=['--recover-orientation-tube']
cases=[('fixed_balance30_roll60','run_nine_strict.sh',base+['--reference-pose-candidate',str(p/'early_balance30_roll60.npz')]),('fixed_balance30_roll100','run_nine_strict.sh',base+['--reference-pose-candidate',str(p/'early_balance30_roll100.npz')]),('fixed_balance30_roll100_mpc','run_nine_strict.sh',base+['--reference-pose-candidate',str(p/'early_balance30_roll100.npz'),'--centroidal-preview-walking','--centroidal-preview-start','.95','--com-feedback-metric-start','.95','--com-feedback-metric-scale','2'])]
for name,runner,args in cases:
 if 'general15' in name and not json.loads((p/'contact_compatible_b15.json').read_text())['preflight_passed']:raise RuntimeError('Reference preflight failed')
 print(json.dumps(dict(starting=name)),flush=True)
 with (p/(name+'.log')).open('w') as log:
  result=subprocess.run(['bash',str(p/runner),'--foot-feedback-starts','1.3','.88','--foot-feedback-limits','150','60','--foot-feedback-joints','leg','ankles','--flight-feedback-ramp','.04','--vertical-feedback-start','1.55','--waist-feedback-start','100','--stop-at-fidelity-loss','--duration','15','--output-path',str(p/(name+'.npz'))]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 report=p/(name+'.tracking.json');data=json.loads(report.read_text()) if report.exists() else {};row=summary.summarize(name) if report.exists() else {}
 print(json.dumps(dict(finished=name,return_code=result.returncode,measured_landings=data.get('measured_landings'),strict_prefix=row.get('prefix_fidelity_passing_landings'),first_fidelity_failure=row.get('first_fidelity_exceeded_time_s'),full_tracking_passed=data.get('full_tracking_passed'),full_motion_tracking_passed=data.get('full_motion_tracking_passed'))),flush=True)
 if data.get('full_motion_tracking_passed',data.get('full_tracking_passed',False)):break

