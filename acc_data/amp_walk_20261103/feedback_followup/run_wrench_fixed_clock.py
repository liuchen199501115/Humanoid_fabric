import subprocess,os,json,importlib.util
from pathlib import Path
p=Path(__file__).parent;env=os.environ.copy();env['G1_PYTHON']='/home/sky/anaconda3/envs/humanoid_fabric/bin/python'
spec=importlib.util.spec_from_file_location('summarizer',p.parent.parent/'amp_walk_20261102/contact_clock_followup/summarize_trials.py');summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary);summary.ROOT=p.resolve()
common=['--smooth-start','.65','--smooth-ramp','.1','--foot-feedback-joints','leg','leg','--foot-feedback-limits','150','150']
arms=['--global-posture-feedback-start','.95','--global-posture-feedback-joints','arms','--global-posture-feedback-limit','200','--global-posture-feedback-metric-scale','4']
base=common+arms+['--reference-pose-candidate',str(p/'early_flat_lower15.npz'),'--tracking-tube-start','.95','--relax-joint-tube','--reference-correction-rmse-limit','.08','--smooth-stride','6','--tube-smooth-stride','2','--late-leg-feedback-start','100','--clock-transition-start','1.33','--clock-transition-ramp','.2']
cases=[('fixed_clock70','run_tracking_tube_clock.sh',base+['--clock-final-speed','.7']),('fixed_clock60','run_tracking_tube_clock.sh',base+['--clock-final-speed','.6']),('fixed_clock70_fullpose','run_tracking_tube_clock.sh',base+['--clock-final-speed','.7','--full-swing-pose-start','1.4'])]
for name,runner,args in cases:
 print(json.dumps(dict(starting=name)),flush=True)
 with (p/(name+'.log')).open('w') as log:
  result=subprocess.run(['bash',str(p/runner),'--foot-feedback-starts','1.3','.88','--foot-feedback-limits','150','60','--foot-feedback-joints','leg','ankles','--flight-feedback-ramp','.04','--vertical-feedback-start','1.55','--waist-feedback-start','100','--stop-at-fidelity-loss','--duration','18','--output-path',str(p/(name+'.npz'))]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 report=p/(name+'.tracking.json');data=json.loads(report.read_text()) if report.exists() else {};row=summary.summarize(name) if report.exists() else {}
 print(json.dumps(dict(finished=name,return_code=result.returncode,measured_landings=data.get('measured_landings'),strict_prefix=row.get('prefix_fidelity_passing_landings'),first_fidelity_failure=row.get('first_fidelity_exceeded_time_s'),full_tracking_passed=data.get('full_tracking_passed'),full_motion_tracking_passed=data.get('full_motion_tracking_passed'))),flush=True)
 if data.get('full_motion_tracking_passed',data.get('full_tracking_passed',False)):break

