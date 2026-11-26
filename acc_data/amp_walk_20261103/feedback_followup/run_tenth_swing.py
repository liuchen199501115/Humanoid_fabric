import subprocess,os,json,importlib.util
from pathlib import Path
p=Path(__file__).parent;env=os.environ.copy();env['G1_PYTHON']='/home/sky/anaconda3/envs/humanoid_fabric/bin/python'
spec=importlib.util.spec_from_file_location('summarizer',p.parent.parent/'amp_walk_20261102/contact_clock_followup/summarize_trials.py');summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary);summary.ROOT=p.resolve()
cases=[('nine_rightleg150_390', 'run_tracking_tube.sh', ['--reference-correction-rmse-limit', '.08', '--smooth-stride', '6', '--tube-smooth-stride', '2', '--late-leg-feedback-start', '3.9', '--late-leg-feedback-limit', '150']), ('nine_rightleg300_390', 'run_tracking_tube.sh', ['--reference-correction-rmse-limit', '.08', '--smooth-stride', '6', '--tube-smooth-stride', '2', '--late-leg-feedback-start', '3.9', '--late-leg-feedback-limit', '300']), ('nine_swingpose390', 'run_tracking_tube.sh', ['--reference-correction-rmse-limit', '.08', '--smooth-stride', '6', '--tube-smooth-stride', '2', '--late-leg-feedback-start', '100', '--full-swing-pose-start', '3.9'])]
for name,runner,args in cases:
 print(json.dumps(dict(starting=name)),flush=True)
 with (p/(name+'.log')).open('w') as log:
  result=subprocess.run(['bash',str(p/runner),'--foot-feedback-starts','1.3','.88','--foot-feedback-limits','150','60','--foot-feedback-joints','leg','ankles','--flight-feedback-ramp','.04','--vertical-feedback-start','1.55','--waist-feedback-start','100','--stop-at-fidelity-loss','--duration','10.1','--output-path',str(p/(name+'.npz'))]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 report=p/(name+'.tracking.json');data=json.loads(report.read_text()) if report.exists() else {};row=summary.summarize(name) if report.exists() else {}
 print(json.dumps(dict(finished=name,return_code=result.returncode,measured_landings=data.get('measured_landings'),strict_prefix=row.get('prefix_fidelity_passing_landings'),first_fidelity_failure=row.get('first_fidelity_exceeded_time_s'),full_tracking_passed=data.get('full_tracking_passed'))),flush=True)
 if row.get('prefix_fidelity_passing_landings',0)>9:break
