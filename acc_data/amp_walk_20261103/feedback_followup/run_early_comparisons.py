import subprocess,os,json
from pathlib import Path
p=Path(__file__).parent
env=os.environ.copy();env['G1_PYTHON']='/home/sky/anaconda3/envs/humanoid_fabric/bin/python'
cases=[('patch_first_after_three','run_patch_priority.sh',['--patch-priority-start','.88']),('leg_lower_after_three','run_leg_posture_lower.sh',['--leg-posture-lower-start','.88']),('clock085_after_three','run_smooth_clock.sh',['--clock-transition-start','.88','--clock-final-speed','.85','--clock-transition-ramp','.2']),('bspline_with_ankle_feedback','run_flight_feedback.sh',['--motion-interpolation','bspline-analytic'])]
for name,runner,args in cases:
 print(json.dumps(dict(starting=name)),flush=True)
 with (p/(name+'.log')).open('w') as log:
  result=subprocess.run(['bash',str(p/runner),'--flight-feedback-start','.5','--flight-feedback-limit','100','--duration','4','--output-path',str(p/(name+'.npz'))]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 report=p/(name+'.tracking.json');data=json.loads(report.read_text()) if report.exists() else {}
 print(json.dumps(dict(finished=name,return_code=result.returncode,measured_landings=data.get('measured_landings'),full_tracking_passed=data.get('full_tracking_passed'),failed_time_s=data.get('failed_time_s'))),flush=True)
