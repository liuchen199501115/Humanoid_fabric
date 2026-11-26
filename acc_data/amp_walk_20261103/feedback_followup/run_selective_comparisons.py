import subprocess,os,json
from pathlib import Path
p=Path(__file__).parent;env=os.environ.copy();env['G1_PYTHON']='/home/sky/anaconda3/envs/humanoid_fabric/bin/python'
cases=[('left_ankle_feedback',['--flight-feedback-side','left']),('ankle_feedback_through_four',['--flight-feedback-end','1.34']),('ankle_feedback_through_three',['--flight-feedback-end','.88'])]
for name,args in cases:
 print(json.dumps(dict(starting=name)),flush=True)
 with (p/(name+'.log')).open('w') as log:
  result=subprocess.run(['bash',str(p/'run_selective_feedback.sh'),'--flight-feedback-start','.5','--flight-feedback-limit','100','--duration','5','--output-path',str(p/(name+'.npz'))]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
 report=p/(name+'.tracking.json');data=json.loads(report.read_text()) if report.exists() else {}
 print(json.dumps(dict(finished=name,return_code=result.returncode,measured_landings=data.get('measured_landings'),full_tracking_passed=data.get('full_tracking_passed'),failed_time_s=data.get('failed_time_s'))),flush=True)
