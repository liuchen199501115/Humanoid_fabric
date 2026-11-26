"""Run up to three explicit, archived trial specifications without a shell."""
import json,re,subprocess,sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
specs=json.loads(Path(sys.argv[1]).read_text())
if not isinstance(specs,list) or not 1<=len(specs)<=3:
    raise ValueError('Expected one to three trial specifications')
commands=[]
for spec in specs:
    name,parent=spec['name'],spec['parent']
    if not all(isinstance(s,str) and re.fullmatch(r'[a-z0-9_]+',s) for s in (name,parent)):
        raise ValueError('Trial names must be simple workspace-local identifiers')
    extra=spec['options']
    if not isinstance(extra,list) or not all(isinstance(s,str) for s in extra):
        raise ValueError('Options must be explicit strings')
    options=json.loads((OUT/(parent+'.result.json')).read_text())['options']+extra
    commands.append([sys.executable,str(OUT/'run_trial.py'),name,*options])
processes=[subprocess.Popen(command) for command in commands]
codes=[process.wait() for process in processes]
if any(codes):
    raise SystemExit('A trial runner failed: '+str(codes))
