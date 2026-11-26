from pathlib import Path
import json,numpy as np
p=Path(__file__).parent
ns={'__file__':str(p/'prepare_spline_fast_contact_acc.py'),'__name__':'contact_audit'}
s=(p/'prepare_spline_fast_contact_acc.py').read_text().split('for iteration in range(args.iterations):')[0]
exec(compile(s,str(p/'prepare_spline_fast_contact_acc.py'),'exec'),ns)
d=np.load(p/'contact_acc_spline_candidate.npz');q,v,a=[d[k] for k in ('q','v','a')];st=ns['snapshot'](q);ns['update_support'](st);ns['dynamics'](st,q,v,a)
ca=ns['contact_acceleration_maps'](st,ns['last_bias_states'][0],ns['last_bias_states'][1],v,a);rows=[]
for k in range(len(q)):
 for side in range(2):
  if ns['contact'][k,side]:
   err=abs(ca[k,side,ns['touch'][k][side]]).max()
   rows.append(dict(frame=k,side=side,peak_acceleration=float(err)))
rows=sorted(rows,key=lambda x:x['peak_acceleration'],reverse=True)
print(json.dumps(rows[:25]),flush=True)
