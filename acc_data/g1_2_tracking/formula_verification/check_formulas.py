"""Small offline checks for the formulas in examples/test_g1_2.py; no Isaac startup."""
import ast,math,json,argparse,sys,io,contextlib,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
root=Path(__file__).resolve().parents[3]
p=root/'examples/test_g1_2.py';tree=ast.parse(p.read_text())
ns=dict(math=math,torch=torch,argparse=argparse,sys=sys,Path=Path,__file__=str(p),OUTPUT_PATH=root/'acc_data/check.npz')
names={'PhaseTrackingLaw','solve_consistent_mit','parse_args','configure_tracking_profile'}
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name',None) in names],type_ignores=[]),str(p),'exec'),ns)
checks=[]
law=ns['PhaseTrackingLaw'](maximum=4);e=torch.zeros(1,3);v=torch.zeros_like(e)
w,a,r=law.update([(e,v)],[1.],.002)
assert torch.equal(w,torch.ones_like(w)) and r.item()==0;checks.append('zero_error_neutral_weights')
w,a,r=law.update([(e+2,v)],[1.],.002)
assert 1<w.item()<4 and 0<a.item()<1;checks.append('bounded_smooth_response')
for _ in range(2000):w,a,r=law.update([(e+2,v)],[1.],.002)
assert abs(w.item()-4)<1e-5 and abs(a.item()-1)<1e-5;checks.append('bounded_saturation')
try:law.update([(e+float('nan'),v)],[1.],.002)
except ValueError:checks.append('nonfinite_state_rejected')
else:raise AssertionError('NaN accepted')
zero=torch.zeros(1,35);sync=torch.full((1,29),3.);b=torch.full((1,29),.5);g=torch.full((1,29),.2)
def qp(**kw):
 c=kw['command_acc_correction'];assert c[:,:6].count_nonzero()==0
 acc=torch.zeros_like(c);acc[:,6:]=(2+c[:,6:]-.3*b)/(1+.3*g)
 pred=acc-c;pred[:,6:]+=.3*(b+g*acc[:,6:])
 return SimpleNamespace(acc=acc,predicted_acc=pred,validated=torch.tensor([True]))
d={};r=ns['solve_consistent_mit'](qp,sync,d,acc_reference=zero,command_acc_correction=zero,servo_bias=b,servo_acc_gain=g)
assert d['converged'] and torch.max(abs(r.acc[:,6:]-5))<.05 and torch.max(abs(r.predicted_acc[:,6:]-2))<1e-5
assert zero.count_nonzero()==0;checks.append('analytical_MIT_fixed_point_and_zero_base_correction')
def stuck(**kw):return SimpleNamespace(acc=zero,predicted_acc=zero+1,validated=torch.tensor([True]))
d={};ns['solve_consistent_mit'](stuck,torch.zeros_like(sync),d,acc_reference=zero,command_acc_correction=zero,servo_bias=b,servo_acc_gain=g)
assert not d['converged'] and d['iterations']==16;checks.append('nonconvergence_reported')
for profile in ('twelve-wbc','twelve-mit','formula-mit'):
 sys.argv=[str(p),'--tracking-profile',profile];args=ns['parse_args']()
 assert args.track_full_motion and args.tracking_steps==0 and args.stop_at_fidelity_loss
 assert args.direct_wbc==(profile=='twelve-wbc')
 assert args.full_body_preview_start==float('inf') and args.centroidal_preview_start==float('inf')
checks.append('profile_execution_and_full_clip_guards')
for flags in (['--direct-wbc'],['--tracking-steps','2'],['--mit-consistent-sync','--mit-sync-limit','0']):
 sys.argv=[str(p),'--tracking-profile','formula-mit',*flags]
 try:
  with contextlib.redirect_stderr(io.StringIO()):ns['parse_args']()
 except SystemExit as exc:assert exc.code==2
 else:raise AssertionError('Invalid profile flags accepted')
checks.append('invalid_profile_combinations_rejected')
assert hashlib.sha256((root/'examples/test_g1.py').read_bytes()).hexdigest()=='0968fe197a771461df88a80303dd67e1654ac4d3c2aee8cad55560f5e13c7264'
checks.append('original_test_g1_unchanged')
report=dict(passed=True,checks=checks,target_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(__file__).with_name('formula_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
