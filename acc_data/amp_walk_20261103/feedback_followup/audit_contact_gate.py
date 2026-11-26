import ast,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from landing_gated_unload import intended_unload_mode,anticipatory_unload_mode
p=Path(__file__).parent
root=ast.parse(Path('examples/test_g1.py').read_text());function=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name=='update_contact_mode_state_machine');namespace={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),'contact_state','exec'),namespace);update=namespace[function.name]
tracker=SimpleNamespace(complete=False,events=[dict(support=0,takeoff_frame=780,end_frame=802)],landings=[],airborne=True,tolerance=.04,loaded_seconds=0.)
feet=np.array([[0.,0.,0.],[0.,0.,0.]]);source=np.array([[0.,0.,0.],[0.,.2,0.]])
plan,reason=intended_unload_mode(tracker,810,feet,source,2);assert plan==1
mode,counter=update(torch.tensor([3]),torch.tensor([plan]),torch.tensor([[True,True]]),torch.tensor([0]),1,torch.tensor([[300.,40.]]));assert int(mode)==1
# A substantial premature touchdown is still modeled as physical contact.
mode,counter=update(mode,torch.tensor([plan]),torch.tensor([[True,True]]),counter,1,torch.tensor([[200.,150.]]));assert int(mode)==3
# Losing the old support must override the controller plan.
mode,counter=update(torch.tensor([1]),torch.tensor([plan]),torch.tensor([[False,True]]),counter,1,torch.tensor([[0.,250.]]));assert int(mode)==2
source[:]=feet;plan,reason=intended_unload_mode(tracker,810,feet,source,2);assert plan==3
tracker.complete=True;assert intended_unload_mode(tracker,810,feet,source,2)[0]==2
tracker=SimpleNamespace(complete=False,events=[dict(support=1,takeoff_frame=485,end_frame=549),dict(support=0,takeoff_frame=572,end_frame=640)],landings=[],airborne=True,tolerance=.04,loaded_seconds=.02)
feet=np.zeros((2,3));source=feet.copy()
assert anticipatory_unload_mode(tracker,566,feet,source,[200.,50.],200.,3)==(1,True)
tracker.landings=[dict(step=1)];tracker.airborne=False;tracker.loaded_seconds=0.
assert anticipatory_unload_mode(tracker,568,feet,source,[200.,50.],200.,3)==(1,True)
assert anticipatory_unload_mode(tracker,568,feet,source,[10.,200.],200.,3)==(3,False)
source[0,1]=.04
assert anticipatory_unload_mode(tracker,568,feet,source,[200.,50.],200.,3)==(3,False)
source[:]=feet;tracker.landings=[];tracker.airborne=True;tracker.loaded_seconds=.02
assert anticipatory_unload_mode(tracker,566,feet,source,[200.,50.],200.,3,hysteresis=True)==(1,True)
assert anticipatory_unload_mode(tracker,568,feet,source,[50.,200.],200.,3,hysteresis=True)==(1,True)
tracker.landings=[dict(step=1)];tracker.airborne=False
assert anticipatory_unload_mode(tracker,574,feet,source,[80.,200.],200.,3,hysteresis=True)==(1,True)
assert anticipatory_unload_mode(tracker,575,feet,source,[10.,200.],200.,3,hysteresis=True)==(3,False)
tracker.landings=[];tracker.airborne=True
assert anticipatory_unload_mode(tracker,566,feet,source,[200.,50.],200.,3,hysteresis=True)==(1,True)
source[0,1]=.045
assert anticipatory_unload_mode(tracker,568,feet,source,[200.,50.],200.,3,hysteresis=True)==(3,False)
report=dict(incorrect_landing_retains_previous_support=True,premature_loaded_collision_still_constrained=True,lost_support_never_claimed=True,correct_landing_allows_double_support_hold=True,completed_tracker_uses_source_plan=True,anticipation_survives_verified_landing_transition=True,anticipation_checks_actual_load_and_source_error=True,physical_tracking_certified=False,optional_hysteresis_survives_load_fluctuations=True,hysteresis_releases_on_support_loss_or_source_error=True)
(p/'contact_gate_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
