"""Offline first-step optimization using the full robot model.

--record uses synchronized measured unloading q/v and WBC predicted acceleration.
Without it, historical lifted-pose zero rates remain an explicit hypothesis.
Neither input resets a live robot or certifies physical tracking.
"""
import json
from pathlib import Path
import numpy as np
import torch,warp as wp
import xml.etree.ElementTree as ET
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.source_step import SourceStepCandidate
from fabrics_sim.reference.source_step_optimization import evaluate_candidate,constraint_score,constraint_family_penalty
from fabrics_sim.reference.collision_audit import audit_collision_rows
from test_g1_two_tasks import G1_JOINT_NAMES,load_motion_trajectory,unitree_g1_effort_limits
from g1_armature import unitree_armatures
import argparse
parser=argparse.ArgumentParser(description='Offline full-model first-step reference optimization; no simulation reset')
parser.add_argument('--method',choices=['Powell','COBYLA'],default='Powell',help='Search method; COBYLA uses parameters normalized by their bounds')
parser.add_argument('--balanced-peaks',action='store_true',help='Minimize relative constraint-family peaks against the initial candidate')
parser.add_argument('--acceleration-and-braking-limits',action=argparse.BooleanOptionalAction,default=False,help='Opt in to software acceleration and joint braking checks')
parser.add_argument('--effort-limit-scale',type=float,default=.8,help='Fraction of online simulated motor effort limits; .8 matches WBC reserve')
parser.add_argument('--projection-probe-search',action='store_true',help='Optimize non-torque preflight gates for bounded WBC projection diagnostics; still report true torque violations')
parser.add_argument('--extended-reference-search', action='store_true', help='Broaden COM reference search only; physical limits and landing targets stay unchanged')
parser.add_argument('--audit-only',action='store_true',help='Evaluate the supplied parameter vector once without invoking an optimizer')
parser.add_argument('--search-radius',type=float,default=None,help='Initial normalized COBYLA trust radius')
parser.add_argument('--search-tolerance',type=float,default=1e-4,help='Normalized COBYLA convergence tolerance; physical acceptance thresholds stay unchanged')
parser.add_argument('--max-evaluations',type=int,default=600)
parser.add_argument('--start-report',type=Path)
parser.add_argument('--output',type=Path,default=Path('acc_data/walk1_full_body_optimization_20261011.json'))
parser.add_argument('--refine-report',type=Path,help='Include failure times and neighbours from an audit of the same duration')
parser.add_argument('--samples',type=int,default=21)
parser.add_argument('--record',type=Path,help='Use a passed unloading record with synchronized measured q/v and WBC predicted acceleration')
parser.add_argument('--duration',type=float,default=1.2)
parser.add_argument('--pelvis-parameters',action='store_true',help='Optimize terminal pelvis roll and interior roll together with COM/body parameters')
parser.add_argument('--body-parameters',action='store_true',help='Also optimize terminal COM height offset and pelvis pitch')
parser.add_argument('--interior',action='store_true',help='Optimize four interior horizontal COM shape coefficients as well')
parser.add_argument('--early-com-parameters',action='store_true',help='Optimize a C2 horizontal COM offset confined to the first 60 percent of the step')
parser.add_argument('--late-com-parameters',action='store_true',help='Optimize a C2 XYZ COM offset in the final 55 percent of the step')
parser.add_argument('--pelvis-roll-bump',type=float,default=None,help='C2 local pelvis-roll offset in the middle of the step')
parser.add_argument('--pelvis-roll',type=float,default=None,help='Terminal local pelvis roll offset in radians; foot targets are unchanged')
parser.add_argument('--arm-clearance-offset',type=float,default=None,help='Symmetric shoulder-roll reference offset in radians; retain original foot targets')
parser.add_argument('--source-clearance-bump',type=float,default=None,help='C2 mid-swing vertical lift in metres; preserve source landing pose and endpoint derivatives')
parser.add_argument('--landing-rate',type=float,default=None,help='Terminal clock rate in [0,1]; preserve spatial path and initial q/v/a')
args=parser.parse_args()
if not np.isfinite(args.effort_limit_scale) or not 0 < args.effort_limit_scale <= 1:
 parser.error('Effort-limit scale must be finite in (0,1]')
if not np.isfinite(args.search_tolerance) or args.search_tolerance<=0:parser.error('Search tolerance must be positive and finite')
if args.search_radius is not None and (not np.isfinite(args.search_radius) or args.search_radius<=0):parser.error('Search radius must be positive and finite')
resume_report=json.loads(args.start_report.read_text()) if args.start_report else {}
if args.arm_clearance_offset is None:args.arm_clearance_offset=resume_report.get('options',{}).get('arm_clearance_offset',0.)
if not np.isfinite(args.arm_clearance_offset):parser.error('Arm clearance offset must be finite')
if args.source_clearance_bump is None:args.source_clearance_bump=resume_report.get('options',{}).get('source_clearance_bump',0.)
if not np.isfinite(args.source_clearance_bump) or args.source_clearance_bump<0:parser.error('Source clearance bump must be nonnegative and finite')
if args.landing_rate is None:args.landing_rate=resume_report.get('landing_rate',1.)
if args.pelvis_roll_bump is None:args.pelvis_roll_bump=resume_report.get('options',{}).get('pelvis_roll_bump',0.)
if not np.isfinite(args.pelvis_roll_bump):parser.error('Pelvis roll bump must be finite')
if args.pelvis_roll is None:args.pelvis_roll=resume_report.get('options',{}).get('pelvis_roll_offset',0.)
if not np.isfinite(args.pelvis_roll):parser.error('Pelvis roll must be finite')
if not np.isfinite(args.landing_rate) or not 0 <= args.landing_rate <= 1:parser.error('Landing rate must be in [0,1]')
if args.pelvis_parameters and not args.body_parameters:parser.error('--pelvis-parameters requires --body-parameters')
if args.late_com_parameters and not args.early_com_parameters:parser.error('--late-com-parameters requires --early-com-parameters')
if args.early_com_parameters and not args.interior:parser.error('--early-com-parameters requires --interior')
if args.max_evaluations < 1 or args.samples < 3 or not np.isfinite(args.duration) or args.duration <= 0:parser.error('Need positive evaluations and at least three samples')
from fabrics_sim.reference.audit_sampling import refined_sample_times
refinement=json.loads(args.refine_report.read_text()) if args.refine_report else None
optimization_times=refined_sample_times(args.duration,args.samples,refinement)
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
f.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True);f.dynamics_tasks.wrist_hip_barrier=True
if args.record is not None:
 from analyze_contact_transfer import summarize_unloading
 with np.load(args.record) as record:
  if not summarize_unloading(record)['unloading_passed']:raise ValueError('Unloading record must pass its audit')
  q=torch.tensor(record['history_q'][-1])
  initial=WholeBodyReference(q,torch.tensor(record['history_qd'][-1]),torch.tensor(record['history_qdd_predicted'][-1]))
  ground_shift=float(record['motion_ground_shift'])
 boundary=dict(record=str(args.record),velocity='measured',acceleration='WBC prediction, not measured')
else:
 with np.load('acc_data/walk1_strict_four_transfer_20261008.reference_failure.npz') as record:q=torch.tensor(record['q'])
 initial=WholeBodyReference(q,torch.zeros(1,35),torch.zeros(1,35))
 ground_shift=-.021463990211486816
 boundary=dict(record='acc_data/walk1_strict_four_transfer_20261008.reference_failure.npz',velocity='zero assumption',acceleration='zero assumption')
source,_=load_motion_trajectory(Path('retarget/lafan/walk1_g1.pkl'))
source[:,2]+=ground_shift
model=f.dynamics_tasks.model;idx=model.kinematics.get_link_index('right_ankle_roll_link')
goal=model.snapshot(source[129:130])['transforms'][:,idx].clone()
limits={n.attrib['name']:n.find('limit') for n in ET.parse(f.urdf_path).getroot().findall('joint')}
velocity=q.new_tensor([float(limits[n].attrib['velocity']) for n in G1_JOINT_NAMES]);effort=q.new_tensor(unitree_g1_effort_limits())
bounds=dict(lower=q.new_tensor([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES]),upper=q.new_tensor([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES]),dt=.002,bandwidth=8.)
dyn=dict(armature=q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),effort=args.effort_limit_scale*effort)
if args.acceleration_and_braking_limits:dyn.update(joint_bounds=bounds,acceleration=5*f.accel_limits)
else:f.fabric_params['joint_limits']['active']=False
from scipy.optimize import minimize
from fabrics_sim.reference.full_body_com_path import EndpointComPath, InteriorComPath, WindowedComPath
body=source.clone();body[:,19:]=q[:,19:]
options=dict(arm_clearance_offset=args.arm_clearance_offset,source_clearance_bump=args.source_clearance_bump,pelvis_roll_bump=args.pelvis_roll_bump,pelvis_roll_offset=args.pelvis_roll,duration=args.duration,body_join='endpoint',pelvis_pitch_offset=-.15,
             com_preview='full_body_source',landing_fraction=1.,landing_search_depth=0.)
plan=SourceStepCandidate(f,initial,G1_JOINT_NAMES,body,114,129,0,goal,**options)
baseline=plan.preview
best=None;history=[]
reference_peaks=None
objective_version=('bounded_projection_non_torque_v1' if args.projection_probe_search else 'relative_family_peaks_v4' if args.balanced_peaks else 'max_plus_family_mean_squared_violation_v3')
report_path=args.output
report_path.parent.mkdir(parents=True,exist_ok=True)
def evaluate(x,dense=False):
 global plan,baseline,reference_peaks
 if args.body_parameters:
  body_options=dict(options,pelvis_pitch_offset=float(x[-1]))
  if args.pelvis_parameters:body_options.update(pelvis_roll_offset=float(x[-4]),pelvis_roll_bump=float(x[-3]))
  plan=SourceStepCandidate(f,initial,G1_JOINT_NAMES,body,114,129,0,goal,**body_options)
  baseline=plan.preview
 height=float(x[-2]) if args.body_parameters else -.06
 plan.preview=EndpointComPath(baseline,[x[0],x[1],height],[x[2],x[3],0.],[x[4],x[5],0.])
 if args.interior:
  plan.preview=InteriorComPath(plan.preview,[[x[6],x[7],0.],[x[8],x[9],0.]])
 if args.early_com_parameters:
  plan.preview=WindowedComPath(plan.preview,[x[10],x[11],0.],0.,.6*plan.duration)
 if args.late_com_parameters:
  plan.preview=WindowedComPath(plan.preview,x[12:15],.45*plan.duration,plan.duration)
 from fabrics_sim.reference.retiming import LandingRetimedReference
 audited_plan=LandingRetimedReference(plan,args.landing_rate)
 rows=evaluate_candidate(audited_plan,(np.linspace(0,plan.duration,601) if dense else optimization_times),velocity,dyn)
 score=constraint_score(rows)
 entry={name:float((q.new_tensor(rows[0][name])-getattr(initial,name)).abs().max()) for name in ('q','v','a')}
 landed=model.snapshot(q.new_tensor(rows[-1]['q']))['transforms'][:,idx]
 endpoint_position=float((landed[:,:3]-goal[:,:3]).norm())
 endpoint_rotation=float(torch.minimum((landed[:,3:]-goal[:,3:]).norm(),(landed[:,3:]+goal[:,3:]).norm()))
 score=max(score,entry['q']/5e-6-1,entry['v']/1e-4-1,entry['a']/1e-3-1,endpoint_position/5e-6-1,endpoint_rotation/5e-6-1)
 collision=audit_collision_rows(f,rows,source)
 from analyze_contact_transfer import sole_clearances
 worst_wrench=max(rows,key=lambda r:r['exact_wrench_score'])
 worst_state=model.snapshot(q.new_tensor(worst_wrench['q']))
 feet_indices=[model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
 worst_sole_heights=sole_clearances(worst_state['transforms'][0,feet_indices].cpu().numpy()).tolist()
 # Retain the worst-case acceptance gate, but also reduce violations across
 # the trajectory so improving a non-worst sample is visible to the search.
 sampled_violation=np.array([constraint_score([row]) for row in rows])
 from fabrics_sim.reference.source_step_optimization import constraint_components,relative_peak_objective
 family_peaks={key:float(value.max()) for key,value in constraint_components(rows).items()}
 family_peaks['collision']=max(0.,collision['maximum_violation']-collision['acceleration_residual_tolerance'])
 family_peaks['boundary']=max(0.,entry['q']/5e-6-1,entry['v']/1e-4-1,entry['a']/1e-3-1,endpoint_position/5e-6-1,endpoint_rotation/5e-6-1)
 family_penalties=constraint_family_penalty(rows)
 distributed_penalty=sum(family_penalties.values())
 worst_row=rows[int(sampled_violation.argmax())]
 worst_sample={key:np.asarray(worst_row[key]).tolist() for key in ('time','cop','force','velocity_ratio','acceleration_ratio','torque_ratio','exact_wrench_score','joint_bound_score')}
 boundary_scores={'initial':float(sampled_violation[0]),'terminal':float(sampled_violation[-1])}
 value=score**2+distributed_penalty+max(0.,collision['maximum_violation'])**2
 if args.projection_probe_search:
  # A nominal torque failure remains in score/report and is never certified.
  # This search is only for the explicit, hard-limited WBC simulation probe.
  selected=[k for k in family_peaks if k not in ('torque','support_region')]
  value=max(family_peaks[k] for k in selected)**2+sum(
      family_penalties.get(k,0.) for k in selected)+max(0.,collision['maximum_violation'])**2
 elif args.balanced_peaks:
  if reference_peaks is None:reference_peaks=family_peaks.copy()
  value=relative_peak_objective(family_peaks,reference_peaks)
 return value,dict(score=score,constraint_family_peaks=family_peaks,constraint_family_penalties=family_penalties,worst_sample=worst_sample,boundary_scores=boundary_scores,distributed_constraint_penalty=distributed_penalty,collision=collision,worst_wrench_sole_heights_m=worst_sole_heights,entry_error=entry,endpoint_position_error_m=endpoint_position,endpoint_quaternion_error=endpoint_rotation,
    worst_joint_braking=dict(max(rows,key=lambda r:r['joint_bound_score'])['joint_bound_detail'],
        name=G1_JOINT_NAMES[max(rows,key=lambda r:r['joint_bound_score'])['joint_bound_detail']['index']]) if args.acceleration_and_braking_limits else None,
    peak_torque_joint=G1_JOINT_NAMES[max(rows,key=lambda r:r['torque_ratio'])['peak_torque_joint_index']],
    peak_wrench_constraint_index=worst_wrench['peak_wrench_constraint_index'],
    contact_motion_diagnostics=dict(
      scope='Reference stance kinematics, not a closed-loop stability certificate',
      component_tolerance=1e-3,
      all_samples_within_tolerance=all(max(r['support_contact_velocity_max'],r['support_contact_acceleration_max'])<=1e-3 for r in rows),
      worst_velocity_time=float(max(rows,key=lambda r:r['support_contact_velocity_max'])['time']),
      worst_acceleration_time=float(max(rows,key=lambda r:r['support_contact_acceleration_max'])['time']),
      after_0p2s_velocity_max=max((r['support_contact_velocity_max'] for r in rows if r['time']>=.2),default=None),
      after_0p2s_acceleration_max=max((r['support_contact_acceleration_max'] for r in rows if r['time']>=.2),default=None)),
    support_contact_velocity_max=max(r['support_contact_velocity_max'] for r in rows),
    support_contact_acceleration_max=max(r['support_contact_acceleration_max'] for r in rows),
    minimum_swing_sole_height_m=min(r['swing_sole_min_height'] for r in rows),
    swing_ground_clearance_passed=all(r['swing_sole_min_height']>=-2e-5 for r in rows),
    peak_torque_ratio=max(r['torque_ratio'] for r in rows),
    peak_wrench_score=max(r['exact_wrench_score'] for r in rows),
    max_joint_bound_score=max(r['joint_bound_score'] for r in rows),
    worst_constraint_times={key:float(max(rows,key=lambda r:r[key])['time']) for key in ('torque_ratio','exact_wrench_score','joint_bound_score')},
    goal_unchanged=bool(torch.equal(plan.landing_pose,goal)),sample_count=len(rows))
def objective(x):
 global best
 try:
  value,detail=evaluate(x)
 except (ValueError,RuntimeError) as e:
  value=100.;detail={'error':str(e)}
 row=dict(evaluation=len(history)+1,parameters=np.asarray(x).tolist(),objective=float(value),**detail)
 history.append(row)
 if best is None or value<best['objective']:
  best=row;print(json.dumps(row),flush=True)
  report_path.write_text(json.dumps(dict(scope='Full-model constrained-reference penalty search; no live tracking',
    effort_limit_source='unitree_g1_effort_limits',effort_limits=(args.effort_limit_scale*effort).tolist(),effort_joint_names=G1_JOINT_NAMES,effort_limit_scale=args.effort_limit_scale,software_acceleration_limits_enabled=args.acceleration_and_braking_limits,joint_braking_limits_enabled=args.acceleration_and_braking_limits,options=options,boundary=boundary,landing_rate=args.landing_rate,optimization_sample_times=optimization_times.tolist(),refinement_report=str(args.refine_report) if args.refine_report else None,late_com_parameters_enabled=args.late_com_parameters,early_com_parameters_enabled=args.early_com_parameters,interior_enabled=args.interior,pelvis_parameters_enabled=args.pelvis_parameters,body_parameters_enabled=args.body_parameters,projection_probe_search=args.projection_probe_search,optimizer_method=args.method,objective_version=objective_version,constraint_reference_peaks=reference_peaks,best=best,evaluations=len(history),simulation_validated=False),indent=2)+'\n')
 return value
start=np.array([.0122457445,.0656912327,0.,0.,0.,0.])
body_start=np.array([-.06,-.15])
pelvis_start=np.array([args.pelvis_roll,args.pelvis_roll_bump])
resume_interior=False
resume_early=False
resume_late=False
if args.start_report is not None:
 resume=resume_report
 start=np.asarray(resume['best']['parameters'],dtype=float)
 resume_interior=bool(resume.get('interior_enabled',False))
 resume_early=bool(resume.get('early_com_parameters_enabled',False))
 resume_late=bool(resume.get('late_com_parameters_enabled',False))
 if resume_late and not args.late_com_parameters:raise ValueError('Late COM resume requires --late-com-parameters')
 if resume_early and not args.early_com_parameters:raise ValueError('Early COM resume requires --early-com-parameters')
 if resume.get('body_parameters_enabled',False):
  if not args.body_parameters:raise ValueError('Body-parameter resume requires --body-parameters')
  body_start=start[-2:];start=start[:-2]
 if resume.get('pelvis_parameters_enabled',False):
  if not args.pelvis_parameters:raise ValueError('Pelvis resume requires --pelvis-parameters')
  pelvis_start=start[-2:];start=start[:-2]
 if resume_interior and not args.interior:raise ValueError('Interior resume requires --interior')
if args.interior and not resume_interior:start=np.r_[start,np.zeros(4)]
if args.early_com_parameters and not resume_early:start=np.r_[start,np.zeros(2)]
if args.late_com_parameters and not resume_late:start=np.r_[start,np.zeros(3)]
if args.pelvis_parameters:start=np.r_[start,pelvis_start]
if args.body_parameters:start=np.r_[start,body_start]
expected=(10 if args.interior else 6)+(2 if args.early_com_parameters else 0)+(3 if args.late_com_parameters else 0)+(2 if args.body_parameters else 0)+(2 if args.pelvis_parameters else 0)
if start.shape!=(expected,) or not np.isfinite(start).all():raise ValueError('Resume parameter shape or finiteness mismatch')
parameter_bounds=[(-.02,.05),(.02,.11),(-.35,.35),(-.35,.35),(-4.,4.),(-4.,4.)]+([(-.06,.06)]*4 if args.interior else [])
if args.early_com_parameters:parameter_bounds += [(-.03,.03)]*2
if args.late_com_parameters:parameter_bounds += [(-.03,.03)]*3
if args.pelvis_parameters:parameter_bounds += [(-.3,.3),(-.2,.2)]
if args.body_parameters:parameter_bounds += [(-.10,-.02),(-.35,.05)]
if args.extended_reference_search:
 parameter_bounds[:2]=[(-.15,.15),(-.05,.25)]
 if args.body_parameters:parameter_bounds[-2:]=[(-.20,.02),(-.60,.30)]
search_options=dict(maxfev=args.max_evaluations,maxiter=20,xtol=.002,ftol=.001)
if args.early_com_parameters:
 # Address the diagnosed early lateral violation before retuning endpoint terms.
 order=[11,10]+[i for i in range(expected) if i not in (10,11)]
 if args.late_com_parameters:order=[14,13,12]+[i for i in order if i not in (12,13,14)]
 search_options['direc']=np.eye(expected)[order]
if args.audit_only:
 objective(start)
 result=None
elif args.method=='COBYLA':
 lower=np.asarray([b[0] for b in parameter_bounds]);width=np.asarray([b[1]-b[0] for b in parameter_bounds])
 result=minimize(lambda z:objective(lower+z*width),(start-lower)/width,method='COBYLA',
     bounds=[(0.,1.)]*expected,options=dict(maxiter=args.max_evaluations,rhobeg=args.search_radius if args.search_radius is not None else .03 if args.extended_reference_search else .015,tol=args.search_tolerance,catol=1e-8))
else:
 result=minimize(objective,start,method='Powell',bounds=parameter_bounds,options=search_options)
report=dict(reference_parameter_bounds=parameter_bounds,extended_reference_search=args.extended_reference_search,scope='Full-model constrained-reference penalty search; no live tracking',effort_limit_source='unitree_g1_effort_limits',effort_limits=(args.effort_limit_scale*effort).tolist(),effort_joint_names=G1_JOINT_NAMES,effort_limit_scale=args.effort_limit_scale,software_acceleration_limits_enabled=args.acceleration_and_braking_limits,joint_braking_limits_enabled=args.acceleration_and_braking_limits,options=options,boundary=boundary,landing_rate=args.landing_rate,optimization_sample_times=optimization_times.tolist(),refinement_report=str(args.refine_report) if args.refine_report else None,late_com_parameters_enabled=args.late_com_parameters,early_com_parameters_enabled=args.early_com_parameters,interior_enabled=args.interior,pelvis_parameters_enabled=args.pelvis_parameters,body_parameters_enabled=args.body_parameters,projection_probe_search=args.projection_probe_search,optimizer_method=args.method,objective_version=objective_version,constraint_reference_peaks=reference_peaks,
 best=best,evaluations=len(history),optimizer_success=None if result is None else bool(result.success),
 optimizer_message=None if result is None else str(result.message),
 audit_only=args.audit_only,
 history=history,simulation_validated=False)
if args.audit_only:
 report['scope']='Fixed-parameter full-model reference audit; no optimization or live tracking'
 report['audit_sample_times']=report.pop('optimization_sample_times')
 report['optimization_sample_times']=[]
# A coarse penalty improvement is never a feasibility certificate.
if not args.audit_only and best.get('score')==0 and best.get('collision',{}).get('passed'):
 try:report['dense_validation']=evaluate(best['parameters'],True)[1]
 except (ValueError,RuntimeError) as e:report['dense_validation']={'error':str(e)}
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
