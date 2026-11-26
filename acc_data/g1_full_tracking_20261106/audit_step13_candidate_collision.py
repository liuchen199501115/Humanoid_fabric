"""Check modified reference mesh pairs without stepping physics."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,torch,warp as wp
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
parser=argparse.ArgumentParser();parser.add_argument('candidate');parser.add_argument('--name',required=True);parser.add_argument('--spacing',type=float,default=1.);parser.add_argument('--start-frame',type=float,default=900.);parser.add_argument('--end-frame',type=float,default=1100.);args=parser.parse_args()
if not 0 <= args.start_frame < args.end_frame < 2000:raise ValueError('Invalid audit frame range')
if not .125<=args.spacing<=1.:raise ValueError('Audit spacing must be between 0.125 and 1 source frame')
p=Path(__file__).parent;report_path=p/(args.name+'.collision_audit.json')
if report_path.exists():raise RuntimeError('Existing audit report')
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
tasks=fabric.dynamics_tasks;tasks.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
candidate=np.load(args.candidate);q=candidate['control_q'] if 'control_q' in candidate else candidate['rejected_or_unexecuted_control_q'];q=torch.tensor(q.astype(np.float32),dtype=torch.float64)
sampler=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
rows=[]
for f in np.arange(args.start_frame,args.end_frame+1e-5,args.spacing):
 pose=sampler.kinematics(float(f),240.)[0].float();snapshot=tasks.model.snapshot(pose)
 sphere,_=tasks._sphere_geometry(snapshot);convex,_=tasks.convex_wrist_geometry.evaluate(snapshot)
 d=convex[0].detach().numpy();j=int(d.argmin());base_sphere=sphere[0,:tasks.original_collision_pairs].detach().numpy();k=int(base_sphere.argmin())
 rows.append(dict(frame=float(f),minimum_wrist_body_convex_distance_m=float(d[j]),mesh_pair=tasks.convex_wrist_geometry.pairs[j],minimum_original_sphere_distance_m=float(base_sphere[k]),original_sphere_pair=tasks.pairs[k].tolist()))
report=dict(candidate=args.candidate,candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),scope='Offline original sphere pairs and added convex wrist/body pairs; not a simulation or all-link collision certificate.',sample_count=len(rows),frame_range=[args.start_frame,args.end_frame],frame_spacing=args.spacing,worst_convex=min(rows,key=lambda r:r['minimum_wrist_body_convex_distance_m']),worst_original_sphere=min(rows,key=lambda r:r['minimum_original_sphere_distance_m']),physical_tracking_certified=False)
report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
