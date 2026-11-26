"""Export a bounded reference experiment, explicitly without a dynamics certificate."""
from pathlib import Path
import argparse,json,pickle,hashlib,shutil
import numpy as np,torch
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
from fabrics_sim.reference.interpolation import MotionInterpolator
parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--name',required=True);args=parser.parse_args();p=Path(__file__).parent
if (p/(args.name+'.json')).exists():raise RuntimeError('Existing report')
copy=p/(args.name+'.checkpoint.npz');shutil.copyfile(args.checkpoint,copy);c=np.load(copy);q=c['rejected_or_unexecuted_control_q'].astype(np.float32);r=np.load(p/'rolling_contact.npz');source=np.load(p/'rolling_contact.control_reference.npz')['source_q'].copy();raw=pickle.load(open('retarget/amp_reference/amp_walk_1ms.pkl','rb'));order=json.load(open('acc_data/amp_walk_20261102/input_report.json'))['source_joint_order'];source[:,7:]=raw['dof_pos'][:,[order.index(n) for n in r['final_joint_order']]];t=torch.tensor(q,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
report=dict(checkpoint=str(copy),checkpoint_sha256=hashlib.sha256(copy.read_bytes()).hexdigest(),scope='Kinematically bounded experimental reference. Nominal dynamics search is unfinished and no nominal dynamics or physical tracking certificate is claimed.',peak_raw_joint_rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=1)).max()),peak_raw_root_position_error_m=float(np.linalg.norm(q[:,:3]-source[:,:3],axis=1).max()),peak_raw_root_orientation_error_rad=float((Rotation.from_quat(q[:,3:7])*Rotation.from_quat(source[:,3:7]).inv()).magnitude().max()),physical_tracking_certified=False,nominal_dynamics_passed=False)
report.update(joint_extrema(curve.smooth,690,1098,240.,r['joint_position_limits'][0],r['history_joint_velocity_limits'][-1].reshape(-1)))
report['reference_pose_bounds_passed']=bool(report['peak_raw_joint_rmse']<.19 and report['peak_raw_root_position_error_m']<.16 and report['peak_raw_root_orientation_error_rad']<.15 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1)
(p/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
if report['reference_pose_bounds_passed']:
 parent=np.load(p/'step13_leg_smooth5_clearance_audit.npz');np.savez_compressed(p/(args.name+'.npz'),control_q=q,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2))
