from pathlib import Path
import json,hashlib
p=Path(__file__).parent
namespace={'__file__':str(p/'prepare_spline_newton.py'),'__name__':'reference_export'}
s=(p/'prepare_spline_newton.py').read_text();s=s[:s.index('for iteration in range(args.iterations):')]
exec(compile(s,str(p/'prepare_spline_newton.py'),'exec'),namespace)
np=namespace['np'];Rotation=namespace['Rotation'];d=np.load(p/'newton_spline_candidate.npz');q=d['q'];st=namespace['snapshot'](q);namespace['update_support'](st);support=namespace['support'].copy();ids=namespace['ids'];points=namespace['points'];radii=namespace['radii'];contact=namespace['contact']
clearance=np.zeros((len(q),2))
for side,link in enumerate(ids):
 offsets=np.einsum('nij,pj->npi',st['rotations'][:,link],points[side]);clearance[:,side]=np.min(st['transforms'][:,link,2,None]+offsets[:,:,2]-radii[side],axis=1)
flights=[]
for side in range(2):
 active=contact[:,side];cuts=np.r_[0,np.flatnonzero(active[1:]!=active[:-1])+1,len(q)]
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if not active[lo]:flights.append(dict(side=side,first_frame=int(lo),last_frame=int(hi-1),peak_native_clearance_m=float(clearance[lo:hi,side].max())))
source=d['source'];delta=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(source[:,3:7]).inv()).magnitude()
report=dict(optimized_source_frames=len(q),source_frames=2000,source_sha256=hashlib.sha256(Path('retarget/amp_reference/amp_walk_1ms.pkl').read_bytes()).hexdigest(),peak_reference_joint_rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=-1)).max()),peak_root_position_offset=float(np.linalg.norm(q[:,:3]-source[:,:3],axis=-1).max()),peak_root_orientation_offset=float(delta.max()),initial_joint_rmse=float(np.sqrt(np.mean((q[0,7:]-source[0,7:])**2))),initial_root_position_offset=float(np.linalg.norm(q[0,:3]-source[0,:3])),reference_flights=flights,full_tracking_passed=False,scope='Offline candidate; all source/landing/fidelity gates require independent physical recording')
np.savez_compressed(p/'newton_control.npz',**{k:d[k] for k in d.files},support_points=support,native_clearance=clearance)
(p/'newton_export.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
