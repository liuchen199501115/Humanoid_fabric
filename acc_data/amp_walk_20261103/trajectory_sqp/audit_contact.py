from pathlib import Path
import sys,json
p=Path(__file__).parent
namespace={'__file__':str(p/'prepare_spline_ipm.py'),'__name__':'contact_audit'}
s=(p/'prepare_spline_ipm.py').read_text();s=s[:s.index('for iteration in range(args.iterations):')]
exec(compile(s,str(p/'prepare_spline_ipm.py'),'exec'),namespace)
np=namespace['np'];d=np.load(p/'damped_spline_candidate.npz');q=d['q'];v=d['v'];st=namespace['snapshot'](q);ids=namespace['ids'];contact=namespace['contact'];points=namespace['points'];radii=namespace['radii'];skew=namespace['skew']
clear=[];vel=[]
for k in range(len(q)):
 for side in range(2):
  if not contact[k,side]:continue
  link=ids[side];R=st['rotations'][k,link];offsets=(R@points[side].T).T;bottom=st['transforms'][k,link,2]+offsets[:,2]-radii[side];pick=int(bottom.argmin());offset=offsets[pick]-np.array([0,0,radii[side,pick]])
  J=st['origin_jac'][k,link]-skew(offset)@st['angular_jac'][k,link]
  clear.append((k,side,float(bottom[pick])));vel.append((k,side,*(J@v[k]).tolist()))
clear=np.array(clear);vel=np.array(vel);report=dict(peak_stance_gap_m=float(clear[:,2].max()),peak_stance_penetration_m=float(-clear[:,2].min()),peak_stance_material_point_speed_m_s=float(np.linalg.norm(vel[:,2:],axis=-1).max()),peak_stance_vertical_speed_m_s=float(abs(vel[:,-1]).max()),rms_stance_point_speed=float(np.sqrt(np.mean(vel[:,2:]**2))),full_tracking_passed=False)
print(report,flush=True);(p/'contact_preflight.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(p/'contact_preflight.npz',clearance=clear,velocity=vel)
