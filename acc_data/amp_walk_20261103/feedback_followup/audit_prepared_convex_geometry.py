"""Sample actual URDF convex clearances of a prepared reference, without stepping."""
from pathlib import Path
import json,sys,numpy as np,torch,yaml,mujoco
from fabrics_sim.reference.convex_collision import ConvexLinkCollision
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from mujoco_centroidal_model import CentroidalModel
p=Path(__file__).parent;name=sys.argv[1];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
q=np.load(p/(name+'.npz'))['control_q'];d=np.load(p/'wrench_fixed_ten.npz');model=CentroidalModel(urdf,d['final_joint_order'])
from reference_convex_model import ReferenceConvexModel
checker=ReferenceConvexModel(model,urdf);geometry=checker.geometry;pairs=checker.pairs;t=torch.tensor(q,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
metadata=json.loads((p/(name+'.json')).read_text());parent=np.load(p/(metadata['parent']+'.npz'))['control_q']
changed=np.flatnonzero(np.any(q!=parent,axis=1));lo=max(0,int(changed.min())-12) if len(changed) else 0;hi=min(len(q)-1,int(changed.max())+12) if len(changed) else len(q)-1
frames=np.unique(np.r_[np.arange(0,len(q),6),np.arange(lo,hi+.01,.5),len(q)-1]);minima=np.full(len(pairs),np.inf);worst=np.zeros(len(pairs));violations=[]
for frame in frames:
    pose=curve.kinematics(float(frame),0.)[0].numpy()[0];model._state(pose)
    geometry.data.mocap_pos[geometry.mocap_ids]=model.data.xpos[geometry.indices]
    geometry.data.mocap_quat[geometry.mocap_ids]=model.data.xquat[geometry.indices]
    mujoco.mj_kinematics(geometry.model,geometry.data)
    for j,(a,b) in enumerate(geometry.pair_indices):
        gap=min(mujoco.mj_geomDistance(geometry.model,geometry.data,ga,gb,geometry.cutoff,None) for ga in geometry.geometry_ids[a] for gb in geometry.geometry_ids[b])
        if gap<minima[j]:minima[j]=gap;worst[j]=frame
        if gap<0:violations.append(dict(frame=float(frame),pair=pairs[j],gap=float(gap)))
changed_violations=[row for row in violations if lo<=row['frame']<=hi]
report=dict(passed=not changed_violations,whole_reference_collision_free=not violations,changed_region_frames=[lo,hi],unchanged_region_violations=len(violations)-len(changed_violations),scope='Configured native convex wrist/hip/torso/shoulder pairs, original pair links, and foot-foot; sampled geometry, not a continuous full-body certificate',samples=len(frames),turn_stride_frames=.5,rest_stride_frames=6,minimum_gap_m=float(minima.min()),acceptance='No native penetration within the changed reference region plus 12-frame interpolation support. Unchanged-region findings remain reported; physical tracking still requires measured self-contact audit.',pairs=[dict(pair=pair,minimum_gap_m=float(gap),frame=float(frame)) for pair,gap,frame in zip(pairs,minima,worst)],violations=violations)
(p/(name+'.convex_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['pairs','violations']})
metadata=json.loads((p/(name+'.json')).read_text());metadata['native_convex_geometry_audit']=str(p/(name+'.convex_audit.json'));metadata['native_convex_geometry_passed']=report['passed'];metadata['native_convex_geometry_scope']=report['acceptance'];metadata['requires_native_body_collision_audit']=False;metadata['preflight_passed']=bool(metadata.get('preflight_passed',False) and report['passed']);(p/(name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
if not report['passed']:raise RuntimeError('Prepared reference has native convex penetration; do not execute')
