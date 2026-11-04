"""Audit every PKL frame and contact phase; this is not a tracking certificate."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from fabrics_sim.reference.source_sequence import audit_source_schedule
from test_g1_two_tasks import G1_JOINT_NAMES, load_motion_trajectory, load_motion_contact_sequence


def audit_clip(source, fps, labels, lower, upper):
    q=np.asarray(source);labels=np.asarray(labels);lower=np.asarray(lower);upper=np.asarray(upper)
    if (q.ndim!=2 or q.shape[1]!=36 or not len(q) or labels.shape!=(len(q),2)
            or not np.isfinite(q).all() or not np.isin(labels,[0,1]).all()
            or not np.isfinite(fps) or fps<=0 or lower.shape!=(29,) or upper.shape!=(29,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower>=upper).any()):
        raise ValueError('Expected finite G1 poses, matching binary contacts, positive FPS and ordered joint limits')
    report=audit_source_schedule(labels[:,0].astype(int)+2*labels[:,1].astype(int))
    errors=np.maximum(lower-q[:,7:],q[:,7:]-upper).clip(0)
    norm=np.linalg.norm(q[:,3:7],axis=1)
    report.update(duration_seconds=float((len(q)-1)/fps),fps=float(fps),
        joint_limit_violation_frames=int((errors.max(1)>1e-5).sum()),
        maximum_joint_limit_violation_rad=float(errors.max()),
        invalid_quaternion_frames=int((np.abs(norm-1)>1e-4).sum()),
        scope='Full-file source data and legacy schedule compatibility; no dynamics or simulation certification')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-pkl',type=Path,default=Path('retarget/lafan/walk1_g1.pkl'))
    p.add_argument('--contacts-pkl',type=Path,default=Path('retarget/lafan/walk1_g1_with_contacts.pkl'))
    p.add_argument('--urdf',type=Path,default=Path('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    q,fps=load_motion_trajectory(args.motion_pkl)
    contact_q,contact_fps=load_motion_trajectory(args.contacts_pkl)
    labels=load_motion_contact_sequence(args.contacts_pkl)
    if labels is None or contact_fps!=fps or not np.array_equal(q.numpy(),contact_q.numpy()):
        raise ValueError('Contact-labelled copy must preserve the original motion exactly')
    limits={n.attrib['name']:n.find('limit') for n in ET.parse(args.urdf).getroot().findall('joint')}
    report=audit_clip(q.numpy(),fps,labels.numpy(),
        [float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES],
        [float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES])
    report.update(motion_pkl=str(args.motion_pkl),contacts_pkl=str(args.contacts_pkl),urdf=str(args.urdf))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('segments','repeated_support','direct_switches')},indent=2))


if __name__=='__main__':main()
