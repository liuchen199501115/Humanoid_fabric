"""Convert the existing 50 FPS G1 archive to the paper's declared 99D layout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .kinematics import PaperFK
from .data import encode_legacy,canonicalize_quaternions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-clips',type=int,default=0,help='0 converts all clips; otherwise explicit smoke subset')
    a=p.parse_args()
    if a.max_clips<0:p.error('max-clips must be nonnegative')
    torch.set_num_threads(4)
    with np.load(a.input,allow_pickle=False) as f:archive=dict(f)
    if float(archive['fps'])!=50. or archive['states'].shape[1]!=38:
        p.error('Input must be the existing 38D archive resampled to 50 FPS')
    count=min(a.max_clips or len(archive['split']),len(archive['split']))
    if count<2:p.error('Need at least two clips')
    offsets=archive['offsets'][:count+1]
    end=int(offsets[-1]);old=archive['states'][:end]
    fk=PaperFK()
    states=np.empty((end,99),dtype=np.float32)
    for begin in range(0,end,4096):
        states[begin:begin+4096]=encode_legacy(old[begin:begin+4096],archive['dof_names'].tolist(),fk)
        if begin%204800==0:print(f'Converted {begin}/{end} frames',flush=True)
    for start,stop in zip(offsets[:-1],offsets[1:]):
        states[start:stop,3:7]=canonicalize_quaternions(states[start:stop,3:7])
    split=archive['split'][:count].copy()
    if len(set(split.tolist()))<2:
        if not a.max_clips:p.error('Input lacks train/validation split')
        split[-1]=1;split[0]=0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,states=states,offsets=offsets,split=split,paths=archive['paths'][:count],
                        fps=50.,dof_names=np.asarray(fk.names),body_names=np.asarray(fk.links),
                        joint_limits=fk.limits.numpy(),representation='g1-paper99',format_version=3)
    report=dict(clips=count,frames=end,hours=end/50/3600,motion_dim=99,fps=50,
                layout='root_xyz[0:3], root_quat_xyzw[3:7], joints[7:30], body_xyz[30:99]',
                joint_names=fk.names,body_names=fk.links,
                omitted_29dof_joints=[n for n in archive['dof_names'].tolist() if n not in fk.names],
                terrain_condition=False,ground_assumption='z=0, used only by soft penetration loss',
                source=str(a.input))
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
