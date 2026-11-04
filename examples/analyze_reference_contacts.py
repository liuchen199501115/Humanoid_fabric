"""Summarize observed foot contact patches and takeoff reference discontinuities."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize_contacts(data):
    time = np.asarray(data['history_time']).reshape(-1)
    feet = np.asarray(data['history_feet']).reshape(-1, 2, 7)
    mode = np.asarray(data['history_commanded_mode']).reshape(-1)
    count = np.asarray(data['history_contact_count']).reshape(-1, 2)
    start = np.asarray(data['history_contact_start']).reshape(-1, 2)
    point = np.asarray(data['history_contact_point'])
    force = np.asarray(data['history_contact_normal_force'])
    distance = np.asarray(data['history_contact_distance'])
    labels = []
    for i in range(len(time)):
        row = []
        for foot in range(2):
            s, n = int(start[i, foot]), int(count[i, foot])
            if n < 0 or s < 0 or s+n > point.shape[1]:
                raise ValueError('Invalid or truncated PhysX contact buffer')
            loaded = ((force[i, s:s+n].reshape(-1) > .5)
                      & (distance[i, s:s+n].reshape(-1) <= .001))
            local = Rotation.from_quat(feet[i, foot, 3:]).inv().apply(
                point[i, s:s+n][loaded]-feet[i, foot, :3])
            if len(local) == 0:
                label = 'unloaded'
            elif np.ptp(local[:, 0]) >= .12 and np.ptp(local[:, 1]) >= .035:
                label = 'area'
            elif local[:, 0].max() < 0.:
                label = 'heel'
            elif local[:, 0].min() > .07:
                label = 'toe'
            else:
                label = 'edge_or_point'
            row.append(label)
        labels.append(row)
    labels = np.asarray(labels)
    com = np.asarray(data['history_com_target']).reshape(-1, 3)
    transitions = []
    for i in np.flatnonzero((mode[1:] != mode[:-1]) & np.isin(mode[1:], [1, 2]))+1:
        transitions.append(dict(time=float(time[i]), support=int(mode[i]),
                                com_target_jump=float(np.linalg.norm(com[i]-com[i-1]))))
    nonflat = []
    for foot in range(2):
        active = (mode & (1 << foot)) != 0
        partial = active & (time >= 1.) & np.isin(labels[:, foot], ['heel', 'toe', 'edge_or_point'])
        for i in np.flatnonzero(partial & ~np.r_[False, partial[:-1]]):
            nonflat.append(dict(time=float(time[i]), foot=('left', 'right')[foot], patch=str(labels[i, foot])))
    result = dict(last_time=float(time[-1]), takeoffs=transitions,
                commanded_support_partial_contact_onsets=nonflat,
                patch_sample_counts={name: {label: int(np.sum(labels[:, foot] == label))
                    for label in np.unique(labels[:, foot])} for foot, name in enumerate(('left', 'right'))},
                scope='Observed loaded contact points (>0.5 N, separation <=1 mm), not planned contacts. '
                      'Area labels describe sampled point extent, not a guaranteed rigid flat-foot constraint. '
                      'Contact force buffers are diagnostic; net forces remain the controller observation.')
    if 'history_toe_foot' in data:
        toe=np.asarray(data['history_toe_foot']).reshape(-1)
        active=np.flatnonzero(toe>=0)
        anchors=np.asarray(data['history_toe_anchor']).reshape(-1,3)
        maximum_error=0.
        if len(active):
            poses=feet[active,toe[active]]
            points=poses[:,:3]+Rotation.from_quat(poses[:,3:]).apply([.12,0.,-.035])
            maximum_error=float(np.max(np.linalg.norm(points-anchors[active],axis=1)))
        forces=np.asarray(data['history_foot_forces']).reshape(-1,2,3)
        released=np.flatnonzero((toe[:-1]>=0)&(toe[1:]<0))+1
        result['toe_off']=dict(active_samples=len(active),maximum_toe_anchor_error=maximum_error,
            activations=[dict(time=float(time[i]),foot=int(toe[i]))
                         for i in np.flatnonzero((toe>=0)&np.r_[True,toe[1:]!=toe[:-1]])],
            releases=[dict(time=float(time[i]),foot=int(toe[i-1]),
                           old_foot_normal_force=float(forces[i,toe[i-1],2]),mode=int(mode[i])) for i in released])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.recording) as data:
        result = summarize_contacts(data)
    content = json.dumps(result, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content, end='')


if __name__ == '__main__':
    main()
