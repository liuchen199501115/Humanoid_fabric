"""Adapt this trusted AMP recording to the existing G1 tracking loader."""
import hashlib
import json
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from inspect_mocap_reference import urdf_fk
from test_g1_two_tasks import G1_JOINT_NAMES, load_motion_trajectory

root = Path(__file__).resolve().parents[2]
source = root / 'retarget/amp_reference/amp_walk_1ms.pkl'
output = Path(__file__).resolve().parent
urdf = root / 'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
with source.open('rb') as stream:
    raw = pickle.load(stream)
tree = ET.parse(urdf).getroot()
joints = tree.findall('joint')
queue = list({x.attrib['name'] for x in tree.findall('link')} -
             {j.find('child').attrib['link'] for j in joints})
assert len(queue) == 1
names = []
for parent in queue:
    for joint in joints:
        if joint.find('parent').attrib['link'] == parent:
            queue.append(joint.find('child').attrib['link'])
            if joint.attrib['type'] != 'fixed':
                names.append(joint.attrib['name'])
assert len(names) == 29 and set(names) == set(G1_JOINT_NAMES)
adapted = dict(raw)
adapted['root_rot'] = np.asarray(raw['root_rot'])[:, [1, 2, 3, 0]].copy()
adapted['dof_names'] = names
q = np.concatenate((raw['root_pos'], adapted['root_rot'], raw['dof_pos']), axis=1)
poses, _, _ = urdf_fk(q, names, urdf)
key_links = ['left_ankle_roll_link', 'right_ankle_roll_link',
             'left_wrist_yaw_link', 'right_wrist_yaw_link',
             'left_shoulder_roll_link', 'right_shoulder_roll_link']
errors = np.linalg.norm(np.stack([poses[n][0] for n in key_links], axis=1)
                        - np.asarray(raw['key_body_pos']), axis=-1)
assert float(errors.max()) < 1e-4, errors.max()
destination = output / 'amp_walk_tracking.pkl'
with destination.open('wb') as stream:
    pickle.dump(adapted, stream, protocol=4)
loaded, fps = load_motion_trajectory(destination)
limits = {j.attrib['name']: j.find('limit') for j in joints}
lo = np.array([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES])
hi = np.array([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES])
violations = np.maximum(np.maximum(lo-loaded.numpy()[:, 7:], loaded.numpy()[:, 7:]-hi), 0)
assert float(violations.max()) <= 1e-4, 'Larger joint-limit violations need separate reference adaptation'
source_lo = np.array([float(limits[n].attrib['lower']) for n in names])
source_hi = np.array([float(limits[n].attrib['upper']) for n in names])
adapted['dof_pos'] = np.clip(raw['dof_pos'], source_lo, source_hi).astype(np.float32)
with destination.open('wb') as stream:
    pickle.dump(adapted, stream, protocol=4)
report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    adapted=str(destination), frames=len(q), fps=fps, sample_span_s=(len(q)-1)/fps,
    quaternion_conversion='wxyz -> xyzw', source_joint_order=names,
    key_body_links=key_links, key_body_fk_max_error_m=float(errors.max()),
    key_body_fk_mean_error_m=float(errors.mean()),
    max_joint_limit_violation_rad=float(violations.max()),
    joint_limit_adjustment=dict(policy='Clip source limit violations only, maximum allowed 1e-4 rad',
        changed_values=int(np.count_nonzero(adapted['dof_pos'] != raw['dof_pos'])),
        max_change_rad=float(np.max(np.abs(adapted['dof_pos']-raw['dof_pos'])))),
    contact_labels='absent; existing online sole-corner inference will be used',
    note='Original sampling rate, root positions and all frames preserved; tiny joint-limit corrections recorded; source unchanged.')
(output / 'input_report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
