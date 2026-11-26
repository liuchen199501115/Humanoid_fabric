"""Resolve actuator limits from the active model and simulator, by joint name."""
import xml.etree.ElementTree as ET
import numpy as np


def model_effort_limits(urdf_path, joint_names, simulator_limits):
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected unique joint names')
    nodes = {node.attrib['name']: node.find('limit')
             for node in ET.parse(urdf_path).getroot().findall('joint')}
    try:
        urdf = np.asarray([float(nodes[name].attrib['effort']) for name in names])
    except (KeyError, AttributeError) as exc:
        raise ValueError('Missing named joint effort limit in active URDF') from exc
    measured = np.asarray(simulator_limits, dtype=float)
    if measured.shape == (1, len(names)):
        measured = measured[0]
    if (measured.shape != (len(names),) or not np.isfinite(measured).all()
            or not np.isfinite(urdf).all() or (measured <= 0).any() or (urdf <= 0).any()):
        raise ValueError('Expected positive finite simulator and URDF effort limits')
    # Never increase either the authored model or the active simulator limit.
    return np.minimum(urdf, measured).astype(np.float32)
