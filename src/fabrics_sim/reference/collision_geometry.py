"""Conservative sphere covers of collision meshes, in each link's frame."""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def box_sphere_cover(lower, upper, segments=3):
    """Partition the longest box axis; each sphere encloses one whole slab."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all() or (upper <= lower).any()
            or type(segments) is not int or segments < 1):
        raise ValueError('Expected a finite nondegenerate box and positive segment count')
    half = .5*(upper-lower)
    axis = int(np.argmax(half))
    centers = np.tile(.5*(upper+lower), (segments, 1))
    centers[:, axis] = lower[axis]+(np.arange(segments)+.5)*(upper[axis]-lower[axis])/segments
    half[axis] /= segments
    return centers, float(np.linalg.norm(half))


def collision_mesh_spheres(urdf_path, link_name, segments=3, geometry_kind='collision'):
    """Cover binary-STL and primitive collisions, including origin/scale.

    Bounding boxes deliberately overestimate mesh volume. These covers are
    soft planning geometry; PhysX collision meshes remain authoritative.
    """
    urdf_path = Path(urdf_path)
    if type(segments) is not int or segments < 1:
        raise ValueError('Collision cover segment count must be a positive integer')
    if geometry_kind not in ('collision', 'visual'):
        raise ValueError('Geometry must be collision or visual')
    link = next((node for node in ET.parse(urdf_path).getroot().findall('link')
                 if node.attrib['name'] == link_name), None)
    if link is None:
        raise ValueError(f'Unknown collision link: {link_name}')
    result = []
    triangle = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    for collision in link.findall(geometry_kind):
        mesh = collision.find('geometry/mesh')
        origin = collision.find('origin')
        xyz = np.zeros(3) if origin is None else np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.zeros(3) if origin is None else np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        rotation = Rotation.from_euler('xyz', rpy)
        if mesh is None:
            sphere=collision.find('geometry/sphere')
            cylinder=collision.find('geometry/cylinder')
            box=collision.find('geometry/box')
            if sphere is not None:
                centers=np.zeros((1,3));radius=float(sphere.attrib['radius'])
            elif cylinder is not None:
                radial=float(cylinder.attrib['radius']);length=float(cylinder.attrib['length'])
                if min(radial,length)<=0 or not np.isfinite([radial,length]).all():
                    raise ValueError('Cylinder dimensions must be finite and positive')
                centers=np.zeros((segments,3))
                centers[:,2]=-.5*length+(np.arange(segments)+.5)*length/segments
                radius=float(np.hypot(radial,.5*length/segments))
            elif box is not None:
                half=.5*np.fromstring(box.attrib['size'],sep=' ')
                centers,radius=box_sphere_cover(-half,half,segments)
            else:
                raise ValueError(f'Unsupported collision geometry on {link_name}')
            if not np.isfinite(radius) or radius<=0:
                raise ValueError('Collision radius must be finite and positive')
            result.extend((center.tolist(),radius) for center in rotation.apply(centers)+xyz)
            continue
        data = (urdf_path.parent/mesh.attrib['filename']).read_bytes()
        if len(data) < 84 or len(data) != 84+50*struct.unpack_from('<I', data, 80)[0]:
            raise ValueError(f'Expected binary STL collision on {link_name}')
        vertices = np.frombuffer(data, offset=84, dtype=triangle)['vertices'].reshape(-1, 3)
        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        vertices = rotation.apply(vertices*scale)+xyz
        centers, radius = box_sphere_cover(vertices.min(0), vertices.max(0), segments)
        result.extend((center.tolist(), radius) for center in centers)
    if not result:
        raise ValueError(f'No collision geometry on {link_name}')
    return result
