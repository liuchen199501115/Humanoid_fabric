"""Finite contact-force rays strictly inside the configured Coulomb cone.

The same measured point/line/polygon and friction coefficient are retained.
Eight directions reduce the conservatism of four-ray or inscribed-square cones.
"""
import math
import numpy as np
import torch
from scipy.spatial import ConvexHull


def contact_ray_cone(points, friction, ray_count=8):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 1
            or not np.isfinite(points).all() or not math.isfinite(friction) or friction <= 0
            or type(ray_count) is not int or ray_count < 4 or ray_count % 4):
        raise ValueError('Finite support points, positive friction, and a multiple of four rays required')
    angles = 2*np.pi*np.arange(ray_count)/ray_count
    directions = np.c_[friction*np.cos(angles), friction*np.sin(angles), np.ones(ray_count)]
    forces = np.tile(directions, (len(points), 1))
    rays = np.c_[forces, np.cross(np.repeat(points, ray_count, axis=0), forces)]
    columns = [0, 1, 3, 4, 5]
    section = rays[:, columns]
    center = section.mean(0)
    _, singular, vt = np.linalg.svd(section-center, full_matrices=True)
    rank = int((singular > 1e-10*max(1., singular[0])).sum())
    basis = vt[:rank].T
    hull = ConvexHull((section-center)@basis)
    normals = hull.equations[:, :-1]@basis.T
    constants = hull.equations[:, -1]-normals@center
    _, unique = np.unique(np.round(np.c_[normals, constants], 10), axis=0, return_index=True)
    normals, constants = normals[np.sort(unique)], constants[np.sort(unique)]
    # Degenerate support has exact moment relations, represented in both signs.
    null = vt[rank:]
    normals = np.vstack((normals, null, -null))
    constants = np.r_[constants, -null@center, null@center]
    G = np.zeros((len(normals)+1, 6))
    G[:-1, columns] = normals
    G[:-1, 2] = constants
    G[-1, 2] = -1.
    return G


def patch_ray_constraints(patch, rotation, friction, maximum_normal_force, minimum_normal_force=0., ray_count=8):
    if not math.isfinite(maximum_normal_force) or not 0 <= minimum_normal_force < maximum_normal_force:
        raise ValueError('Invalid normal-force bounds')
    if getattr(patch, 'polygon_support', False):
        offsets = patch.vertices-patch.local_point[:, None]
        points = (patch.frame.transpose(-1, -2)@rotation@offsets.transpose(-1, -2))[0].T.detach().cpu().numpy()
    elif patch.point_support:
        points = np.zeros((1, 3))
    else:
        points = np.array([[0., -patch.half_width, 0.], [0., patch.half_width, 0.]])
    signature = (float(friction), float(maximum_normal_force), float(minimum_normal_force),
                 ray_count, np.round(points, 7).tobytes(), rotation.dtype, rotation.device)
    cached = getattr(patch, '_ray_wrench_cone_cache', None)
    if cached is None or cached[0] != signature:
        G = contact_ray_cone(points, friction, ray_count)
        b = np.zeros(len(G));b[-1] = -minimum_normal_force
        G = np.vstack((G, [0., 0., 1., 0., 0., 0.]));b = np.r_[b, maximum_normal_force]
        cached = signature, rotation.new_tensor(G)[None], rotation.new_tensor(b)[None]
        patch._ray_wrench_cone_cache = cached
    return cached[1:]
