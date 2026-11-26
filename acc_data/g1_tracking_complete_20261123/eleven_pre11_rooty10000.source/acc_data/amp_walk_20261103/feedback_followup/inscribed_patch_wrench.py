"""Inscribed Coulomb rays at measured contact vertices; instantaneous WBC only."""
import math
import numpy as np
import torch
from scipy.spatial import ConvexHull


def inscribed_wrench_cone(points, friction, rays=8):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or not math.isfinite(friction) or friction <= 0
            or type(rays) is not int or rays < 8 or rays % 8):
        raise ValueError('Expected finite vertices, positive friction, and a multiple of eight rays')
    angles = np.arange(rays)*2*np.pi/rays
    directions = np.c_[friction*np.cos(angles), friction*np.sin(angles), np.ones(rays)]
    force = np.tile(directions, (len(points),1))
    wrench = np.c_[force,np.cross(np.repeat(points,rays,axis=0),force)]
    section = wrench[:,[0,1,3,4,5]]  # Every generating ray has fz=1.
    center = section.mean(0)
    _, singular, basis = np.linalg.svd(section-center, full_matrices=True)
    rank = int((singular > max(1.,singular[0])*1e-10).sum())
    if rank < 2:
        raise ValueError('Degenerate friction section')
    reduced = (section-center)@basis[:rank].T
    facets = ConvexHull(reduced).equations
    normals = facets[:,:rank]@basis[:rank]
    offsets = facets[:,rank]-normals@center
    # Point/line contacts have exact absent moments. Preserve their affine hull
    # using paired inequalities rather than inventing a full support polygon.
    if rank < 5:
        null = basis[rank:]
        normals = np.vstack((normals,null,-null))
        offsets = np.r_[offsets,-null@center,null@center]
    facets = np.c_[normals,offsets]
    _, unique = np.unique(np.round(facets,10),axis=0,return_index=True)
    facets = facets[np.sort(unique)]
    G = np.zeros((len(facets)+1,6))
    G[:-1,[0,1,3,4,5]] = facets[:,:5]
    G[:-1,2] = facets[:,5]
    G[-1,2] = -1.
    return G


def observed_patch_constraints(patch, rotation, friction, maximum_normal_force, rays, minimum_normal_force=0., yaw_limit=None):
    if not hasattr(patch,'vertices') or patch.vertices is None:
        raise ValueError('Refined wrench cone requires observed material vertices')
    offsets = patch.vertices-patch.local_point[:,None]
    points = (patch.frame.transpose(-1,-2)@rotation@offsets.transpose(-1,-2))[0].T.detach().cpu().numpy()
    # Match the existing contact Jacobian: an ideal point or horizontal line.
    if patch.point_support:
        points=np.zeros((1,3))
    elif not patch.polygon_support:
        points=np.array([[0.,-patch.half_width,0.],[0.,patch.half_width,0.]])
    signature = (float(friction),float(maximum_normal_force),rays,float(minimum_normal_force),yaw_limit,np.round(points,7).tobytes())
    cached = getattr(patch,'_inscribed_wrench_cache',None)
    if cached is None or cached[0] != signature:
        G = inscribed_wrench_cone(points,friction,rays)
        G = np.vstack((G,[0.,0.,1.,0.,0.,0.]))
        b = np.zeros(len(G));b[-2] = -minimum_normal_force;b[-1] = maximum_normal_force
        if yaw_limit is not None:
            G=np.vstack((G,[0.,0.,-yaw_limit,0.,0.,1.],[0.,0.,-yaw_limit,0.,0.,-1.]))
            b=np.r_[b,0.,0.]
        cached = (signature,torch.as_tensor(G,device=rotation.device,dtype=rotation.dtype)[None],
                  torch.as_tensor(b,device=rotation.device,dtype=rotation.dtype)[None])
        patch._inscribed_wrench_cache = cached
    return cached[1:]
