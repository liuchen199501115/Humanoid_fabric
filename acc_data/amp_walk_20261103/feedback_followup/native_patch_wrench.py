"""Exact polyhedral wrench cone of native contact points and friction rays."""
import numpy as np
from scipy.spatial import ConvexHull

def native_patch_wrench(points, friction=.5, maximum_normal=800., force_frame=None):
    points=np.unique(np.round(np.asarray(points,dtype=float),10),axis=0)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError('Expected at least one 3D contact offset')
    directions=np.array([[friction,0.,1.],[-friction,0.,1.],[0.,friction,1.],[0.,-friction,1.]])
    if force_frame is not None:directions=directions@np.asarray(force_frame).T
    forces=np.tile(directions,(len(points),1));offsets=np.repeat(points,4,axis=0)
    rays=np.c_[forces,np.cross(offsets,forces)]
    # All ray normal forces equal one. Handle point/line patches by reducing
    # to their exact affine hull, rather than inventing a finite footprint.
    columns=[0,1,3,4,5];section=rays[:,columns];center=section.mean(0)
    _,singular,basis=np.linalg.svd(section-center,full_matrices=True)
    rank=int(np.sum(singular>1e-9));coordinates=(section-center)@basis[:rank].T
    facets=ConvexHull(coordinates).equations
    normals=facets[:,:rank]@basis[:rank]
    offsets=facets[:,rank]-normals@center
    combined=np.c_[normals,offsets];_,indices=np.unique(np.round(combined,9),axis=0,return_index=True)
    combined=combined[np.sort(indices)]
    G=np.zeros((len(combined)+2,6));G[:-2,columns]=combined[:,:5];G[:-2,2]=combined[:,5]
    G[-2,2]=-1.;G[-1,2]=1.;b=np.zeros(len(G));b[-1]=maximum_normal
    E=np.zeros((5-rank,6));E[:,columns]=basis[rank:];E[:,2]=-basis[rank:]@center
    return E,G,b
