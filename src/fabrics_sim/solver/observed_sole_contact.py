"""Loaded sole geometry: a point, an arbitrary edge, or a support polygon."""
import numpy as np
import torch
from scipy.spatial import ConvexHull

from .toe_contact import ToeContact
from fabrics_sim.fabrics.dynamics_aware import rotation_error


def polygon_wrench_cone(points, friction):
    """Half spaces for forces at coplanar points, about the supplied origin.

    Each point has a four-ray, inscribed Coulomb pyramid. Taking the convex
    hull of normalized wrenches retains the coupling between CoP, friction,
    and yaw moment; a triangle is never replaced by its bounding rectangle.
    Columns are [fx, fy, fz, mx, my, mz], with G @ wrench <= 0.
    """
    points=np.asarray(points,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<3
            or not np.isfinite(points).all() or not np.isfinite(friction) or friction<=0):
        raise ValueError('Expected a finite polygon and positive friction')
    if np.linalg.matrix_rank(points[:,:2]-points[:,:2].mean(0),tol=1e-6)<2:
        raise ValueError('Wrench polygon must span an area')
    directions=np.array([[friction,0.,1.],[-friction,0.,1.],
                         [0.,friction,1.],[0.,-friction,1.]])
    forces=np.tile(directions,(len(points),1))
    offsets=np.repeat(points,4,axis=0)
    rays=np.c_[forces,np.cross(offsets,forces)]
    section=rays[:,[0,1,3,4,5]]  # fz == 1 on every ray
    facets=ConvexHull(section).equations
    # Qhull triangulates coplanar faces; keep each half space only once.
    _,indices=np.unique(np.round(facets,10),axis=0,return_index=True)
    facets=facets[np.sort(indices)]
    result=np.zeros((len(facets)+1,6))
    result[:-1,[0,1,3,4,5]]=facets[:,:5]
    result[:-1,2]=facets[:,5]
    result[-1,2]=-1.  # unilateral normal force
    return result


class ObservedSoleContact(ToeContact):
    """Represent loaded sole geometry during both landing and toe-off.

    Flattening versus rolling selects the soft orientation objective. Neither
    permits retaining an unloaded corner in place of a newly loaded toe tip,
    or discarding a loaded oblique edge or third support point.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sole_frame=self.frame.clone()
        self.polygon_support=False
        self.vertices=None
        self._geometry=None
        self._cone_cache=None

    def update_support(self,position,rotation,world_points,normal_forces,separations,
                       *,touching_tolerance=None):
        points=np.asarray(world_points,dtype=float)
        forces=np.asarray(normal_forces).reshape(-1)
        distances=np.asarray(separations).reshape(-1)
        if (len(position)!=1 or points.shape!=(len(forces),3) or distances.shape!=forces.shape
                or not all(np.isfinite(v).all() for v in (points,forces,distances))):
            raise ValueError('Expected one robot and finite observed sole contacts')
        loaded=(forces>.5)&(distances<=.001)
        if not loaded.any():return False
        if touching_tolerance is not None:
            if not np.isfinite(touching_tolerance) or not 0 <= touching_tolerance <= .0001:
                raise ValueError('Touching support tolerance must be within 0..100 micrometres')
            # An already bearing foot may redistribute load onto other real
            # near-touching manifold points. Never infer points from its mesh
            # or use this option to create an unloaded/airborne support foot.
            if forces[loaded].sum() >= 20.:
                loaded |= (forces >= 0.) & (distances <= touching_tolerance)
        points=points[loaded]
        origin=position[0].detach().cpu().numpy();R=rotation[0].detach().cpu().numpy()
        local=(points-origin)@R
        # Duplicate points may be reported by different foot collision shapes.
        _,indices=np.unique(np.round(local,4),axis=0,return_index=True)
        points=points[np.sort(indices)];local=local[np.sort(indices)]
        singular=np.linalg.svd(points[:,:2]-points[:,:2].mean(0),compute_uv=False)
        polygon=len(singular)>1 and singular[1]>.002
        if polygon:
            indices=ConvexHull(points[:,:2]).vertices
        elif len(points)>1 and singular[0]>.005:
            distance=np.sum((points[:,None,:2]-points[None,:,:2])**2,axis=-1)
            indices=np.array(np.unravel_index(distance.argmax(),distance.shape))
        else:
            indices=np.array([0])
        vertices=local[indices]
        vertices=vertices[np.lexsort((vertices[:,2],vertices[:,1],vertices[:,0]))]
        kind='polygon' if polygon else 'point' if len(vertices)==1 else 'line'
        if (self._geometry is not None and self._geometry[0]==kind
                and self._geometry[1].shape==vertices.shape
                and np.max(np.linalg.norm(self._geometry[1]-vertices,axis=1))<.001):
            return False
        self._geometry=(kind,vertices.copy());self._cone_cache=None
        self.vertices=position.new_tensor(vertices)[None]
        self.local_point=self.vertices.mean(1)
        self.anchor=position+(rotation@self.local_point[...,None]).squeeze(-1)
        self.initial_rotation=rotation.clone()
        self.frame=self.sole_frame.clone()
        self.point_support=kind=='point';self.polygon_support=kind=='polygon'
        if kind=='line':
            world=vertices@R.T+origin
            direction=world[1]-world[0];direction[2]=0.
            length=np.linalg.norm(direction);direction/=length
            # The free rotation is always local y, aligned with this edge.
            self.frame[0,:,1]=position.new_tensor(direction)
            self.frame[0,:,0]=position.new_tensor([direction[1],-direction[0],0.])
            self.half_width=length/2
        return True

    def evaluate(self,position,rotation,jacobian,bias,velocity,time,orientation_reference=None):
        mapped_jac,mapped_bias,error,task=super().evaluate(
            position,rotation,jacobian,bias,velocity,time,orientation_reference)
        if self.flatten:
            inverse=self.frame.transpose(-1,-2)
            flat_error=(inverse@rotation_error(rotation,self.sole_frame)[...,None]).squeeze(-1)
            rate=(mapped_jac[:,3:]@velocity[...,None]).squeeze(-1)
            task.desired=(-50*flat_error-14*rate).clamp(-25,25)
        # A measured polygon fixes all rotations regardless of whether the
        # reference would like to roll. A soft pitch task cannot free it.
        if self.polygon_support:task.active[:]=False
        return mapped_jac,mapped_bias,error,task

    def wrench_constraints(self,rotation,friction,maximum_normal_force):
        if not self.polygon_support:raise ValueError('Expected polygon support')
        offsets=self.vertices-self.local_point[:,None]
        points=(self.frame.transpose(-1,-2)@rotation@offsets.transpose(-1,-2))[0].T.detach().cpu().numpy()
        signature=(float(friction),float(maximum_normal_force),np.round(points,7).tobytes())
        if self._cone_cache is None or self._cone_cache[0]!=signature:
            G=polygon_wrench_cone(points,friction)
            G=np.vstack((G,[0.,0.,1.,0.,0.,0.]))
            b=np.zeros(len(G));b[-1]=maximum_normal_force
            self._cone_cache=(signature,rotation.new_tensor(G)[None],rotation.new_tensor(b)[None])
        return self._cone_cache[1:]
