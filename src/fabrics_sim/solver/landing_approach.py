"""Normal-speed limits for an airborne foot approaching a horizontal floor."""
import itertools
import math

import numpy as np
import torch


def collision_support_points(stage, link_path):
    """Read actual sphere/cube colliders in the foot's local frame.

    Sphere centers have a separate radius: their lowest world point is below
    the center along world Z, not along the rotating foot's local Z.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    link=stage.GetPrimAtPath(link_path)
    if not link.IsValid():raise ValueError(f'Missing collision link: {link_path}')
    cache=UsdGeom.XformCache();inverse=cache.GetLocalToWorldTransform(link).GetInverse()
    points=[];radii=[]
    for prim in Usd.PrimRange(link,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        transform=cache.GetLocalToWorldTransform(prim)*inverse
        if prim.IsA(UsdGeom.Sphere):
            scales=np.linalg.svd(np.asarray(transform)[:3,:3],compute_uv=False)
            if not np.allclose(scales,scales[0],rtol=1e-6):
                raise ValueError('Landing speed limits require uniformly scaled spheres')
            points.append(tuple(transform.Transform(Gf.Vec3d(0.))))
            radii.append(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())*scales[0])
        elif prim.IsA(UsdGeom.Cube):
            half=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            for corner in itertools.product((-half,half),repeat=3):
                points.append(tuple(transform.Transform(Gf.Vec3d(*corner))));radii.append(0.)
        else:
            raise ValueError(f'Unsupported landing collider: {prim.GetPath()} ({prim.GetTypeName()})')
    if not points:raise ValueError(f'No enabled foot collision geometry: {link_path}')
    return np.asarray(points),np.asarray(radii)


def landing_approach_constraints(position,rotation,jacobian,bias,velocity,points,radii,
                                 touchdown_speed=.05,braking_acceleration=2.,bandwidth=20.,
                                 activation_height=.08):
    """Return C,d for C*qdd<=d, surface height/speed and active rows.

    The speed envelope is v_z >= -sqrt(v_touch^2 + 2*a_brake*z).
    Its first-order barrier requests finite braking as the floor approaches.
    This is a local model condition; it cannot certify unmodelled impulses or
    recover all states already outside the envelope. Physical QP limits apply.
    """
    if (not all(math.isfinite(v) and v>0 for v in
                (touchdown_speed,braking_acceleration,bandwidth,activation_height))
            or points.ndim!=2 or points.shape[1]!=3 or radii.shape!=(len(points),)
            or not torch.isfinite(points).all() or not torch.isfinite(radii).all()
            or (radii<0).any()):
        raise ValueError('Invalid landing geometry or speed-envelope parameters')
    offset=(rotation[:,None]@points[None,:,:,None]).squeeze(-1)
    angular=jacobian[:,3:].transpose(-1,-2)[:,None]
    point_jac=jacobian[:,None,:3]+torch.linalg.cross(
        angular,offset[:,:,None],dim=-1).transpose(-1,-2)
    omega=(jacobian[:,3:]@velocity[...,None]).squeeze(-1)[:,None]
    point_bias=(bias[:,None,:3]+torch.linalg.cross(bias[:,None,3:],offset,dim=-1)
                +torch.linalg.cross(omega,torch.linalg.cross(omega,offset,dim=-1),dim=-1))
    height=position[:,None,2]+offset[...,2]-radii
    Jz=point_jac[:,:,2];speed=(Jz@velocity[...,None]).squeeze(-1)
    envelope=torch.sqrt(touchdown_speed**2+2*braking_acceleration*height.clamp_min(0))
    # Below the floor the clamped envelope has zero derivative. Contact
    # detection owns the transition; no fictitious support force is added.
    slope=torch.where(height>0,braking_acceleration/envelope,torch.zeros_like(height))
    minimum=-slope*speed-bandwidth*(speed+envelope)
    active=height<=activation_height
    C=torch.where(active[...,None],-Jz,torch.zeros_like(Jz))
    d=torch.where(active,point_bias[...,2]-minimum,torch.zeros_like(height))
    return C,d,height,speed,active
