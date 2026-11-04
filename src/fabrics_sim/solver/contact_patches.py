"""Independent observed sole patches for the two feet during load transfer."""
import numpy as np
import torch

from .observed_sole_contact import ObservedSoleContact
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix


def loaded_sole_contacts(poses, contact_data):
    """Qualify real loaded sole points; a planned pose alone is insufficient."""
    forces,points,normals,distances,counts,starts=(np.asarray(v) for v in contact_data)
    forces,distances=forces.reshape(-1),distances.reshape(-1)
    rotations=_quat_xyzw_to_matrix(poses[...,3:]).detach().cpu().numpy()[0]
    positions=poses[0,:,:3].detach().cpu().numpy()
    result=np.zeros(2,dtype=bool)
    for foot in range(2):
        start,count=int(starts.reshape(-1)[foot]),int(counts.reshape(-1)[foot])
        sl=slice(start,start+count)
        local=(points[sl]-positions[foot])@rotations[foot]
        # The USD also has a toe-tip sphere centered at x=.14 with radius
        # .005. Its loaded surface can lie beyond .14 when the foot rolls.
        sole=((local[:,0]>=-.07)&(local[:,0]<=.15)&(np.abs(local[:,1])<=.045)
              &(np.abs(local[:,2]+.035)<=.015))
        result[foot]=bool((sole&(forces[sl]>.5)&(distances[sl]<=.001)
                           &(np.abs(normals[sl,2])>.9)).any())
    return result


class ContactPatchManager:
    """Keep actual material anchors until a foot unloads or becomes flat.

    The legacy planned toe release can supply one primary patch. Other active
    feet get their own heel/toe/point observations and a soft flattening task.
    No patch is inferred for an airborne or unloaded foot.
    """
    def __init__(self):self.patches={}

    def update(self, active, poses, angular_speeds, contact_data, time, dt,
               primary_foot=None, primary_contact=None):
        forces,points,normals,distances,counts,starts=(np.asarray(v) for v in contact_data)
        qualified=loaded_sole_contacts(poses,contact_data)
        rotations=_quat_xyzw_to_matrix(poses[...,3:])
        corners=poses.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],
                                  [.12,-.025,-.035],[.12,.025,-.035]])
        restored=[]
        for foot in range(2):
            if not bool(active[0,foot]):
                self.patches.pop(foot,None)
                continue
            if foot==primary_foot and primary_contact is not None:
                self.patches[foot]=primary_contact
                continue
            start,count=int(starts.reshape(-1)[foot]),int(counts.reshape(-1)[foot])
            sl=slice(start,start+count)
            rotation=rotations[:,foot];position=poses[:,foot,:3]
            z=(position[:,:,None]+rotation@corners.T)[:,2]
            patch=self.patches.get(foot)
            if patch is None and qualified[foot]:
                patch=ObservedSoleContact.from_lifted_sole(position,rotation,time,float(z.max()),
                    points[sl],forces[sl],distances[sl],flatten=True)
                if patch is not None:self.patches[foot]=patch
            if patch is not None:
                patch.update_support(position,rotation,points[sl],forces[sl],distances[sl])
                flat=bool(z.max()<=.002 and z.min()>=-.002 and forces[sl].sum()>=20.
                          and angular_speeds[foot]<=.2)
                patch.flat_seconds=patch.flat_seconds+dt if flat else 0.
                if patch.flat_seconds>=.06-1e-9:
                    self.patches.pop(foot);restored.append(foot)
        return dict(self.patches),restored
