"""23-DOF URDF FK; 23 actuated child links define the declared body ordering."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch import nn
from ..kinematics import rpy_matrix

# paper/ -> motion_generation/ -> repository
URDF = Path(__file__).resolve().parents[2]/'src/fabrics_sim/models/robots/urdf/g1/g1_23dof.urdf'
DIM = 99


def normalize_quaternion(q):
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    identity = torch.zeros_like(q); identity[..., 3] = 1
    return torch.where(norm > 1e-6, q/norm.clamp_min(1e-6), identity)


def quat_matrix(q):
    x,y,z,w = normalize_quaternion(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


class PaperFK(nn.Module):
    def __init__(self):
        super().__init__()
        tree = ET.parse(URDF).getroot()
        joints = tree.findall('joint')
        active = [j for j in joints if j.get('type') == 'revolute']
        self.names = [j.get('name') for j in active]
        self.links = [j.find('child').get('link') for j in active]
        if len(active) != 23:
            raise ValueError('Expected 23 actuated joints')
        self.register_buffer('limits', torch.tensor([[float(j.find('limit').get(k)) for k in ('lower','upper')] for j in active]))
        by_child = {j.find('child').get('link'):j for j in joints}
        needed = set(self.links)
        for link in self.links:
            while link in by_child:
                link = by_child[link].find('parent').get('link'); needed.add(link)
        relevant = [j for j in joints if j.find('child').get('link') in needed]
        roots = needed-set(by_child)
        if roots != {'pelvis'}:
            raise ValueError(f'Unexpected base links {roots}')
        visited = {'pelvis':0}; self.chain=[]
        while relevant:
            ready = [j for j in relevant if j.find('parent').get('link') in visited]
            if not ready: raise ValueError('Cyclic URDF')
            for j in ready:
                parent = visited[j.find('parent').get('link')]
                i = len(self.chain)
                origin=j.find('origin')
                def vector(tag):
                    return np.fromstring(origin.get(tag,'0 0 0') if origin is not None else '0 0 0',sep=' ')
                self.register_buffer(f'p{i}',torch.tensor(vector('xyz'),dtype=torch.float32))
                self.register_buffer(f'r{i}',torch.tensor(rpy_matrix(vector('rpy')),dtype=torch.float32))
                axis=j.find('axis')
                a=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ');a/=np.linalg.norm(a)
                x,y,z=a
                self.register_buffer(f'k{i}',torch.tensor([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=torch.float32))
                index=self.names.index(j.get('name')) if j.get('type')=='revolute' else -1
                self.chain.append((parent,index))
                visited[j.find('child').get('link')]=i+1;relevant.remove(j)
        self.output_ids=[visited[n] for n in self.links]
        self.foot_ids=[visited[f'{side}_ankle_roll_link'] for side in ('left','right')]
        self.register_buffer('sole_centres',torch.tensor([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]]))

    def forward(self, states, soles=False):
        # states: root xyz, quaternion xyzw, 23 joint angles, optional link xyz.
        positions=[states[...,:3]];rotations=[quat_matrix(states[...,3:7])]
        eye=torch.eye(3,device=states.device,dtype=states.dtype)
        for i,(parent,joint) in enumerate(self.chain):
            rp=rotations[parent]
            p=positions[parent]+(rp@getattr(self,f'p{i}')[...,None]).squeeze(-1)
            r=rp@getattr(self,f'r{i}')
            if joint>=0:
                theta=states[...,7+joint,None,None];k=getattr(self,f'k{i}')
                r=r@(eye+theta.sin()*k+(1-theta.cos())*(k@k))
            positions.append(p);rotations.append(r)
        links=torch.stack([positions[i] for i in self.output_ids],-2)
        if not soles:return links
        points=[positions[i][...,None,:]+(rotations[i][...,None,:,:]@self.sole_centres[...,None]).squeeze(-1) for i in self.foot_ids]
        return links,torch.stack(points,-3)
