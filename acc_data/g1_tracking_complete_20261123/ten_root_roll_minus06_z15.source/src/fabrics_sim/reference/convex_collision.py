"""CPU convex-mesh distances and witness-point Jacobians; no physics stepping."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _skew


class ConvexLinkCollision:
    def __init__(self, urdf_path, kinematics, link_pairs, cutoff=1.):
        import mujoco
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('Convex collision cutoff must be positive')
        # Query beyond the .4 m barrier activation: clipping at activation
        # would create a false Jdot*v spike when finite differences cross it.
        self.mujoco, self.cutoff = mujoco, float(cutoff)
        self.pairs = tuple(tuple(pair) for pair in link_pairs)
        if not self.pairs or any(len(pair)!=2 or pair[0]==pair[1] for pair in self.pairs):
            raise ValueError('Expected nonempty distinct link pairs')
        self.links = tuple(dict.fromkeys(link for pair in self.pairs for link in pair))
        self.indices = [kinematics.get_link_index(link) for link in self.links]
        local = {name:i for i,name in enumerate(self.links)}
        self.pair_indices = [(local[a],local[b]) for a,b in self.pairs]
        urdf_path = Path(urdf_path)
        links = {node.attrib['name']:node for node in ET.parse(urdf_path).getroot().findall('link')}
        root = ET.Element('mujoco', model='collision_distance_only')
        ET.SubElement(root,'compiler',angle='radian')
        assets,world = ET.SubElement(root,'asset'),ET.SubElement(root,'worldbody')
        geometry_names=[]
        for i,name in enumerate(self.links):
            body=ET.SubElement(world,'body',name=f'link_{i}',mocap='true')
            node=links[name]; names=[]
            for j,geom in enumerate(node.findall('visual' if 'rubber_hand' in name else 'collision')):
                origin=geom.find('origin')
                xyz='0 0 0' if origin is None else origin.get('xyz','0 0 0')
                rpy='0 0 0' if origin is None else origin.get('rpy','0 0 0')
                quat=Rotation.from_euler('xyz',np.fromstring(rpy,sep=' ')).as_quat()
                attrs=dict(name=f'geom_{i}_{j}',pos=xyz,quat=' '.join(map(str,np.roll(quat,1))))
                geometry=geom.find('geometry'); mesh=geometry.find('mesh')
                if mesh is not None:
                    asset=f'mesh_{i}_{j}'
                    ET.SubElement(assets,'mesh',name=asset,file=str((urdf_path.parent/mesh.attrib['filename']).resolve()),scale=mesh.get('scale','1 1 1'))
                    attrs.update(type='mesh',mesh=asset)
                elif geometry.find('sphere') is not None:
                    attrs.update(type='sphere',size=geometry.find('sphere').attrib['radius'])
                elif geometry.find('cylinder') is not None:
                    g=geometry.find('cylinder')
                    attrs.update(type='cylinder',size=f"{g.attrib['radius']} {float(g.attrib['length'])/2}")
                elif geometry.find('box') is not None:
                    half=np.fromstring(geometry.find('box').attrib['size'],sep=' ')/2
                    attrs.update(type='box',size=' '.join(map(str,half)))
                else:
                    raise ValueError(f'Unsupported collision geometry on {name}')
                ET.SubElement(body,'geom',**attrs);names.append(attrs['name'])
            if not names:raise ValueError(f'No collision geometry on {name}')
            geometry_names.append(names)
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        self.geometry_ids=[[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in names] for names in geometry_names]
        self.mocap_ids=[int(self.model.body_mocapid[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'link_{i}')]) for i in range(len(self.links))]

    def evaluate(self, snapshot):
        poses=snapshot['transforms'][:,self.indices]
        if poses.device.type!='cpu':
            raise ValueError('Convex mesh collision currently requires CPU dynamics')
        distances=poses.new_empty(poses.shape[0],len(self.pairs))
        rows=poses.new_zeros(poses.shape[0],len(self.pairs),35)
        for batch in range(poses.shape[0]):
            transforms=poses[batch].detach().numpy()
            self.data.mocap_pos[self.mocap_ids]=transforms[:,:3]
            self.data.mocap_quat[self.mocap_ids]=np.roll(transforms[:,3:],1,axis=1)
            self.mujoco.mj_kinematics(self.model,self.data)
            for pair,(a,b) in enumerate(self.pair_indices):
                best=self.cutoff; witness=None
                for ga in self.geometry_ids[a]:
                    for gb in self.geometry_ids[b]:
                        segment=np.zeros(6)
                        distance=self.mujoco.mj_geomDistance(self.model,self.data,ga,gb,self.cutoff,segment)
                        if distance<best:best,witness=distance,segment.copy()
                distances[batch,pair]=best
                if witness is None:continue
                # Signed-distance witnesses give the separation/penetration normal.
                delta=witness[:3]-witness[3:]
                if abs(best)<1e-9 or not np.isfinite(delta).all() or np.linalg.norm(delta)<1e-10:
                    raise RuntimeError('Degenerate convex distance witness; cannot certify a collision row')
                normal=poses.new_tensor(delta/best)
                normal=normal/normal.norm()
                point_jac=[]
                for link,point in ((a,witness[:3]),(b,witness[3:])):
                    index=self.indices[link]
                    offset=poses.new_tensor(point)-poses[batch,link,:3]
                    point_jac.append(snapshot['origin_jac'][batch,index]-_skew(offset)@snapshot['angular_jac'][batch,index])
                rows[batch,pair]=normal@(point_jac[0]-point_jac[1])
        if not torch.isfinite(distances).all() or not torch.isfinite(rows).all():
            raise RuntimeError('Nonfinite convex collision geometry')
        return distances,rows
