"""Native convex distances and generalized-coordinate derivatives offline."""
from pathlib import Path
import numpy as np,yaml,mujoco
from fabrics_sim.reference.convex_collision import ConvexLinkCollision

class ReferenceConvexModel:
    def __init__(self,dynamics,urdf):
        self.dynamics=dynamics
        config=yaml.safe_load(Path('src/fabrics_sim/fabric_params/g1_dynamics_aware.yaml').read_text())['collision']
        spheres=config['spheres'];pairs=[(spheres[a]['link'],spheres[b]['link']) for a,b in config['pairs']]
        wrists=[f'{s}_{j}_link' for s in ['left','right'] for j in ['wrist_roll','wrist_pitch','wrist_yaw']]+['left_rubber_hand','right_rubber_hand']
        hips=[f'{s}_hip_{j}_link' for s in ['left','right'] for j in ['pitch','roll','yaw']]
        for wrist in wrists:
            pairs.extend((wrist,hip) for hip in hips)
            opposite='right' if wrist.startswith('left') else 'left'
            pairs.extend((wrist,body) for body in ['torso_link']+[f'{opposite}_shoulder_{j}_link' for j in ['pitch','roll','yaw']])
        pairs.append(('left_ankle_roll_link','right_ankle_roll_link'))
        pairs=[tuple('pelvis_contour_link' if link=='pelvis' else link for link in pair) for pair in pairs]
        self.pairs=list(dict.fromkeys(pairs))
        class Indices:
            def get_link_index(_,name):
                index=mujoco.mj_name2id(dynamics.model,mujoco.mjtObj.mjOBJ_BODY,name)
                if index<0:raise ValueError(name)
                return index
        self.geometry=ConvexLinkCollision(urdf,Indices(),self.pairs)

    def evaluate(self,q,activation=.08):
        model=self.dynamics;g=self.geometry;transform=model._state(q)
        g.data.mocap_pos[g.mocap_ids]=model.data.xpos[g.indices]
        g.data.mocap_quat[g.mocap_ids]=model.data.xquat[g.indices]
        mujoco.mj_kinematics(g.model,g.data);rows=[];gaps=[];indices=[]
        for j,(a,b) in enumerate(g.pair_indices):
            best=g.cutoff;witness=None
            for ga in g.geometry_ids[a]:
                for gb in g.geometry_ids[b]:
                    segment=np.zeros(6);gap=mujoco.mj_geomDistance(g.model,g.data,ga,gb,g.cutoff,segment)
                    if gap<best:best=gap;witness=segment.copy()
            if best>=activation:continue
            normal=(witness[:3]-witness[3:])/best
            normal/=np.linalg.norm(normal)
            point_rows=[]
            for link,point in [(a,witness[:3]),(b,witness[3:])]:
                jp=np.zeros((3,35));jr=np.zeros((3,35))
                mujoco.mj_jac(model.model,model.data,jp,jr,point,g.indices[link])
                point_rows.append(jp@transform)
            rows.append(normal@(point_rows[0]-point_rows[1]));gaps.append(best);indices.append(j)
        return np.array(gaps),np.array(rows).reshape(-1,35),np.array(indices)
