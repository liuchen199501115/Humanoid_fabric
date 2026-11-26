"""The runtime sphere-cover barrier evaluated in the independent URDF model."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import mujoco
from centroidal_pose_linearization import skew
from mujoco_reference_dynamics import ReferenceDynamics


def runtime_geometry(urdf):
    # Invoke the same geometry builder, without creating a simulator or fabric.
    import yaml
    from fabrics_sim.fabrics.dynamics_aware import DynamicsAwareTasks
    config=yaml.safe_load(Path('src/fabrics_sim/fabric_params/g1_dynamics_aware.yaml').read_text())
    result={}
    def capture(spheres,pairs,weights=None):
        result.update(spheres=spheres,pairs=pairs[len(config['collision']['pairs']):],
                      safe_distance=config['collision']['safe_distance'])
    proxy=SimpleNamespace(config=config,fabric=SimpleNamespace(urdf_path=urdf),_set_collision_geometry=capture)
    DynamicsAwareTasks.enable_wrist_hip_avoidance(proxy,include_upper_body=True)
    return result


class CollisionDynamics(ReferenceDynamics):
    def __init__(self,urdf,names,geometry=None):
        super().__init__(urdf,names)
        self.collision_geometry=runtime_geometry(urdf) if geometry is None else geometry
        g=self.collision_geometry;names=list(g['spheres']);lookup={name:k for k,name in enumerate(names)}
        self.collision_pairs=np.array([[lookup[a],lookup[b]] for a,b in g['pairs']],dtype=int)
        spheres=list(g['spheres'].values())
        self.collision_bodies=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,s['link']) for s in spheres])
        if np.any(self.collision_bodies<0):raise ValueError('Collision link absent from URDF dynamics')
        self.collision_centers=np.array([s['center'] for s in spheres])
        self.collision_radii=np.array([s['radius'] for s in spheres]);self.safe_distance=float(g['safe_distance'])
        # Self distance only depends on joints between the two links.
        # Shared ancestors, including the floating base, move both rigidly.
        dof_columns={int(dof):6+j for j,dof in enumerate(self.vindex)}
        ancestors=[]
        for body in self.collision_bodies:
            columns=set();node=int(body)
            while node:
                for joint in range(int(self.model.body_jntadr[node]),int(self.model.body_jntadr[node]+self.model.body_jntnum[node])):
                    dof=int(self.model.jnt_dofadr[joint])
                    if dof in dof_columns:columns.add(dof_columns[dof])
                node=int(self.model.body_parentid[node])
            ancestors.append(columns)
        self.collision_dependencies=np.zeros((len(self.collision_pairs),35),dtype=bool)
        for row,(a,b) in enumerate(self.collision_pairs):
            self.collision_dependencies[row,list(ancestors[a]^ancestors[b])]=True

    def evaluate(self,q,v,a=None,kinematic_bias=False):
        result=super().evaluate(q,v,a,kinematic_bias=kinematic_bias)
        if kinematic_bias:result['collision']=self.collision_state(q,v)
        return result

    def collision_state(self,q,v):
        from scipy.spatial.transform import Rotation
        m,d=self.model,self.data;T=np.zeros((35,35));T[:3,:3]=np.eye(3);T[3:6,3:6]=Rotation.from_quat(q[3:7]).as_matrix().T;T[self.vindex,np.arange(6,35)]=1
        bodies=self.collision_bodies;rotations=d.xmat[bodies].reshape(-1,3,3)
        offsets=np.einsum('nij,nj->ni',rotations,self.collision_centers)
        positions=d.xpos[bodies]+offsets;jp_all=np.empty((len(bodies),3,35));jr_all=np.empty_like(jp_all)
        origin_bias=np.empty((len(bodies),3));angular_bias=np.empty_like(origin_bias)
        for body in np.unique(bodies):
            indices=np.flatnonzero(bodies==body)
            jp=np.zeros((3,35));jr=jp.copy();jpd=jp.copy();jrd=jp.copy()
            mujoco.mj_jacBody(m,d,jp,jr,body);mujoco.mj_jacDot(m,d,jpd,jrd,d.xpos[body],body)
            jp_all[indices]=jp@T;jr_all[indices]=jr@T
            origin_bias[indices]=jpd@d.qvel;angular_bias[indices]=jrd@d.qvel
        omega=jr_all@v
        # Cross products broadcast across the generalized-coordinate columns.
        jac=jp_all+np.cross(jr_all.transpose(0,2,1),offsets[:,None,:]).transpose(0,2,1)
        bias=origin_bias+np.cross(angular_bias,offsets)+np.cross(omega,np.cross(omega,offsets))
        ia,ib=self.collision_pairs.T;delta=positions[ia]-positions[ib];length=np.linalg.norm(delta,axis=1);normal=delta/np.maximum(length[:,None],1e-8)
        relative_jac=jac[ia]-jac[ib];A=np.einsum('ni,nij->nj',normal,relative_jac)
        relative_v=relative_jac@v;speed=A@v;distance=length-self.collision_radii[ia]-self.collision_radii[ib]
        acceleration_bias=np.einsum('ni,ni->n',normal,bias[ia]-bias[ib])+(np.einsum('ni,ni->n',relative_v,relative_v)-speed**2)/np.maximum(length,1e-8)
        return dict(dependencies=self.collision_dependencies,distance=distance,jacobian=A,bias=acceleration_bias,speed=speed,
                    bound=acceleration_bias+16*speed+64*(distance-self.safe_distance))


def barrier_residual(state,acceleration):
    c=state['collision'];return -c['jacobian']@acceleration-c['bound']
