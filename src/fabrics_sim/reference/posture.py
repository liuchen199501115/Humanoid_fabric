"""A flexed-knee seed for managed walking, preserving each foot's pitch sum."""
import torch


def crouched_posture(joints, joint_names, elapsed, duration=1.5, minimum_knee=.5):
    if duration<=0 or minimum_knee<0:
        raise ValueError('Invalid crouch configuration')
    s=max(0.,min(1.,float(elapsed)/duration))
    blend=10*s**3-15*s**4+6*s**5
    result=joints.clone()
    for side in ('left','right'):
        hip=joint_names.index(side+'_hip_pitch_joint')
        knee=joint_names.index(side+'_knee_joint')
        ankle=joint_names.index(side+'_ankle_pitch_joint')
        bend=(minimum_knee-joints[:,knee]).clamp_min(0)*blend
        result[:,hip]-=.5*bend
        result[:,knee]+=bend
        result[:,ankle]-=.5*bend
    return result, blend


def crouched_posture_kinematics(joints, velocity, acceleration, joint_names, elapsed,
                                duration=1.5, minimum_knee=.5):
    """Analytic derivatives of the existing crouch blend, away from its knee kink."""
    if velocity.shape != joints.shape or acceleration.shape != joints.shape:
        raise ValueError('Crouch derivatives must match joint positions')
    result, blend = crouched_posture(joints,joint_names,elapsed,duration,minimum_knee)
    v,a=velocity.clone(),acceleration.clone()
    s=max(0.,min(1.,float(elapsed)/duration))
    blend_v=30*s*s*(1-s)**2/duration
    blend_a=60*s*(1-s)*(1-2*s)/duration**2
    for side in ('left','right'):
        indices=[joint_names.index(side+'_'+name+'_joint') for name in ('hip_pitch','knee','ankle_pitch')]
        knee=indices[1];active=joints[:,knee]<minimum_knee
        deficit=(minimum_knee-joints[:,knee]).clamp_min(0)
        bend_v=deficit*blend_v-velocity[:,knee]*active*blend
        bend_a=deficit*blend_a-2*velocity[:,knee]*active*blend_v-acceleration[:,knee]*active*blend
        for index, factor in zip(indices,(-.5,1.,-.5)):
            v[:,index]+=factor*bend_v;a[:,index]+=factor*bend_a
    return result,v,a,blend
