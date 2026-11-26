"""Keep an intended outgoing foot's soft pose task during measured double support.
This does not modify contact rows, wrench bounds, forces, or contact state.
"""
def activate_transition_swing(requests, contact_mode, intended_mode, normal_forces, minimum_support_load=100.):
    if int(contact_mode)!=3 or int(intended_mode) not in (1,2):
        return None
    incoming=int(intended_mode)-1
    if float(normal_forces[incoming])<minimum_support_load:
        return None
    outgoing=1-incoming
    for request in requests:
        if request.name=='feet_pose':
            request.active[:,6*outgoing:6*outgoing+6]=True
            return tuple(range(6*outgoing,6*outgoing+6))
    raise ValueError('Transition swing activation requires a feet_pose request')
