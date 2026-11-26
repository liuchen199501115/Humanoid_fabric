"""Detect a warm contact-restoration plateau and request a fresh cold solve.

This never admits a candidate or changes a physical tolerance. The caller
must still run the cold fallback and its full nonlinear verification.
"""
def contact_restoration_stalled(history, releases, window=4):
    if len(history) < window:
        return False
    recent=history[-window:]
    if any(change['iteration'] >= recent[0]['iteration'] for change in releases):
        return False
    gaps=[row.get('nonlinear_support_gap',float('inf')) for row in recent]
    if not all(5e-5 < gap < float('inf') for gap in gaps) or min(gaps) < .9*gaps[0]:
        return False
    limits={'nonlinear_collision_barrier':.02,'nonlinear_base_force_residual':20.,
            'nonlinear_base_moment_residual':5.,'nonlinear_joint_residual':5.,
            'nonlinear_integration_position':1e-4,'nonlinear_integration_orientation':1e-4,
            'nonlinear_integration_joint':1e-4}
    return all(row.get('elastic_slacks') and all(row.get(key,float('inf'))<bound for key,bound in limits.items()) for row in recent)
