"""Task selection for physically constrained source-step swing tracking."""


def source_swing_priority(enabled, has_source_step, switching, finished, mode, *, pose=False, body_priority=None):
    """Keep balance first, then the airborne foot's position or full pose.

    Double support and load transfers retain their existing task selection.
    This only selects soft objectives; physical QP constraints are unchanged.
    """
    if body_priority not in (None, 'balance', 'swing'):
        raise ValueError('Unknown source body priority stage')
    if mode not in (0, 1, 2, 3):
        raise ValueError('Expected a two-foot contact mode')
    if not enabled or not has_source_step or switching or finished or mode not in (1, 2):
        return None
    swing = 2-mode
    rows = tuple(6*swing+i for i in range(6 if pose else 3))
    body = ('body_orientation', 'source_pelvis_orientation') if body_priority else ()
    balance = ('center_of_mass',)+(body if body_priority == 'balance' else ())
    swing_tasks = ('feet_pose',)+(body if body_priority == 'swing' else ())
    return (balance+swing_tasks, {'feet_pose': rows}, (balance, swing_tasks))
