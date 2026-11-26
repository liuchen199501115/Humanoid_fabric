"""Map named joints to position and floating-base acceleration coordinates."""


def named_braking_coordinates(joint_names, selected):
    names, selected = list(joint_names), list(selected)
    if len(set(names)) != len(names):
        raise ValueError('Joint order contains duplicate names')
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select at least one distinct joint for braking')
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError('Unknown braking joints: ' + ', '.join(sorted(unknown)))
    indices = [names.index(name) for name in selected]
    return indices, [6 + index for index in indices]
