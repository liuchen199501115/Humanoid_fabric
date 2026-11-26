"""Release a measured stop once the previous solution no longer loads it."""
import torch


def loaded_stop_directions(directions, previous_acceleration, previous_reaction):
    if (directions.shape != previous_acceleration.shape
            or directions.shape != previous_reaction.shape
            or not all(torch.isfinite(x).all() for x in
                       (directions, previous_acceleration, previous_reaction))):
        raise ValueError('Stop modes require matching finite joint arrays')
    # A zero reaction must not permanently lock a stationary joint. When
    # released, an inward motor command can leave the stop on the next tick.
    # A fresh outward acceleration prediction at a measured stationary stop
    # supplies evidence to activate it; no additional optimization is run.
    pressing = directions*previous_acceleration > 1.
    loaded = -directions*previous_reaction > .05
    return directions*(pressing | loaded)
