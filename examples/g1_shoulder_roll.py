"""Optional shoulder-roll posture weights; task targets and constraints stay intact."""
import math


def shoulder_roll_multiplier(time_s, start_s, ramp_s, maximum):
    if (not math.isfinite(time_s) or time_s < 0 or math.isnan(start_s) or start_s < 0
            or not math.isfinite(ramp_s) or ramp_s <= 0
            or not math.isfinite(maximum) or maximum < 1):
        raise ValueError("Shoulder weights need nonnegative time/start, positive ramp and scale >= 1")
    phase = min(1., max(0., (time_s - start_s) / ramp_s))
    return 1. + (maximum - 1.) * phase**3 * (10. - 15.*phase + 6.*phase**2)


def apply_shoulder_roll_weights(requests, joint_names, multiplier, side="both"):
    if side not in ("left", "right", "both") or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Expected left/right/both and a finite multiplier >= 1")
    if multiplier == 1:
        return
    names = tuple(joint_names)
    indices = [names.index(f"{s}_shoulder_roll_joint") for s in ("left", "right")
               if side in (s, "both")]
    posture = [request for request in requests if request.name == "joint_posture"]
    if len(posture) != 1 or posture[0].metric.shape[-2:] != (len(names), len(names)):
        raise ValueError("Expected one posture task with the recorded joint order")
    request = posture[0]
    factors = request.metric.new_ones(len(names))
    factors[indices] = math.sqrt(multiplier)
    request.metric = request.metric * factors[None, :, None] * factors[None, None, :]
