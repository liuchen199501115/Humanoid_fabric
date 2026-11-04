"""Verify simulation control cadence and separately report measured QP latency."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize_control_rate(data, qp_hz=500, torque_hz=500):
    if min(qp_hz, torque_hz) <= 0 or torque_hz % qp_hz:
        raise ValueError('Torque rate must be a positive integer multiple of QP rate')
    required = ('history_time', 'servo_time', 'servo_physics_time', 'servo_qp_update_index')
    missing = [key for key in required if key not in data]
    if missing:
        return dict(simulation_rate_passed=False, missing_fields=missing)
    t, servo, physics, index = [np.asarray(data[key]).reshape(-1) for key in required]
    if len(t) < 2 or len(servo) < 2 or len(physics) != len(servo) or len(index) != len(servo):
        raise ValueError('Need matching nonempty control and torque histories')
    finite = all(np.isfinite(value).all() for value in (t, servo, physics, index))
    expected_index = np.repeat(np.arange(len(t)), torque_hz // qp_hz)
    qp_error = float(np.max(np.abs(np.diff(t)-1/qp_hz)))
    torque_error = float(np.max(np.abs(np.diff(servo)-1/torque_hz)))
    physics_error = float(np.max(np.abs(np.diff(physics)-1/torque_hz)))
    indices_match = bool(np.array_equal(index, expected_index))
    aligned = (len(expected_index) == len(servo) and bool(np.allclose(
        np.repeat(t, torque_hz//qp_hz),
        servo-np.tile(np.arange(torque_hz//qp_hz)/torque_hz,len(t)),rtol=0,atol=1e-8)))
    result = dict(qp_hz=qp_hz, torque_hz=torque_hz,
        qp_updates=len(t), torque_updates=len(servo),
        maximum_qp_period_error_seconds=qp_error,
        maximum_torque_period_error_seconds=torque_error,
        maximum_physics_period_error_seconds=physics_error,
        torque_uses_expected_qp_updates=indices_match,
        simulation_rate_passed=bool(finite and aligned and indices_match and
                                   max(qp_error,torque_error,physics_error) < 1e-7))
    if 'history_qp_wall_seconds' in data:
        wall = np.asarray(data['history_qp_wall_seconds']).reshape(-1)
        valid = len(wall) == len(t) and np.isfinite(wall).all() and (wall >= 0).all()
        result['qp_wall_time_valid'] = bool(valid)
        if valid:
            result.update(qp_wall_median_seconds=float(np.median(wall)),
                          qp_wall_p99_seconds=float(np.percentile(wall,99)),
                          qp_wall_max_seconds=float(wall.max()),
                          all_qp_solves_within_period=bool((wall <= 1/qp_hz).all()))
    result['scope'] = 'Simulation cadence only. QP latency excludes other controller work; this is not a real-time deployment certificate.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--qp-hz',type=int,default=500)
    parser.add_argument('--torque-hz',type=int,default=500)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with np.load(args.recording) as data:
        result=summarize_control_rate(data,args.qp_hz,args.torque_hz)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
