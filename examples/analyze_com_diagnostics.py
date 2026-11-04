"""Compare instantaneous COM predictions with forward-interval observations."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float, default=float("inf"))
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as data:
        time = data["history_time"].reshape(-1)
        mask = ((time >= args.start) & (time < args.end)
                & data["history_com_observed_valid"].reshape(-1))
        if not mask.any():
            raise ValueError("No observed intervals in selected range; run with --diagnose-com.")
        stages = ["requested", "fabric", "projected", "qp_nominal", "qp_predicted", "observed"]
        vectors = {name: data["history_com_acc_" + name][mask].reshape(-1, 3) for name in stages}
        print(f"{mask.sum()} intervals, t=[{time[mask].min():.3f}, {time[mask].max():.3f}] s; units m/s^2")
        print("Instantaneous model predictions vs interval-average observed acceleration; contact impulses may dominate.")
        for name, value in vectors.items():
            print(f"{name:14s} mean_xyz={value.mean(0).round(5)} rms_norm={np.sqrt(np.mean(np.sum(value**2, axis=1))):.5f}")
        for first, second in zip(stages[:-1], stages[1:]):
            error = vectors[second] - vectors[first]
            print(f"{first:12s} -> {second:12s}: delta_rms={np.sqrt(np.mean(np.sum(error**2, axis=1))):.5f}")
        print("Differences localize disagreement; they do not alone prove which task or model caused it.")


if __name__ == "__main__":
    main()
