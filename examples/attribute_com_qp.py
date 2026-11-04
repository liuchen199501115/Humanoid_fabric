"""Offline constraint ablations and held-model torque-response checks. Never controls physics."""
import argparse
import numpy as np
import osqp
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--history", required=True)
    parser.add_argument("--response", help="Optional 2 ms response snapshot from the same controller tick.")
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as data:
        p = {k: data["problem_" + k][0].astype(float) for k in
             ("H", "h", "G", "b", "lb", "ub", "A_eq", "b_eq")}
        mass = data["mass"][0].astype(float)
        J = data["contact_jacobian"][0].astype(float)
        mode = int(data["contact_mode"].item())
        mask = np.repeat([mode in (1, 3), mode in (2, 3)], 6)
        J = J * mask[:, None]
        inv = np.linalg.inv(mass)
        response_force = -np.linalg.solve(J @ inv @ J.T + np.diag(~mask), (J @ inv)[:, 6:])
        response_acc = inv[:, 6:] + inv @ J.T @ response_force
        comJ = data["com_jacobian"][0].astype(float)
        curvature = data["diagnostic_com_curvature"][0]
        correction = data["correction"][0]
        bias, gain = data["servo_bias"][0], data["servo_acc_gain"][0]
        transform = np.zeros((35, 76)); transform[:, :35] = np.eye(35)
        transform[:, 6:35] += response_acc * gain[None, :]
        C = comJ @ transform
        offset = comJ @ (response_acc @ bias - correction) + curvature
        requested = data["diagnostic_com_acc_requested"][0]
        original = data["problem_solution"][0]
        nominal = data["nominal_acc"][0]
        total = data["total_torque"][0]
        ff = data["nominal_torque"][0]
        order = np.argsort(data["fabric_to_isaac_joint"])
        time = float(data["time"])
        start_v = data["qd"][0]
        start_com_v = data["diagnostic_com_velocity"][0]
        bias_force = data["bias_force"][0]
    if p["A_eq"].shape != (47, 76) or p["G"].shape != (82, 76):
        raise ValueError("Only standard 47-equality / 82-inequality servo QPs are supported.")
    matrix = np.vstack((p["A_eq"], p["G"], np.eye(76)))
    lo = np.r_[p["b_eq"], np.full(82, -np.inf), p["lb"]]
    hi = np.r_[p["b_eq"], p["b"], p["ub"]]
    groups = {"none": [], "acc_bounds": np.arange(129,164),
              "contact_equalities": np.arange(35,47),
              "wrench_all": np.arange(47,71), "effort": np.arange(71,129),
              "wrench_tangential": 47+np.array([0,1,2,3,12,13,14,15]),
              "wrench_cop": 47+np.array([6,7,8,9,18,19,20,21]),
              "wrench_yaw": 47+np.array([10,11,22,23])}
    def solve(rows, exact_com=False):
        lower, upper = lo.copy(), hi.copy()
        lower[rows], upper[rows] = -np.inf, np.inf
        A = matrix
        if exact_com:
            A = np.vstack((A,C)); lower=np.r_[lower,requested-offset]; upper=np.r_[upper,requested-offset]
        solver = osqp.OSQP()
        solver.setup(P=sparse.triu(sparse.csc_matrix(p["H"]),format="csc"), q=p["h"],
                     A=sparse.csc_matrix(A), l=lower, u=upper, verbose=False,
                     eps_abs=1e-6, eps_rel=1e-6, max_iter=30000, polish=True)
        return solver.solve(), A, lower, upper
    print("snapshot time",time,"mode",mode,"requested COM",requested)
    print("controller predicted COM", C @ original + offset)
    for name, rows in groups.items():
        result,A,lower,upper=solve(rows)
        if result.info.status_val in (1,2):
            x=result.x
            violation=max(0.,np.max(lower-A@x),np.max(A@x-upper))
            print("OFFLINE remove",name,result.info.status,"COM",np.round(C@x+offset,5),"violation",violation)
            if name=="none":
                print("replay/controller max difference",np.max(abs(x-original)))
                for row in np.argsort(abs(result.y[47:]))[-12:][::-1]+47:
                    print("active row",row,"G/box index",row-47,"dual",round(result.y[row],4),
                          "slack",round(min(upper[row]-(A@x)[row],(A@x)[row]-lower[row]),6))
        else:
            print("OFFLINE remove",name,result.info.status)
        exact,exact_A,exact_lo,exact_hi=solve(rows, True)
        print("  exact requested COM feasibility:",exact.info.status)
        if name == "none" and exact.info.status_val in (1,2):
            x=exact.x
            print("  exact COM residual",np.max(abs(C@x+offset-requested)),
                  "constraint violation",max(0.,np.max(exact_lo-exact_A@x),np.max(exact_A@x-exact_hi)))
            print("  exact COM objective",.5*x@p["H"]@x+p["h"]@x,
                  "original objective",.5*original@p["H"]@original+p["h"]@original)
    with np.load(args.history, allow_pickle=False) as h:
        i=int(np.argmin(abs(h["history_time"].reshape(-1)-time)))
        if abs(float(h["history_time"][i])-time)>1e-5:
            raise ValueError("Snapshot does not align with history.")
        start=int(round(time/.002))
        applied=h["servo_torque"][start:start+10,0,:][:,order]
        print("first applied/predicted torque max error",np.max(abs(applied[0]-total)))
        mean=applied.mean(axis=0)
        print("mean applied vs initial torque RMS",np.sqrt(np.mean((mean-total)**2)))
        print("COM held model initial torque",comJ @ (nominal+response_acc@(total-ff))+curvature)
        print("COM held model mean torque",comJ @ (nominal+response_acc@(mean-ff))+curvature)
        print("COM observed interval",h["history_com_acc_observed"][i].reshape(3))
        # First 2 ms joint acceleration: compare like-for-like before averaging PD.
        joint_acc=(h["servo_velocity"][start+1,0,order]-h["servo_velocity"][start,0,order])/.002
        prediction=(nominal+response_acc@(total-ff))[6:]
        print("first 2ms joint acceleration prediction RMS error",np.sqrt(np.mean((joint_acc-prediction)**2)))
    if args.response:
        with np.load(args.response, allow_pickle=False) as r:
            dt=float(r["dt"])
            a=(r["qd"][0]-start_v)/dt
            nextJ=r["contact_jacobian"][0]*mask[:,None]
            contact_acc=(nextJ@r["qd"][0]-J@start_v)/dt
            com_acc=(r["com_velocity"][0]-start_com_v)/dt
            torque=r["applied_torque"][0]
        print("2ms observed COM",com_acc,"predicted",C@original+offset)
        print("2ms observed foot accelerations [linear/angular]",contact_acc.reshape(2,6))
        rhs=mass@a+bias_force-np.r_[np.zeros(6),torque]
        implied=np.linalg.lstsq(J.T,rhs,rcond=None)[0]
        print("2ms implied contact wrench",implied.reshape(2,6))
        print("2ms inverse dynamics residual RMS",np.sqrt(np.mean((J.T@implied-rhs)**2)))
        # Joint hard limits provide additional generalized impulses not in a
        # two-rigid-foot response model. Fit the active right ankle-roll axis
        # diagnostically; this is NOT a modification to the live model.
        stop_axis=np.eye(35)[:,17:18]
        extended=np.c_[J.T,stop_axis]
        fitted=np.linalg.lstsq(extended,rhs,rcond=None)[0]
        print("2ms with ankle-stop reaction residual RMS",np.sqrt(np.mean((extended@fitted-rhs)**2)),
              "fitted stop torque",fitted[-1])
    print("Ablations are diagnostic only; instantaneous and interval-average models are not identical.")


if __name__ == "__main__":
    main()
