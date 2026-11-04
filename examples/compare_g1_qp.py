"""Replay a saved controller QP with OSQP; never run robot physics or change gains."""
import argparse
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--max-iter", type=int, default=20000,
                        help="Offline iteration budget; does not change controller settings.")
    parser.add_argument("--check-feasibility", action="store_true",
                        help="Also check the unchanged hard constraints with HiGHS linear programming.")
    parser.add_argument("--diagnose-constraints", action="store_true",
                        help="Offline only: remove one constraint group at a time to diagnose infeasibility.")
    args = parser.parse_args()
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    with np.load(args.snapshot, allow_pickle=False) as data:
        problem = {key: data["problem_" + key][0].astype(np.float64)
                   for key in ("H", "h", "G", "b", "lb", "ub", "A_eq", "b_eq")}
        previous = data["problem_solution"][0].astype(np.float64)
    n = len(problem["h"])
    if args.diagnose_constraints and (problem["A_eq"].shape[0] != 47 or problem["G"].shape[0] != 82):
        parser.error("Constraint-group diagnosis requires the standard 47-equality/82-inequality servo QP.")
    matrix = sparse.vstack((sparse.csc_matrix(problem["A_eq"]),
                            sparse.csc_matrix(problem["G"]), sparse.eye(n)), format="csc")
    lower = np.concatenate((problem["b_eq"], np.full(len(problem["b"]), -np.inf), problem["lb"]))
    upper = np.concatenate((problem["b_eq"], problem["b"], problem["ub"]))
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(sparse.csc_matrix(problem["H"]), format="csc"),
                 q=problem["h"], A=matrix, l=lower, u=upper,
                 eps_abs=1e-6, eps_rel=1e-6, max_iter=args.max_iter,
                 eps_prim_inf=1e-8, eps_dual_inf=1e-8,
                 polish=True, verbose=False, adaptive_rho_interval=25)
    result = solver.solve()
    print("OSQP", result.info.status, "iterations", result.info.iter)
    def report(name, x):
        objective = .5 * x @ problem["H"] @ x + problem["h"] @ x
        residual = max(np.max(lower - matrix @ x), np.max(matrix @ x - upper), 0.)
        print(name, "objective", objective, "max_constraint_violation", residual)
    report("controller", previous)
    if args.check_feasibility:
        from scipy.optimize import linprog
        feasible = linprog(np.zeros(n), A_ub=problem["G"], b_ub=problem["b"],
                           A_eq=problem["A_eq"], b_eq=problem["b_eq"],
                           bounds=list(zip(problem["lb"], problem["ub"])), method="highs")
        print("HiGHS feasibility:", feasible.message)
        if feasible.success:
            report("feasible_only_not_optimized", feasible.x)
    if result.info.status_val in (1, 2):
        report("replay", result.x)
    else:
        if (result.info.status_val == osqp.constant("OSQP_MAX_ITER_REACHED")
                and result.x is not None and np.isfinite(result.x).all()):
            report("iteration_limit_candidate_not_certified_optimal", result.x)
        if args.diagnose_constraints:
            eq_count = len(problem["b_eq"])
            inequality_count = len(problem["b"])
            groups = {
                "command_acceleration_bounds": slice(eq_count + inequality_count, eq_count + inequality_count + 35),
                "contact_acceleration_equalities": slice(35, eq_count),
                "wrench_inequalities": slice(eq_count, eq_count + 24),
                "effort_inequalities": slice(eq_count + 24, eq_count + inequality_count),
            }
            for name, rows in groups.items():
                relaxed_lower, relaxed_upper = lower.copy(), upper.copy()
                relaxed_lower[rows], relaxed_upper[rows] = -np.inf, np.inf
                solver.update(l=relaxed_lower, u=relaxed_upper)
                solver.warm_start(x=np.zeros(n), y=np.zeros(len(lower)))
                relaxed = solver.solve()
                print("DIAGNOSTIC ONLY without", name, ":", relaxed.info.status)
                if name == "command_acceleration_bounds" and relaxed.info.status_val in (1, 2):
                    for index in range(35):
                        value = relaxed.x[index]
                        if value < problem["lb"][index] - 1e-3 or value > problem["ub"][index] + 1e-3:
                            print("  acceleration coordinate", index, "requested", value,
                                  "allowed", problem["lb"][index], problem["ub"][index])
        raise SystemExit("Independent replay did not establish a solved QP.")


if __name__ == "__main__":
    main()
