"""Replay recorded controller QPs on CPU without changing their constraints.

Repeated solves of an identical QP are a microbenchmark, not a certificate
for changing control problems, a GPU controller, or real-time deployment.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.qp_refinement import _violation


def benchmark(path, backend='daqp', repeats=3):
    if repeats<1:raise ValueError('At least one solve is required')
    if backend=='daqp':
        from fabrics_sim.solver.daqp_backend import DAQPBackend
        factory=DAQPBackend
    elif backend=='osqp':
        from fabrics_sim.solver.osqp_backend import OSQPBackend
        factory=lambda:OSQPBackend(max_iter=100000,reuse_workspace=True)
    else:raise ValueError('Unsupported benchmark backend')
    rows=[]
    with np.load(path,allow_pickle=False) as data:
        levels=sorted({key[:-1] for key in data.files
                       if key.startswith('priority_level_') and key.endswith('_H')},
                      key=lambda prefix:int(prefix.split('_')[2]))
        for prefix in (levels or ['priority_problem_'])+['problem_']:
            if prefix+'H' not in data:continue
            names=('H','h','G','b','lb','ub','A_eq','b_eq')
            arrays={name:data[prefix+name].astype(np.float64) for name in names}
            qp=QPData(**{name:torch.from_numpy(value) for name,value in arrays.items()})
            solver=factory();times=[]
            for _ in range(repeats):
                start=time.perf_counter();solution=solver.solve(qp)
                times.append(time.perf_counter()-start)
            violation=max(_violation({name:value[i] for name,value in arrays.items()},solution.z[i].numpy())
                          for i in range(len(qp.h)))
            rows.append(dict(recording=str(path),stage=prefix,backend=backend,
                cold_seconds=times[0],repeat_seconds=times[1:],
                accepted=bool(solution.status_ok.all()),physical_residual=float(violation),
                status=solver.last_status,iterations=solution.iters))
    if not rows:raise ValueError(f'No recorded QP in {path}')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots',type=Path,nargs='+')
    parser.add_argument('--backend',choices=('daqp','osqp'),default='daqp')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    rows=[]
    for path in args.snapshots:
        result=benchmark(path,args.backend,args.repeats)
        rows.extend(result)
        print(json.dumps(result),flush=True)
    if args.output:args.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
