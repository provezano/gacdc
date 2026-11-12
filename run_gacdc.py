"""Command-line entry point for GACDC.

Runs the evolutionary algorithm in GACDC.py on one instance file without
editing the source. Defaults match the values set in GACDC.main().
The algorithm itself is not changed; see CHANGES.md for known issues.
"""
import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import GACDC  # noqa: E402
from Utils import Utils  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run GACDC on one cross-docking instance.")
    p.add_argument("instance", help="path to a .dat instance file")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (default: none, as in the original code)")
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--crossover-rate", type=float, default=0.8)
    p.add_argument("--mutation-rate", type=float, default=0.2)
    p.add_argument("--total-select", type=int, default=2,
                   help="chromosomes selected per generation")
    p.add_argument("--runs", type=int, default=1, help="independent runs")
    p.add_argument("--quiet", action="store_true",
                   help="hide the per-generation log printed by the GA")
    return p.parse_args(argv)


def run(args):
    j1, j2, m1, m2, jobs_1, jobs_2 = Utils.read_from_file(args.instance)
    results = []
    for r in range(args.runs):
        if args.seed is not None:
            random.seed(args.seed + r)
            np.random.seed(args.seed + r)
            # evolutionary_algorithm() calls random.seed(None) on entry, which
            # would discard the seed. Skip that call only when a seed is given.
            orig_seed = GACDC.random.seed
            GACDC.random.seed = lambda *a, **k: None
        stdout = sys.stdout
        if args.quiet:
            sys.stdout = open(os.devnull, "w")
        start = time.time()
        try:
            best, pop_mean, pop_std = GACDC.evolutionary_algorithm(
                m1, m2, jobs_1, jobs_2, args.pop_size, args.crossover_rate,
                args.mutation_rate, args.generations, args.total_select)
        finally:
            if args.quiet:
                sys.stdout.close()
                sys.stdout = stdout
            if args.seed is not None:
                GACDC.random.seed = orig_seed
        elapsed = time.time() - start
        results.append((best.makespan_1, best.makespan_2, pop_mean, pop_std, elapsed))
    print("run\tmakespan_stage1\tmakespan\tpop_mean\tpop_std\tseconds")
    for i, (a, b, c, d, e) in enumerate(results, 1):
        print(f"{i}\t{a}\t{b}\t{c:.3f}\t{d:.3f}\t{e:.2f}")
    return results


if __name__ == "__main__":
    run(parse_args())
