# GACDC

A genetic algorithm for truck scheduling in a cross-docking center with several
inbound and outbound docks.

## The problem

Inbound trucks are unloaded at parallel inbound docks (stage 1) and outbound
trucks are loaded at parallel outbound docks (stage 2). An outbound truck can
start only after every inbound truck that carries goods for it has been
unloaded. The goal is to order the trucks so that the last outbound truck
finishes as early as possible (minimum makespan).

## What the code does

- **Encoding** (`Chromosome.py`). A chromosome is a permutation of the inbound
  jobs. Jobs are assigned in that order to the inbound machine that becomes
  free first. Each outbound job gets a release date equal to the latest
  completion among its predecessors. Outbound jobs are then sorted by release
  date (ties broken by longest processing time plus release date) and assigned
  to the outbound machine that becomes free first.
- **Fitness**. The stage-2 makespan (`makespan_2`); lower is better.
- **Initial population** (`Population.py`). Four individuals are built with the
  LPT, SPT and successor-count rules (most and fewest successors first); the
  rest are random permutations.
- **Operators** (`UtilsAE.py`). Crossover: order crossover (OX) and partially
  mapped crossover (PMX). Mutation: swap of two positions, inversion of a
  segment, moving a segment to the front, moving a segment to a random
  position, and a rotation of the list. The operator used at each step is
  drawn from a distribution that is adapted during the run: an operator's
  weight goes up when its offspring improves on the parent and down when it
  gets worse.
- **Loop** (`GACDC.py`, `evolutionary_algorithm`). Each generation selects two
  chromosomes, applies crossover with probability `crossover_rate` and mutation
  with probability `mutation_rate`, and, if anything changed, puts the new
  chromosomes in place of two chromosomes picked by a survival tournament.
  The best chromosome seen is kept.

`UtilsGA.py` holds an older set of operators and is not used by `GACDC.py`.

This is the 2018 research version of the code with a few fixes made in 2026,
listed in `CHANGES.md`. The original upload is kept at the git tag
`v2018-original`. The code has not been re-validated against the tables of the
2025 paper.

## Install and run

Python 3.12 was used for testing.

```bash
pip install -r requirements.txt
python run_gacdc.py Intancias/2maquinas/20jobs/58.dat --seed 3 --generations 20 --pop-size 20
```

Options (defaults are those set in `GACDC.main`):

| option | default |
|---|---|
| `--pop-size` | 100 |
| `--generations` | 1000 |
| `--crossover-rate` | 0.8 |
| `--mutation-rate` | 0.2 |
| `--total-select` | 2 |
| `--runs` | 1 |
| `--seed` | none |
| `--quiet` | off (hides the per-generation log) |

The output is one line per run: stage-1 makespan, makespan, and the mean and
standard deviation of the population's mean fitness across generations.

`python GACDC.py --data-dir Intancias` runs the original batch loop over every
instance found under that folder and writes summaries to `./results/`.

Tests (the last one runs 1000 generations three times, about 20 s):

```bash
python tests/test_smoke.py
python tests/test_fixes.py
```

## Instance format

Plain text, one value per line:

```
n1            number of inbound jobs
n2            number of outbound jobs
m1            number of inbound machines (docks)
m2            number of outbound machines (docks)
(blank)
n1 lines      processing times of the inbound jobs
(blank)
n2 lines      processing times of the outbound jobs
(blank)
n2 rows of n1 values (0/1): entry (i, j) = 1 if outbound job i needs inbound job j
```

`Intancias/` holds the 2-machine instances with 20 jobs and one `40.dat` file
for each of 2, 4 and 10 machines. (The folder name is kept as in the original
upload.) `Instancias.zip` holds about 900 instances
(2, 4 and 10 machines, 20 jobs).

## Reference numbers

Makespan of the best schedule found by `run_gacdc.py` with the default
parameters (population 100, 1000 generations) on the three `40.dat` files,
one run per seed. Python 3.12, numpy 2.5.4.

| instance | jobs (in/out) | docks per stage | seed 3 | seed 4 | seed 7 |
|---|---|---|---|---|---|
| `Intancias/2maquinas/30jobs/40.dat` | 30 / 29 | 2 | 1434 | 1449 | 1447 |
| `Intancias/4maquinas/30jobs/40.dat` | 20 / 19 | 4 | 478 | 481 | 477 |
| `Intancias/10maquinas/30jobs/40.dat` | 30 / 29 | 10 | 341 | 338 | 334 |

These are given to check an installation, not as results of the paper. Note
that the file in `4maquinas/30jobs/` has 20 inbound jobs, not 30.

## License

MIT, see `LICENSE`.

## How to cite

Nogueira, T. H., Coutinho, F. P., Peixoto, M. G. M., Carrano, E. G., & Ravetti,
M. G. (2025). Evolutionary algorithm framework for optimizing truck scheduling
in multi-dock truck cross-docking centers. *Evolutionary Intelligence*, 18(1),
2. https://doi.org/10.1007/s12065-024-00992-x

```bibtex
@article{nogueira2025evolutionary,
  author  = {Nogueira, Thiago Henrique and Coutinho, Felipe Provezano and Peixoto, Maria Gabriela Mendon{\c{c}}a and Carrano, Eduardo Gontijo and Ravetti, Mart{\'\i}n G{\'o}mez},
  title   = {Evolutionary algorithm framework for optimizing truck scheduling in multi-dock truck cross-docking centers},
  journal = {Evolutionary Intelligence},
  volume  = {18},
  number  = {1},
  pages   = {2},
  year    = {2025},
  doi     = {10.1007/s12065-024-00992-x}
}
```

## Related work

Nogueira, T. H., Coutinho, F. P., Ribeiro, R. P., & Ravetti, M. G. (2020).
Parallel-machine scheduling methodology for a multi-dock truck sequencing
problem in a cross-docking center. *Computers & Industrial Engineering*, 143,
106391. https://doi.org/10.1016/j.cie.2020.106391
