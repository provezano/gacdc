# Changes

## 2026

The original 2018 code is at tag `v2018-original`.

- Added `run_gacdc.py` (command-line runner with seed and GA parameters),
  tests, `requirements.txt` and `CITATION.cff`.
- Inversion mutation no longer raises `TypeError` (`len(list - 1)`).
- Tournament selection now samples `total_chromosomes_tournament` individuals
  and returns population indices; before, only individuals 0 and 1 were ever
  selected or replaced.
- `Chromosome.iter` has its own setter; before, it returned the chromosome id.
- Population mean and standard deviation are updated every generation.
- `GACDC.main` takes `--data-dir`, `--results-dir`, `--crossover-rate` and
  `--mutation-rate`, creates the results folders and skips missing instances;
  the hard-coded path is gone.
- Summary files: execution-time columns are tab-separated and correctly
  labelled (min/max were swapped; the std column repeated the mean).
- Removed `__MACOSX/` and `.DS_Store` entries from `Instancias.zip`, and the
  `Resultados/` files, which came from an older output format.

## Open question

Crossover keeps the child with the larger makespan
(`best_c = c1 if c1.makespan_2 > c2.makespan_2 else c2` in `GACDC.py`). It
may be intended for diversity; it is left as in the original.
