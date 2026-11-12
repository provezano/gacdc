# Changes

## Added (no change to the algorithm)

- `run_gacdc.py`: command-line runner for one instance (path, seed, GA
  parameters). With `--seed`, it seeds `random` and `numpy.random` and skips
  the `random.seed(None)` call at the start of `evolutionary_algorithm`, which
  would otherwise discard the seed. Without `--seed` the behaviour is the
  original one.
- `requirements.txt`, `tests/test_smoke.py`, `CITATION.cff`, `.gitignore`,
  English `README.md`.

## Known issues (not fixed)

These were found by reading the code and, where noted, by running it. They are
listed here so that the owner can decide how to handle them.

1. **Inversion mutation always crashes** (`UtilsAE.inversion_mutation`):
   `len(new_chromosome.jobs_list_1-1)` subtracts 1 from a list and raises
   `TypeError`. It is one of the five mutations drawn at random, so a run with
   the default settings (1000 generations) stops with this error. Confirmed on
   `Intancias/4maquinas/30jobs/40.dat` with seeds 3, 4 and 7. The likely intent
   is `len(new_chromosome.jobs_list_1) - 1`.
2. **Tournament selection returns positions inside the tournament, not in the
   population** (`UtilsAE.tournament_selection` and
   `tournament_selection_survive`): `np.argmin`/`np.argmax` give 0 or 1, which
   are then used as population indices. Each sample also has
   `total_chromosomes` members instead of `total_chromosomes_tournament`.
   In a check of 200 calls, only individuals 0 and 1 were ever selected or
   replaced, so the search works on two slots of the population.
3. **`iter` property returns the chromosome id** (`Chromosome.py`): the setter
   is declared with `@id.setter`, so the `iter` property reuses the `id`
   getter. The "Iter. Best" column written by `main` is therefore the id, not
   the generation in which the best was found.
4. **Crossover keeps the worse child** (`GACDC.py`):
   `best_c = c1 if c1.makespan_2 > c2.makespan_2 else c2` picks the child with
   the larger makespan. This may be intended (diversity) but the name suggests
   otherwise.
5. **Population statistics go stale** (`Population.compute_stats`): mean and
   standard deviation are updated only when the best value changes, so the
   logged mean/std can stay fixed for many generations (the smoke run reports
   a std of 0).
6. **Command-line options in `GACDC.main` do not work**: the getopt string
   `"hcr:mr:"` does not define `-cr`/`-mr`, the long options are swapped, and
   the values are kept as strings. `main` also reads instances from a
   hard-coded absolute path and writes to `./results/...` folders that must
   already exist.
7. **Summary file labels** (`GACDC.main`): the min and max execution times are
   written under swapped labels, and the general results file writes the mean
   execution time twice (the second was probably meant to be the std).
