"""Tests for the 2026 fixes (see CHANGES.md). Run: python tests/test_fixes.py"""
import contextlib
import io
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_gacdc  # noqa: E402
from Chromosome import Chromosome  # noqa: E402
from Population import Population  # noqa: E402
from Utils import Utils  # noqa: E402
from UtilsAE import UtilsAE  # noqa: E402

SMALL = os.path.join(ROOT, "Intancias", "2maquinas", "20jobs", "58.dat")
MID = os.path.join(ROOT, "Intancias", "4maquinas", "30jobs", "40.dat")


def population(n=30):
    random.seed(0)
    _, _, m1, m2, a, b = Utils.read_from_file(SMALL)
    return Population(m1, m2, a, b, n)


class FixTests(unittest.TestCase):
    def test_inversion_mutation_runs(self):
        p = population()
        for _ in range(200):
            c = UtilsAE.inversion_mutation(p.population[5])
            ids = sorted(j.id for j in c.jobs_list_1)
            self.assertEqual(ids, list(range(len(ids))))

    def test_tournament_indices_cover_population(self):
        p = population(30)
        chosen, dying = set(), set()
        for _ in range(500):
            chosen |= {c.id for c in UtilsAE.tournament_selection(p.population, 7, 2)}
            dying |= set(UtilsAE.tournament_selection_survive(p.population, 5, 2))
        self.assertGreater(len(chosen), 2)
        self.assertGreater(len(dying), 2)
        self.assertTrue(dying <= set(range(30)))

    def test_iter_is_separate_from_id(self):
        c = population().population[5]
        c.iter = 42
        self.assertEqual((c.id, c.iter), (5, 42))

    def test_stats_updated_every_generation(self):
        p = population()
        p.compute_stats(0)
        p.population[1] = UtilsAE.mutation_partition(p.population[0])
        p.population[1].makespan_2 += 10 ** 6
        before = p.mean_fitness_2
        p.compute_stats(1)
        self.assertNotEqual(before, p.mean_fitness_2)

    def test_1000_generations_complete(self):
        for seed in (3, 4, 7):
            args = run_gacdc.parse_args([MID, "--seed", str(seed), "--quiet"])
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_gacdc.run(args)
            self.assertGreater(res[0][1], 0)


if __name__ == "__main__":
    unittest.main()
