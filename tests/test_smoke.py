"""Smoke test: one small instance, a few generations, fixed seed.

Run from the repository root:  python -m unittest tests.test_smoke
"""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_gacdc  # noqa: E402

INSTANCE = os.path.join(ROOT, "Intancias", "2maquinas", "20jobs", "58.dat")


class SmokeTest(unittest.TestCase):
    def test_small_run(self):
        args = run_gacdc.parse_args([INSTANCE, "--seed", "3", "--pop-size", "20",
                                     "--generations", "20", "--quiet"])
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_gacdc.run(args)
        ms1, ms2 = res[0][0], res[0][1]
        # Total outbound processing time on 2 machines bounds the makespan from below.
        self.assertGreater(ms1, 0)
        self.assertGreater(ms2, ms1 / 2)

    def test_seed_is_reproducible(self):
        args = run_gacdc.parse_args([INSTANCE, "--seed", "3", "--pop-size", "20",
                                     "--generations", "20", "--quiet"])
        with contextlib.redirect_stdout(io.StringIO()):
            a = run_gacdc.run(args)[0][:2]
            b = run_gacdc.run(args)[0][:2]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
