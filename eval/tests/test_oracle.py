"""Validation oracle: reproduce the paper's cl100k_base premiums.

Petrov et al. (arXiv:2305.15425), Table 1 — cl100k_base (ChatGPT/GPT-4)
premiums vs. English on FLORES-200:  Vietnamese 2.45, Chinese (Simp.) 1.91,
German 1.58.  Those are computed offline from the *same committed corpus* this
eval uses, so they are a hard correctness target: if our measurement core does
not reproduce them (to 2 d.p.), the pipeline is wrong.

Run:  python -m unittest eval.tests.test_oracle   (from the repo root)
"""
import unittest

from eval import config
from eval.corpora import load_flores
from eval.measure import TiktokenCounter

# Paper Table 1, cl100k_base column, and the tolerance — both read from config,
# NOT restated here. They used to be hardcoded in this file AND in analyze.py
# under two different key schemes (language code here, display name there), so
# the two implementations of the same gate could disagree without either failing:
# renaming a language silently dropped it from analyze's check while this test
# stayed green. One definition, two readers.
#
# Values are reported to 2 d.p., so the paper value carries ±0.005 of rounding;
# that rounding half-step is the tolerance — the precision the paper actually
# pins, and what the docs advertise. Observed deltas are all under it
# (max |Δ| ≈ 0.0043).
PAPER_CL100K = config.PAPER_CL100K_FLORES
TOL = config.ORACLE_TOL


def aggregate_premium(counter, lang: str, baseline: str) -> float:
    """Premium the paper's way: concatenate all sentences, count once."""
    n_lang = counter.count(" ".join(load_flores(lang)))
    n_base = counter.count(" ".join(load_flores(baseline)))
    return n_lang / n_base


class TestCl100kOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = TiktokenCounter(config.MATRIX_BY_ID["cl100k_base"])

    def test_reproduces_paper_premiums(self):
        for lang, expected in PAPER_CL100K.items():
            got = aggregate_premium(self.counter, lang, config.BASELINE_LANG)
            self.assertAlmostEqual(
                got, expected, delta=TOL,
                msg=f"{lang}: cl100k premium {got:.3f} vs paper {expected} "
                    f"(tol {TOL}) — measurement core disagrees with the oracle",
            )


if __name__ == "__main__":
    unittest.main()
