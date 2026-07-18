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

# Paper Table 1, cl100k_base column. Values are reported to 2 d.p.; allow a
# small tolerance for corpus-version / concatenation differences.
PAPER_CL100K = {"vie_Latn": 2.45, "zho_Hans": 1.91, "deu_Latn": 1.58}
TOL = 0.03


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
