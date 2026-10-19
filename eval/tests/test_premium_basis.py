"""The published premium is the full-corpus ratio, and figures print it in-locale.

Two traps a citer hit in practice: the per-sentence sample size sat on the same
row as the full-corpus premium under a generic name and was read as its basis;
and the Vietnamese heatmap's colour bar printed English decimal marks.

Run:  python -m unittest eval.tests.test_premium_basis   (from the repo root)
"""
from __future__ import annotations

import unittest

import pandas as pd

from eval import config, figures


class PremiumBasisTest(unittest.TestCase):

    def test_premium_aggregate_is_the_aggregate_counts_ratio(self):
        agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")
        prem = pd.read_csv(config.RESULTS_DIR / "premium_by_language.csv")
        tot = agg.set_index(["corpus", "counter_id", "lang"])["total_tokens"]
        for r in prem.itertuples():
            want = tot[(r.corpus, r.counter_id, r.lang)] / tot[(r.corpus, r.counter_id,
                                                                 config.BASELINE_LANG)]
            self.assertAlmostEqual(r.premium_aggregate, round(want, 4), places=4,
                                   msg=f"{r.corpus}/{r.counter_id}/{r.lang}")

    def test_bases_are_named_apart(self):
        cols = set(pd.read_csv(config.RESULTS_DIR / "premium_by_language.csv").columns)
        self.assertIn("aggregate_basis", cols)
        self.assertIn("distribution_n", cols)
        self.assertNotIn("n_sentences", cols)

    def test_colour_bar_ticks_follow_the_locale(self):
        with figures._use_locale(figures.VI):
            self.assertEqual(figures._cbar_tick(0.5), "0,5")
        with figures._use_locale(figures.EN):
            self.assertEqual(figures._cbar_tick(1.5), "1.5")


if __name__ == "__main__":
    unittest.main()
