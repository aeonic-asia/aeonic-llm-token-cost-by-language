"""Every "shared tokenizer" claim must survive the API message envelope.

An API counter's total carries one fixed frame that the offline tokenizer it runs
on does not, so raw totals differ by that frame alone. `content_totals` strips it;
drop that call and the GPT-6 ≡ o200k_base proof silently flips to "not shared" in
summary.json while every other gate stays green. The caption filter is pinned too:
unpriced offline proofs are evidence, not products, and must not be named on a
figure.

Pinned against the committed dataset: these are the claims the article cites.

Run:  python -m unittest eval.tests.test_shared_tokenizer   (from the repo root)
"""
from __future__ import annotations

import unittest

import pandas as pd

from eval import analyze, config, figures


class SharedTokenizerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")

    def test_api_model_matches_its_offline_tokenizer_once_the_frame_is_stripped(self):
        shared = analyze._coincidence_check(self.agg)["shared_in_all_corpora"]
        self.assertIn(["gpt", "o200k_base"], shared)
        for corpus in ("flores", "massive"):
            self.assertTrue(figures._identical(self.agg, corpus, "gpt", "o200k_base"))

    def test_captions_name_sold_models_only(self):
        for corpus in ("flores", "massive"):
            lines = " ".join(figures._legend_lines(
                self.agg, corpus, ["gpt", "gemini-3-8-flash"]))
            self.assertIn(figures._label("gpt-6-astra"), lines)
            for proof in ("o200k_base", "gemma4"):
                self.assertNotIn(figures._label(proof), lines)


if __name__ == "__main__":
    unittest.main()
