"""A hedged price must say so on the figure, not only in the CSV.

`Price.confidence` grades how well-sourced a price is. It reached
`cost_by_language.price_confidence` from the start, but never a figure — so a bar
computed from a `medium` price rendered under a caption reading only "input list
price", carrying exactly the authority of a fully-sourced one. The hedge was
visible solely to a reader who opened the CSV, which is not who reads a chart.

Every priced row is `high` as of 2026-07-27, so the qualifier renders nothing
today. That is precisely why it needs a test: a guard that is inert against the
current data will not be exercised by running the pipeline, and would rot
silently until the day it mattered.

Run:  python -m unittest eval.tests.test_price_confidence   (from the repo root)
"""
from __future__ import annotations

import unittest
from unittest import mock

from eval import config, figures


class PriceConfidenceNoteTest(unittest.TestCase):

    def test_all_high_renders_no_qualifier(self):
        """The committed dataset must not carry a hedge line."""
        priced = [cid for cid, p in config.PRICING.items()
                  if p.input_usd_per_mtok is not None]
        self.assertEqual(figures._price_confidence_note(priced), [])

    def test_committed_prices_are_all_high(self):
        """Pins the 2026-07-27 re-ratification: o200k was the last `medium`.

        Not redundant with the test above — that one asserts the *renderer* is
        quiet, this one asserts *why*. If a future price is added as `medium`,
        this fails with the offender named while the renderer test starts
        failing for the opposite reason, and the two together say which changed.
        """
        hedged = {cid: p.confidence for cid, p in config.PRICING.items()
                  if p.input_usd_per_mtok is not None and p.confidence != "high"}
        self.assertEqual(hedged, {})

    def test_hedged_price_reaches_the_caption_by_name(self):
        """The whole point: a non-high price must surface, and must be named."""
        hedged = dict(config.PRICING)
        hedged["o200k_base"] = config.Price(5.00, "2026-07-27", "medium",
                                            "scripted for test")
        with mock.patch.object(config, "PRICING", hedged):
            lines = figures._price_confidence_note(["o200k_base", "claude-new"])
        self.assertEqual(len(lines), 1)
        self.assertIn("medium", lines[0])
        # Named, not counted — a reader needs to know WHICH bar to distrust.
        self.assertIn(figures._label("o200k_base"), lines[0])
        self.assertNotIn(figures._label("claude-new"), lines[0])

    def test_unpriced_rows_never_trigger_the_note(self):
        """`unknown` on an unpriced row grades nothing — it draws no bar.

        cl100k, Llama 4, Qwen 3.6 and Opus 4.8 all carry `unknown` by design. If
        the note keyed on confidence alone it would fire permanently, and a
        permanently-present warning is one nobody reads.
        """
        unpriced = [cid for cid, p in config.PRICING.items()
                    if p.input_usd_per_mtok is None]
        self.assertTrue(unpriced, "fixture assumes some rows are unpriced")
        self.assertEqual(figures._price_confidence_note(unpriced), [])


if __name__ == "__main__":
    unittest.main()
