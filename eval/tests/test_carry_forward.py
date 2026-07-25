"""Carry-forward must never lose a committed measurement.

This is the eval's load-bearing invariant and the one that keeps breaking. Three
separate reviews have found three distinct ways to silently drop committed rows:
a keyless default pass rewriting the dataset without the key-gated counters; a
selected-but-errored counter having its rows deleted with nothing to replace
them; and carry-forward being gated on the *joint* existence of the two CSVs, so
one missing file discarded the other. Each was a one-line condition, each was
invisible until `git diff`, and each would have shipped wrong numbers.

The oracle test guards the measurement core. This guards the plumbing around it —
the part more likely to be wrong, because nothing about a dropped row looks wrong
in the output.

Run:  python -m unittest eval.tests.test_carry_forward   (from the repo root)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from eval import config, run as run_mod
from eval.measure import TokenCounter


class _FakeCounter(TokenCounter):
    """Counts characters. Deterministic, offline, and distinguishable per counter."""

    def __init__(self, spec: config.Counter, factor: int = 1):
        self.spec = spec
        self._factor = factor

    def count(self, text: str) -> int:
        return len(text) * self._factor


_CORPUS = {"eng_Latn": ["one", "two"], "vie_Latn": ["mot", "hai"]}


class CarryForwardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results = Path(self._tmp.name)
        self._saved = {k: getattr(config, k) for k in
                       ("RESULTS_DIR", "MODEL_MATRIX", "MATRIX_BY_ID",
                        "CORPORA", "LANGUAGES", "API_KINDS")}
        config.RESULTS_DIR = self.results
        config.CORPORA = {"flores": "FLORES+"}
        config.LANGUAGES = {"eng_Latn": "English", "vie_Latn": "Vietnamese"}
        config.API_KINDS = set()
        self.alpha = config.Counter("alpha", "Alpha", "X", "tiktoken", "live", spec="a")
        self.beta = config.Counter("beta", "Beta", "X", "tiktoken", "live", spec="b")
        config.MODEL_MATRIX = [self.alpha, self.beta]
        config.MATRIX_BY_ID = {c.id: c for c in config.MODEL_MATRIX}
        run_mod.load_corpus = lambda cid, langs: {l: _CORPUS[l] for l in langs}
        self._built = {"alpha": 1, "beta": 2}
        self._fail: set[str] = set()
        run_mod.build_counter = self._build

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)
        self._tmp.cleanup()

    def _build(self, spec):
        if spec.id in self._fail:
            return None, "unavailable in this test"
        return _FakeCounter(spec, self._built[spec.id]), ""

    def _agg(self):
        return pd.read_csv(self.results / "aggregate_counts.csv")

    # ── the invariant ────────────────────────────────────────────────────────
    def test_unavailable_counter_keeps_its_committed_rows(self):
        """A counter that cannot run must be carried, not erased.

        This is the keyless-`make reproduce` case: on a fresh clone with no API
        key, the Claude counters skip for cause. If the pass writes only what it
        measured, the committed dataset loses six counters and the run still
        exits 0.
        """
        run_mod.run()
        self.assertEqual(set(self._agg().counter_id), {"alpha", "beta"})
        before = self._agg()

        self._fail = {"beta"}          # e.g. ANTHROPIC_API_KEY now unset
        run_mod.run()
        after = self._agg()
        self.assertEqual(set(after.counter_id), {"alpha", "beta"},
                         "a skipped counter's committed rows were dropped")
        pd.testing.assert_frame_equal(
            after[after.counter_id == "beta"].reset_index(drop=True),
            before[before.counter_id == "beta"].reset_index(drop=True),
            obj="carried rows must be preserved verbatim")

    def test_errored_counter_keeps_its_committed_rows(self):
        """A failure part-way through measuring must roll back to committed rows.

        The realistic case is a transient 429 at sentence 1,900 of 2,000: the
        counter has produced thousands of good rows and then dies. It used to end
        up with neither those nor its committed ones.
        """
        run_mod.run()
        before = self._agg()

        class _Flaky(_FakeCounter):
            calls = 0

            def count(self, text):
                _Flaky.calls += 1
                if _Flaky.calls > 2:          # succeed, then fail mid-counter
                    raise RuntimeError("simulated 429")
                return len(text)

        def _build(spec):
            if spec.id == "beta":
                return _Flaky(spec), ""
            return _FakeCounter(spec, self._built[spec.id]), ""
        run_mod.build_counter = _build
        run_mod.run()
        pd.testing.assert_frame_equal(
            self._agg()[lambda d: d.counter_id == "beta"].reset_index(drop=True),
            before[before.counter_id == "beta"].reset_index(drop=True),
            obj="an errored counter must fall back to its committed rows")

    def test_a_counter_that_fails_to_construct_does_not_kill_the_pass(self):
        """build_counter is contracted to return (None, reason); if it raises
        anyway, that must cost one counter — not the manifest and both CSVs."""
        run_mod.run()
        before = self._agg()

        def _boom(spec):
            if spec.id == "beta":
                raise RuntimeError("simulated construction failure")
            return _FakeCounter(spec, self._built[spec.id]), ""
        run_mod.build_counter = _boom
        run_mod.run()
        self.assertEqual(set(self._agg().counter_id), {"alpha", "beta"})
        pd.testing.assert_frame_equal(
            self._agg()[lambda d: d.counter_id == "beta"].reset_index(drop=True),
            before[before.counter_id == "beta"].reset_index(drop=True))

    def test_one_missing_csv_does_not_discard_the_other(self):
        """Carry-forward is per file, not gated on the pair existing."""
        run_mod.run()
        before = self._agg()
        (self.results / "raw_counts.csv").unlink()

        self._fail = {"beta"}
        run_mod.run()
        self.assertEqual(set(self._agg().counter_id), {"alpha", "beta"},
                         "a missing raw_counts.csv erased the aggregate's rows")
        pd.testing.assert_frame_equal(
            self._agg()[lambda d: d.counter_id == "beta"].reset_index(drop=True),
            before[before.counter_id == "beta"].reset_index(drop=True))

    def test_refuses_to_write_an_empty_dataset(self):
        """Every counter unavailable and nothing to carry → abort, don't truncate."""
        self._fail = {"alpha", "beta"}
        with self.assertRaises(SystemExit):
            run_mod.run()

    def test_empty_corpus_aborts_before_overwriting(self):
        """A truncated corpus must not be measured as zeros."""
        run_mod.run()
        run_mod.load_corpus = lambda cid, langs: {l: [] for l in langs}
        with self.assertRaises(SystemExit):
            run_mod.run()
        self.assertEqual(len(self._agg()), 4, "zeros overwrote real measurements")

    def test_carrying_a_row_measured_on_a_different_spec_is_refused(self):
        """A column must be measured on the model it names.

        Repointing a counter's spec and then running an unrelated selective pass
        would otherwise republish rows measured on the old endpoint under the new
        model's name, with the figure caption asserting it was verified.
        """
        run_mod.run()
        config.MODEL_MATRIX = [self.alpha, replace(self.beta, spec="b-v2")]
        config.MATRIX_BY_ID = {c.id: c for c in config.MODEL_MATRIX}
        self._fail = {"beta"}
        with self.assertRaises(SystemExit):
            run_mod.run()

    def test_rows_are_written_in_canonical_order(self):
        """Byte-identity must not depend on which counters a pass re-measured."""
        run_mod.run()
        full = (self.results / "aggregate_counts.csv").read_bytes()
        import os
        os.environ["EVAL_COUNTERS"] = "beta"
        try:
            run_mod.run()
        finally:
            del os.environ["EVAL_COUNTERS"]
        self.assertEqual(full, (self.results / "aggregate_counts.csv").read_bytes(),
                         "a selective pass reordered rows relative to a full rebuild")

    def test_manifest_records_every_counter_in_the_dataset(self):
        """`measured_on_by_counter` must cover carried counters, not just measured ones."""
        run_mod.run()
        self._fail = {"beta"}
        run_mod.run()
        m = json.loads((self.results / "run_manifest.json").read_text())
        self.assertEqual(set(m["measured_on_by_counter"]), {"alpha", "beta"})
        self.assertIn("beta", m["counters_carried_forward"])


if __name__ == "__main__":
    unittest.main()
