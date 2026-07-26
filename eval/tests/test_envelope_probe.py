"""The message-envelope frame must be derived, not assumed — and not over-refused.

`count_tokens` counts a rendered chat prompt, so every Claude count carries a
fixed turn/role frame. The frame is subtracted from per-sentence counts, which
means a frame wrong by one shifts an entire published premium distribution. It
is measured by probing single characters and subtracting one token.

The subtlety this suite pins is the direction of error. A single character can
never be worth FEWER than one token in a byte-level BPE, so a probe that happens
to be a multi-token character reports a frame one too HIGH and never one too low.
The floor across a diverse probe set is therefore the estimator, and a probe
above the floor is a fact about that character rather than evidence the frame is
wrong.

Requiring unanimity instead — which is what the code did until 2026-07-26 —
conflates the two and aborts on the first character class an endpoint tokenizes
differently. Measured that day: every digit costs two tokens on the older Claude
tokenizer, and uppercase 'Z' costs two on the newer one. Both real, both
harmless to the frame, and the unanimity rule blocked two counters on the first
credentialed pass while letting four others through only because the probe set
happened to miss their outlier.

Run:  python -m unittest eval.tests.test_envelope_probe   (from the repo root)
"""
from __future__ import annotations

import unittest

from eval import config
from eval.measure import AnthropicCounter


class _ScriptedCounter(AnthropicCounter):
    """AnthropicCounter with count() scripted from a table. No API, no key."""

    def __init__(self, counts: dict[str, int]):
        self.spec = config.Counter("scripted", "Scripted", "X", "anthropic", "live",
                                   spec="scripted-model")
        self._counts = counts
        self._client = None

    def count(self, text: str) -> int:
        return self._counts[text]


def _counts(frame: int, multi: tuple[str, ...] = ()) -> dict[str, int]:
    """Counts for every configured probe at `frame`, with `multi` worth 2 tokens."""
    probes = (config.ENVELOPE_PROBE, *config.ENVELOPE_PROBE_ALTS)
    return {p: frame + (2 if p in multi else 1) for p in probes}


class EnvelopeProbeTest(unittest.TestCase):
    def test_unanimous_probes_give_the_frame(self):
        self.assertEqual(_ScriptedCounter(_counts(6)).envelope_tokens(), 6)

    def test_a_multi_token_probe_does_not_change_the_frame(self):
        """The real 2026-07-26 failures, both directions.

        Older tokenizer: digits cost two tokens, frame 7.
        Newer tokenizer: uppercase 'Z' costs two tokens, frame 6.
        Under the old unanimity rule each of these aborted the counter and
        carried stale rows forward instead.
        """
        old = _ScriptedCounter(_counts(7, multi=("7", "0")))
        self.assertEqual(old.envelope_tokens(), 7)
        new = _ScriptedCounter(_counts(6, multi=("Z",)))
        self.assertEqual(new.envelope_tokens(), 6)

    def test_probe_evidence_is_retained_for_the_manifest(self):
        """Outliers are a measured property of the tokenizer, not scratch work."""
        c = _ScriptedCounter(_counts(7, multi=("7",)))
        c.envelope_tokens()
        self.assertEqual(c.envelope_probe_frames["x"], 7)
        self.assertEqual(c.envelope_probe_frames["7"], 8,
                         "a multi-token probe must be recorded at its real frame")

    def test_too_few_agreeing_probes_aborts(self):
        """The floor supported by almost nothing is not a frame.

        If nearly every character is multi-token on an endpoint, there is no
        one-token anchor to subtract and the method does not apply — that must
        abort rather than quietly return the floor.
        """
        probes = (config.ENVELOPE_PROBE, *config.ENVELOPE_PROBE_ALTS)
        multi = tuple(probes[1:])          # only the first probe is single-token
        with self.assertRaises(RuntimeError):
            _ScriptedCounter(_counts(6, multi=multi)).envelope_tokens()

    def test_implausible_frame_still_aborts(self):
        """A unanimous but absurd frame means the probe measured something else."""
        with self.assertRaises(RuntimeError):
            _ScriptedCounter(_counts(config.ENVELOPE_MAX_PLAUSIBLE + 1)).envelope_tokens()

    def test_probe_set_spans_classes_and_can_outvote_its_outliers(self):
        """Config guard: the set must be able to lose its known outliers.

        Both real outlier classes (digits, uppercase) are deliberately kept in
        the probe set so the outlier path stays exercised. That only works if
        enough probes remain to clear ENVELOPE_MIN_AGREEING without them.
        """
        probes = (config.ENVELOPE_PROBE, *config.ENVELOPE_PROBE_ALTS)
        self.assertEqual(len(set(probes)), len(probes), "duplicate probe")
        self.assertTrue(all(len(p) == 1 for p in probes), "probes must be single characters")
        known_outliers = {p for p in probes if p.isdigit() or p.isupper()}
        self.assertTrue(known_outliers, "no outlier class left to exercise the floor rule")
        survivors = len(probes) - len(known_outliers)
        self.assertGreaterEqual(
            survivors, config.ENVELOPE_MIN_AGREEING,
            f"only {survivors} probes remain once the known multi-token classes are "
            f"removed, below ENVELOPE_MIN_AGREEING={config.ENVELOPE_MIN_AGREEING}")


if __name__ == "__main__":
    unittest.main()
