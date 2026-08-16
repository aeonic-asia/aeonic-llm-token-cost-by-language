"""The adjacent-pair colour gate, enforced instead of remembered.

WHY THIS EXISTS. The dollar figures sort bars by COST, so which fills end up
next to each other is decided by the *data*, not by the palette. On 2026-08-16
Anthropic repriced Sonnet 5 from $3.00 to $2.00; that lifted it from
8th-cheapest to 5th, put it beside GPT-5.6, and created a pair measuring normal
OKLab ΔE 14.1 against a floor of 15 — a real accessibility defect introduced by
a *vendor price change*, with no colour edited and nothing watching. Re-running
the check by hand had already been recorded as "outstanding" once before, which
is exactly the failure mode a test removes.

WHAT IT CHECKS. For every pair of bars ADJACENT in the rendered dollar order:

  * cross-family pair (different tokenizers, so different hues) — the
    CATEGORICAL gate: normal-vision ΔE×100 >= 15 and >= 8 under protanopia and
    deuteranopia simulation. Hue is carrying identity here, so the pair must be
    tellable apart.
  * same-family pair (two tint steps of one hue) — the categorical gate does NOT
    apply and asserting it would be wrong: two steps of one hue *are* one
    tokenizer, and their similarity is the message the encoding exists to send.
    The governing rule is ordinal instead — monotone lightness, adjacent OKLCH
    ΔL >= 0.06 — checked in `test_tint_ramps_are_ordinal`.

PROVENANCE OF THE NUMBERS. The gate was originally run with the validator
bundled in Claude Code's `dataviz` Agent Skill, which is not vendored here. The
implementation below was calibrated against 11 pairs measured with that tool and
reproduces them exactly: normal-vision ΔE on all 11, and protan/deutan ΔE on all
11 to within 0.05 (i.e. to the 1-decimal precision the tool prints). The CVD
model is Machado, Oliveira & Fernandes (2009) at severity 1.0, applied to LINEAR
sRGB and clamped to gamut before conversion — Viénot-style LMS simulation was off
by up to 4.6 and applying Machado in gamma space by up to 8.4, so those are not
interchangeable. Tritanopia is deliberately NOT gated: it is reported by the
original tool but never gated on, and this implementation agrees with it on 10 of
11 pairs rather than all 11, so gating it would be asserting more than is backed.
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from .. import config
from .. import figures

# ── gate thresholds (see module docstring) ───────────────────────────────────
MIN_NORMAL_DE = 15.0
MIN_CVD_DE = 8.0
MIN_RAMP_DL = 0.06

# Machado, Oliveira & Fernandes (2009), severity 1.0, for LINEAR sRGB.
_CVD = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
}


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hex_to_linear(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(_srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))


def _linear_to_oklab(rgb) -> tuple[float, float, float]:
    r, g, b = rgb
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    cbrt = lambda v: v ** (1 / 3) if v > 0 else -((-v) ** (1 / 3))
    l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _simulate(rgb, kind: str):
    m = _CVD[kind]
    out = [sum(m[i][j] * rgb[j] for j in range(3)) for i in range(3)]
    return [min(1.0, max(0.0, c)) for c in out]


def delta_e(hex_a: str, hex_b: str, kind: str | None = None) -> float:
    """OKLab ΔE×100 between two hexes, optionally under a CVD simulation."""
    ra, rb = _hex_to_linear(hex_a), _hex_to_linear(hex_b)
    if kind:
        ra, rb = _simulate(ra, kind), _simulate(rb, kind)
    return 100 * math.dist(_linear_to_oklab(ra), _linear_to_oklab(rb))


def oklch_lightness(hex_c: str) -> float:
    return _linear_to_oklab(_hex_to_linear(hex_c))[0]


def _rendered_dollar_order() -> list[str]:
    """The counter ids the dollar figures actually draw, in drawn order.

    Reuses figures._priced_order so the test tracks whatever the chart does,
    rather than re-deriving the ordering rule and drifting from it.
    """
    cost = pd.read_csv(config.RESULTS_DIR / "cost_by_language.csv")
    order = figures._priced_order(cost)
    drawn = set(cost.counter_id)
    return [c for c in order if c in drawn]


class PaletteAdjacencyTest(unittest.TestCase):

    def test_calibration_reproduces_the_reference_validator(self):
        """Guard the guard: if this drifts, every number below is meaningless."""
        # (a, b, expected_normal, expected_worst_cvd)
        reference = [
            ("#e34948", "#eda100", 20.8, 15.3),
            ("#4a3aa7", "#fe7a73", 39.3, 29.2),
            ("#008300", "#eda100", 30.3, 16.2),
            ("#6c62d2", "#36a231", 33.3, 27.8),
            ("#008300", "#6c62d2", 32.9, 26.5),
            ("#eda100", "#4a3aa7", 45.9, 41.0),
            ("#59c253", "#36a231", 10.0, 10.0),
            ("#36a231", "#008300", 9.9, 9.9),
            ("#fe7a73", "#e34948", 11.3, 10.7),
            ("#e34948", "#b51221", 12.9, 12.9),
            ("#6c62d2", "#4a3aa7", 13.0, 11.9),
        ]
        for a, b, exp_normal, exp_cvd in reference:
            self.assertAlmostEqual(delta_e(a, b), exp_normal, delta=0.05,
                                   msg=f"normal ΔE drifted for {a}/{b}")
            worst = min(delta_e(a, b, k) for k in _CVD)
            self.assertAlmostEqual(worst, exp_cvd, delta=0.05,
                                   msg=f"CVD ΔE drifted for {a}/{b}")

    def test_adjacent_bars_in_the_dollar_figures_are_distinguishable(self):
        """The gate a vendor price change can break without touching a colour."""
        order = _rendered_dollar_order()
        fills = figures._series_style(order)
        slots = [figures._slot(c) for c in order]

        failures = []
        for i in range(len(order) - 1):
            if slots[i] == slots[i + 1]:
                continue  # same tokenizer family — ordinal rule applies instead
            normal = delta_e(fills[i], fills[i + 1])
            cvd = min(delta_e(fills[i], fills[i + 1], k) for k in _CVD)
            if normal < MIN_NORMAL_DE or cvd < MIN_CVD_DE:
                failures.append(
                    f"{order[i]} ({fills[i]}) next to {order[i+1]} ({fills[i+1]}): "
                    f"normal ΔE {normal:.1f} (need >={MIN_NORMAL_DE}), "
                    f"CVD ΔE {cvd:.1f} (need >={MIN_CVD_DE})")
        self.assertEqual(failures, [], "\n  ".join(
            ["adjacent bars are too close to tell apart. NOTE the rendered order "
             "follows COST, so a vendor price change can cause this with no colour "
             "edited — check config.PRICING before assuming a palette bug. Fix by "
             "re-stepping the offending tint in figures._TINT_BY_COUNTER, or by "
             "re-slotting a hue if the ramp has no room left."] + failures))

    def test_tint_ramps_are_ordinal(self):
        """Within a family the rule is monotone lightness, not colour distance."""
        families: dict[int, list[str]] = {}
        for c in config.MODEL_MATRIX:
            if c.id in figures._TINT_BY_COUNTER or not c.headline:
                if c.id in config.PRICING and config.PRICING[c.id].input_usd_per_mtok:
                    families.setdefault(figures._slot(c.id), []).append(c.id)
        for cid in figures._SLOT_BY_FLAGSHIP:
            if config.PRICING.get(cid) and config.PRICING[cid].input_usd_per_mtok:
                families.setdefault(figures._slot(cid), []).append(cid)

        for slot, members in families.items():
            if len(members) < 2:
                continue
            # lighter = cheaper, so sort by price and expect descending lightness
            ordered = sorted(members,
                             key=lambda c: config.PRICING[c].input_usd_per_mtok)
            fills = [figures._TINT_BY_COUNTER.get(c, figures._SERIES[slot])
                     for c in ordered]
            lightness = [oklch_lightness(f) for f in fills]
            for a, b, la, lb in zip(ordered, ordered[1:], lightness, lightness[1:]):
                self.assertGreater(
                    la, lb,
                    f"{a} is priced below {b} but is not lighter — the ramp encodes "
                    "price, so lighter must mean cheaper")
                self.assertGreaterEqual(
                    la - lb, MIN_RAMP_DL,
                    f"{a} -> {b} steps only ΔL {la - lb:.4f}; the ramp needs "
                    f">={MIN_RAMP_DL} to read as ordered")


if __name__ == "__main__":
    unittest.main()
