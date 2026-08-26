"""The figure filename grammar, and the prune that has to parse it back.

WHY THIS EXISTS. `_save` builds every stem as `<figure>-<corpus><locale><theme>`,
and `_prune_stale` parses that back to recover the corpus so it can delete
figures for a corpus the dataset no longer covers. The parse is the fragile
half: it strips known suffixes off the end, so every new render axis is a new
way for it to mis-parse. That has already happened once — before the theme
suffix was stripped, every dark figure read as corpus "<corpus>-dark", matched
nothing in `live`, and was deleted by the very run that had just written it.
Adding the locale axis creates the identical trap one layer further out, and it
is silent: the run still exits 0, having removed half its own output.

WHAT IT CHECKS. That a full, current locale x theme x corpus set survives a
prune untouched, and that a retired corpus is removed in every locale and theme
rather than in the default one only.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eval import config, figures


def _stems() -> list[str]:
    """Every stem `_render_all` would write, for every corpus/locale/theme."""
    figs = ("fig-premium-heatmap", "fig-cost-driver-bars", "fig-dollar-cost",
            "fig-dollar-cost-per-sentence", "fig-vietnamese-cost-ladder",
            "fig-vietnamese-tax-gap")
    return [f"{f}-{corpus}{loc.suffix}{theme.suffix}"
            for f in figs for corpus in config.CORPORA
            for loc in figures.LOCALES for theme in figures.THEMES]


class PruneParsesEveryStem(unittest.TestCase):
    def _prune(self, live: set[str]) -> set[str]:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            for stem in _stems():
                for ext in ("svg", "png"):
                    (d / f"{stem}.{ext}").write_text("x")
            with patch.object(figures, "FIG_DIR", d):
                figures._prune_stale(live)
            return {p.stem for p in d.iterdir()}

    def test_current_corpora_all_survive(self):
        kept = self._prune(set(config.CORPORA))
        self.assertEqual(kept, set(_stems()),
                         "a figure the run had just written was pruned — the stem "
                         "parse lost a suffix")

    def test_retired_corpus_goes_in_every_locale_and_theme(self):
        live = {"flores"}
        kept = self._prune(live)
        expected = {s for s in _stems() if "-massive" in s} & kept
        self.assertEqual(expected, set(),
                         "a retired corpus survived in some locale/theme — the "
                         "prune only reached the default variant")
        self.assertTrue(kept, "the prune removed everything, including live corpora")


if __name__ == "__main__":
    unittest.main()
