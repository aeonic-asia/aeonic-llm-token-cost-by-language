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
    """Every stem `_render_all` would write, for every corpus/locale/theme.

    Both halves are taken from the module under test rather than restated here.
    The figure names come from `FIGURE_STEMS` — the same tuple `_prune_stale`
    matches on, so adding a figure to one and not the other cannot pass. The
    suffix order comes from `_rendered_stem`, the single statement of the
    grammar, so a test built on a private copy of the rule cannot keep passing
    while `_save` changes underneath it.
    """
    out = []
    for pre in figures.FIGURE_STEMS:
        for corpus in config.CORPORA:
            for loc in figures.LOCALES:
                for theme in figures.THEMES:
                    with figures._use_locale(loc), figures._use_theme(theme):
                        out.append(figures._rendered_stem(f"{pre}{corpus}"))
    return out


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


    def test_stem_grammar_is_corpus_then_locale_then_theme(self):
        """Pins the suffix ORDER itself.

        `_stems()` now derives from `_rendered_stem`, which makes the two prune
        tests agree with `_save` by construction — and therefore blind to the
        order changing. This is the assertion that is not blind: `_prune_stale`
        strips themes before locales, so it only parses names built the other
        way round.
        """
        with figures._use_locale(figures.VI), figures._use_theme(figures.DARK):
            self.assertEqual(figures._rendered_stem("fig-premium-heatmap-flores"),
                             "fig-premium-heatmap-flores-vi-dark",
                             "locale must precede theme — _prune_stale strips "
                             "themes first and would misread the corpus")


if __name__ == "__main__":
    unittest.main()
