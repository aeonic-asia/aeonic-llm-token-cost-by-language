"""Figures for the article: premium heatmap + per-language cost-driver bars.

Exports SVG (for later fig-NN-*.svg article assets) and PNG (quick view) to
eval/results/figures/. Driven entirely by the committed result CSVs, so figures
regenerate deterministically from the dataset.

Readability: the full matrix carries proxy counters — four models on the newer
Claude tokenizer (Opus 5, Opus 4.8, Sonnet 5, Fable 5) and two on the older
(Sonnet 4.6, Haiku 4.5) — so the eval can *verify* equivalence rather than assert
it. The headline figures collapse those to **one column per distinct tokenizer**,
named by its flagship model (config `headline` / `flagship_group`), and print a
legend beneath spelling out the folds. Which counters fold is config; that a
"shared" fold really is byte-identical is re-verified here against the counts —
never asserted, in every caption that makes the claim.

**Two render axes beyond the data: theme and locale.** Every figure is emitted
once per (locale, theme) pair, with both suffixes appended to the stem in that
order — `fig-premium-heatmap-flores-vi-dark.svg`. English and light carry empty
suffixes, so adding an axis never renames an existing file. Localisation happens
here rather than by editing the SVGs because matplotlib draws every label as a
`<path>`: there is no text in the output to translate.

**Two column orders, not one.** The tokenizer figures (heatmap, cost-driver)
ascend by the lead language's premium via `_headline_order`; the two dollar
figures ascend by price via `_priced_order`. Both prefer FLORES+ for stability
across corpora, and neither drops a counter that the preferred corpus does not
rank — it sorts last instead, and `_check_drawn_coverage` raises if a measured
counter would go missing anyway.

Two dollar figures are emitted per corpus, differing only in denominator:
per-1,000,000-characters and per-1,000-sentences. The corpora are parallel, so
the per-sentence figure is cost for the same *meaning* across languages. They are
meant to be read together — a dense script (Chinese) towers per character yet
ranks low per sentence, because it says the same thing in far fewer characters.
The "cost driver" bars show tokens-per-1,000-NFC-characters — the measured,
price-independent driver of the per-character view.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
# Determinism: pin the SVG element-id hash salt (else matplotlib re-randomises
# clip-path / marker ids every run) so re-runs are byte-identical. The wall-clock
# <dc:date> is stripped per-save below. Together these keep `make reproduce` from
# churning the committed figures when only the timestamp/ids would differ.
matplotlib.rcParams["svg.hashsalt"] = "aeonic-token-cost-eval"
# Render "$5" literally. Matplotlib parses $...$ as LaTeX math, so a caption
# listing two prices ("Opus 5 $5 / Sonnet 5 $3") silently ate both dollar signs
# and italicised everything between them. Nothing here wants math, so turn the
# parser off globally rather than escaping each string and hoping the next
# caption remembers to.
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# ── design tokens ────────────────────────────────────────────────────────────
# Categorical hues assigned in FIXED slot order and never cycled (provenance and
# validation notes live with _SLOT_BY_FLAGSHIP below). Aqua and yellow fall below
# 3:1 against the surface, which obliges "relief" — the committed
# cost_by_language.csv / premium_by_language.csv are that table view, and the
# extreme in each group is directly labelled. (Magenta is also sub-3:1 but is
# currently unassigned; see _SLOT_BY_FLAGSHIP.)
# TWO THEMES. The repo's own project page is light; the Insights article that
# consumes these figures is dark. Both render from one code path — a Theme
# carries every colour that depends on the surface, and make_figures() emits the
# whole set once per theme.
#
# The dark theme changes the SURFACE and the INK, and deliberately keeps the
# series fills. That is not laziness — it is the finding that shaped this work.
# CVD separation is measured fill-against-fill, so it does not depend on the
# surface at all: reusing the fills inherits every colourblind-safety property
# the light palette already validated, for free. Recolouring the marks forfeits
# it. A first attempt did exactly that — it inverted the price ramps so the dear
# end stayed bright on black — and put Sonnet 5's red beside Gemini 3.1 Pro's
# green at deutan dE 1.0 in the RENDERED (cost-sorted) order: indistinguishable
# to a red-green colourblind reader, where the light palette scores 8.9. The
# slot-order check passed and hid it; only the rendered order caught it, which
# is the same lesson _SLOT_BY_FLAGSHIP already records.
#
# So exactly one family moves, and only because it had to: violet's dear end
# #4a3aa7 sits at 2.04:1 on the dark surface — over the 2:1 ordinal floor by a
# hair, and dark violet on near-black in practice. Lifting the pair at fixed hue
# and chroma puts it at 3.02:1. Everything else is the shipped light value, so
# `lighter = cheaper` still reads the same way in both themes.
#
# Two dark-theme numbers beat their light counterparts: muted ink goes 3.50:1 ->
# 4.90:1 (light misses the 4.5 WCAG text threshold; dark clears it) and the
# violet dear end 2.04 -> 3.02:1.
#
# Validated with the dataviz validator, dark surface #1a1a19: categorical set
# and all three ordinal ramps PASS; rendered-order cross-family CVD dE 8.9
# (protan), identical to light. The residual within-family FAILs are the
# documented expected ones (see Theme.tints).
@dataclass(frozen=True)
class Theme:
    name: str
    suffix: str          # appended to every figure stem ("" keeps light's filenames)
    surface: str         # painted on the figure + axes patch
    ink: str             # primary ink (titles)
    ink_2: str           # secondary ink (legend, axis titles)
    ink_muted: str       # muted ink (tick labels, captions)
    grid: str            # hairline gridline, one step off the surface
    baseline: str        # baseline / axis rule
    series: tuple[str, ...]      # categorical slots, fixed order, never cycled
    tints: dict[str, str]        # per-counter lightness steps within a family hue


# Categorical hues assigned in FIXED slot order and never cycled (provenance and
# validation notes live with _SLOT_BY_FLAGSHIP below). On the light surface aqua
# and yellow fall below 3:1, which obliges "relief" — the committed
# cost_by_language.csv / premium_by_language.csv are that table view, and the
# extreme in each group is directly labelled. (Magenta is also sub-3:1 but is
# currently unassigned; see _SLOT_BY_FLAGSHIP.)
_LIGHT_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948")

LIGHT = Theme(
    name="light", suffix="",
    # #ffffff, not #fcfcfb. The old _SURFACE constant said #fcfcfb but was never
    # read by anything, so every figure has always rendered on matplotlib's
    # default white while the documented contrast ratios were reasoned against a
    # surface that never shipped. Recording what actually renders.
    surface="#ffffff",
    ink="#0b0b0b", ink_2="#52514e", ink_muted="#898781",
    grid="#e1e0d9", baseline="#c3c2b7",
    series=_LIGHT_SERIES, tints={
        "claude-sonnet-5": "#f06d67",        # $2 — lightest of the newer-Claude family
        "claude-fable-5": "#b51221",         # $10 — darkest
        "claude-haiku-4-5": "#6c62d2",       # $1 — lighter of the older-Claude pair
        "gemini-3-5-flash": "#36a231",       # $1.50 — middle step of the green ramp
        "gemini-3-1-flash-lite": "#59c253",  # $0.25 — lightest; cheapest bar in the chart
    })

DARK = Theme(
    name="dark", suffix="-dark",
    surface="#1a1a19",
    ink="#f5f4f1", ink_2="#b8b6b0", ink_muted="#8a8880",
    grid="#2a2926", baseline="#3d3b37",
    # Slot 6 (violet) is the one lifted value; every other slot is light's.
    series=_LIGHT_SERIES[:6] + ("#6156c5",) + _LIGHT_SERIES[7:],
    tints={
        "claude-sonnet-5": "#f06d67",
        "claude-fable-5": "#b51221",
        "claude-haiku-4-5": "#857ef1",       # lifted with its family (was #6c62d2)
        "gemini-3-5-flash": "#36a231",
        "gemini-3-1-flash-lite": "#59c253",
    })

THEMES = (LIGHT, DARK)

# The theme in force for the current render. Module state rather than a
# parameter threaded through every helper: the render functions are already
# long, and a Theme is read in thirteen places across five of them.
_T: Theme = LIGHT


@contextmanager
def _use_theme(theme: Theme):
    """Render under `theme`, restoring the previous one afterwards."""
    global _T
    prev, _T = _T, theme
    try:
        yield theme
    finally:
        _T = prev


# ── locale ──────────────────────────────────────────────────────────────────
# A second render axis, built exactly like Theme above: one frozen record per
# locale, a module-level current value, and a context manager to swap it. The
# suffix lands in the filename the same way the theme's does, so English keeps
# its existing stems byte-for-byte and a locale is additive by construction.
#
# Why the figures are localised at SOURCE rather than by editing the SVGs: every
# label in a matplotlib SVG is a <path>, not text. There is no string in the file
# to replace, so a translated figure can only be produced by re-rendering.
#
# The catalogue owns whole sentences, never fragments joined at the call site.
# Vietnamese does not mark plurals with -s and orders its clauses differently, so
# a template like "USD per 1,000 {unit}s" cannot be assembled from a translated
# {unit} plus English glue — the glue is the part that has to move.
@dataclass(frozen=True)
class Locale:
    name: str
    suffix: str                 # appended to every figure stem ("" keeps English's filenames)
    languages: dict[str, str]   # display name per FLORES code
    units: dict[str, str]       # per-corpus counting unit (sentence / message)
    decimal: str                # decimal mark
    group: str                  # thousands separator
    money: str                  # currency template, e.g. "${v}" or "{v} USD"
    date: str                   # "iso" | "dmy"
    t: dict[str, str]           # message catalogue


EN = Locale(
    name="en", suffix="",
    languages=config.LANGUAGES,
    units={"flores": "sentence", "massive": "message"},
    decimal=".", group=",", money="${v}", date="iso",
    t={
        "ylabel_usd_per_1m_chars": "USD per 1,000,000 input characters",
        "ylabel_usd_per_1k_units": "USD per 1,000 {unit}s",
        "ylabel_tokens_per_1k_chars": "Tokens per 1,000 NFC characters",
        "xlabel_vi_ladder": "USD per 1,000 sentences of Vietnamese input",
        "xlabel_dumbbell": "USD per 1,000 sentences \u2014 same content, both languages",
        "title_heatmap": "Token premium vs. English ({corpus})",
        "title_cost_driver": "Cost driver: tokens per 1,000 characters \u2014 {corpus} "
                             "(lower = cheaper)",
        "title_dollar_chars": "Serving cost: USD per 1M input characters \u2014 {corpus} "
                              "(price \u00d7 tokens; lower = cheaper)",
        "title_dollar_units": "Serving cost: USD per 1,000 {unit}s \u2014 {corpus} "
                              "(price \u00d7 tokens per {unit}; lower = cheaper)",
        "title_vi_ladder": "What serving Vietnamese costs \u2014 {corpus} "
                           "({spread}\u00d7 between the cheapest and dearest tier)",
        "title_vi_tax": "The Vietnamese tax in money \u2014 {corpus} "
                        "(each line is what the same content costs extra in Vietnamese)",
        "cbar_premium": "\u00d7 English tokens (1.00 = parity)",
        "legend_baseline_lang": "English (same content)",
        "legend_lead_lang": "Vietnamese",
        "cap_shared_tokenizer": "{names} \u2014 one shared tokenizer, identical counts "
                                "(verified here)",
        "cap_prices_differ_named": "; prices differ ({cheapest} cheapest)",
        "cap_prices_differ": "; prices differ",
        "cap_cl100k": "{label} \u2014 GPT-4/3.5-era baseline, historical anchor",
        "cap_price_confidence": "Not every price is high-confidence \u2014 {named}. See "
                                "`source` in config.PRICING and `price_confidence` in "
                                "cost_by_language.csv",
        "cap_same_tokens": "Same tokens, different price: {parts} (per 1M input tokens)",
        "cap_unpriced": "{names} omitted \u2014 no serving list price",
        "cap_usd_per_1m_chars": "USD to serve 1,000,000 input characters \u2014 input list "
                                "price ({as_of}); VND = USD \u00d7 {rate}",
        "cap_usd_per_1k_units": "USD to serve 1,000 {unit}s \u2014 parallel corpus, so the "
                                "SAME content across languages (price \u00d7 tokens per "
                                "{unit}); input list price ({as_of}); VND = USD \u00d7 {rate}",
        "cap_read_against": "Read against the per-character chart: a dense script (Chinese) "
                            "needs few characters, so per-character overstates its cost; "
                            "per {unit} it ranks far lower.",
        "cap_dumbbell_gaps": "Widest gap: {widest}, +{widest_delta} per 1,000 sentences "
                             "({widest_ratio}\u00d7). Steepest ratio: {steepest} at "
                             "{steepest_ratio}\u00d7 \u2014 only +{steepest_delta}, because a "
                             "cheap model with an inefficient tokenizer is costly in "
                             "proportion and small in money.",
        "cap_dumbbell_axes": "The two are different axes: the tokenizer sets the ratio, the "
                             "serving tier sets the size of the bill it applies to.",
    })

# Vietnamese. Terminology is held identical to the published Vietnamese article
# so a reader moves between prose and figure without re-learning a word:
# b\u1ed9 t\u00e1ch token (tokenizer), b\u1ed9i s\u1ed1 token (premium), ng\u1eef v\u1ef1c (register).
VI = Locale(
    name="vi", suffix="-vi",
    languages={
        "eng_Latn": "Ti\u1ebfng Anh",
        "vie_Latn": "Ti\u1ebfng Vi\u1ec7t",
        "zho_Hans": "Ti\u1ebfng Trung (gi\u1ea3n th\u1ec3)",
        "rus_Cyrl": "Ti\u1ebfng Nga",
        "deu_Latn": "Ti\u1ebfng \u0110\u1ee9c",
    },
    units={"flores": "c\u00e2u", "massive": "tin nh\u1eafn"},
    decimal=",", group=".", money="{v} USD", date="dmy",
    t={
        "ylabel_usd_per_1m_chars": "USD tr\u00ean 1.000.000 k\u00fd t\u1ef1 \u0111\u1ea7u v\u00e0o",
        "ylabel_usd_per_1k_units": "USD tr\u00ean 1.000 {unit}",
        "ylabel_tokens_per_1k_chars": "Token tr\u00ean 1.000 k\u00fd t\u1ef1 NFC",
        "xlabel_vi_ladder": "USD tr\u00ean 1.000 c\u00e2u \u0111\u1ea7u v\u00e0o ti\u1ebfng Vi\u1ec7t",
        "xlabel_dumbbell": "USD tr\u00ean 1.000 c\u00e2u \u2014 c\u00f9ng m\u1ed9t n\u1ed9i dung, c\u1ea3 hai ng\u00f4n ng\u1eef",
        "title_heatmap": "B\u1ed9i s\u1ed1 token so v\u1edbi ti\u1ebfng Anh ({corpus})",
        "title_cost_driver": "Y\u1ebfu t\u1ed1 sinh chi ph\u00ed: token tr\u00ean 1.000 k\u00fd t\u1ef1 \u2014 "
                             "{corpus} (th\u1ea5p h\u01a1n = r\u1ebb h\u01a1n)",
        "title_dollar_chars": "Chi ph\u00ed ph\u1ee5c v\u1ee5: USD tr\u00ean 1 tri\u1ec7u k\u00fd t\u1ef1 \u0111\u1ea7u v\u00e0o \u2014 "
                              "{corpus} (gi\u00e1 \u00d7 s\u1ed1 token; th\u1ea5p h\u01a1n = r\u1ebb h\u01a1n)",
        "title_dollar_units": "Chi ph\u00ed ph\u1ee5c v\u1ee5: USD tr\u00ean 1.000 {unit} \u2014 {corpus} "
                              "(gi\u00e1 \u00d7 s\u1ed1 token m\u1ed7i {unit}; th\u1ea5p h\u01a1n = r\u1ebb h\u01a1n)",
        "title_vi_ladder": "Ph\u1ee5c v\u1ee5 ti\u1ebfng Vi\u1ec7t t\u1ed1n bao nhi\u00eau \u2014 {corpus} "
                           "(ch\u00eanh {spread}\u00d7 gi\u1eefa b\u1eadc r\u1ebb nh\u1ea5t v\u00e0 b\u1eadc \u0111\u1eaft nh\u1ea5t)",
        "title_vi_tax": "Thu\u1ebf ti\u1ebfng Vi\u1ec7t t\u00ednh b\u1eb1ng ti\u1ec1n \u2014 {corpus} "
                        "(m\u1ed7i \u0111o\u1ea1n n\u1ed1i l\u00e0 ph\u1ea7n c\u00f9ng m\u1ed9t n\u1ed9i dung t\u1ed1n th\u00eam khi vi\u1ebft "
                        "b\u1eb1ng ti\u1ebfng Vi\u1ec7t)",
        "cbar_premium": "\u00d7 s\u1ed1 token ti\u1ebfng Anh (1,00 = ngang b\u1eb1ng)",
        "legend_baseline_lang": "Ti\u1ebfng Anh (c\u00f9ng n\u1ed9i dung)",
        "legend_lead_lang": "Ti\u1ebfng Vi\u1ec7t",
        "cap_shared_tokenizer": "{names} \u2014 chung m\u1ed9t b\u1ed9 t\u00e1ch token, s\u1ed1 \u0111\u1ebfm gi\u1ed1ng h\u1ec7t "
                                "(\u0111\u00e3 ki\u1ec3m ch\u1ee9ng t\u1ea1i \u0111\u00e2y)",
        "cap_prices_differ_named": "; gi\u00e1 kh\u00e1c nhau ({cheapest} r\u1ebb nh\u1ea5t)",
        "cap_prices_differ": "; gi\u00e1 kh\u00e1c nhau",
        "cap_cl100k": "{label} \u2014 m\u1ed1c tham chi\u1ebfu th\u1eddi GPT-4/3.5, neo l\u1ecbch s\u1eed",
        "cap_price_confidence": "Kh\u00f4ng ph\u1ea3i m\u1ecdi m\u1ee9c gi\u00e1 \u0111\u1ec1u \u0111\u1ed9 tin c\u1eady cao \u2014 {named}. Xem "
                                "`source` trong config.PRICING v\u00e0 `price_confidence` trong "
                                "cost_by_language.csv",
        "cap_same_tokens": "C\u00f9ng s\u1ed1 token, kh\u00e1c gi\u00e1: {parts} (tr\u00ean 1 tri\u1ec7u token \u0111\u1ea7u v\u00e0o)",
        "cap_unpriced": "{names} \u0111\u01b0\u1ee3c b\u1ecf ra ngo\u00e0i \u2014 kh\u00f4ng c\u00f3 gi\u00e1 ni\u00eam y\u1ebft \u0111\u1ec3 ph\u1ee5c v\u1ee5",
        "cap_usd_per_1m_chars": "USD \u0111\u1ec3 ph\u1ee5c v\u1ee5 1.000.000 k\u00fd t\u1ef1 \u0111\u1ea7u v\u00e0o \u2014 gi\u00e1 ni\u00eam y\u1ebft "
                                "\u0111\u1ea7u v\u00e0o ({as_of}); VND = USD \u00d7 {rate}",
        "cap_usd_per_1k_units": "USD \u0111\u1ec3 ph\u1ee5c v\u1ee5 1.000 {unit} \u2014 kho ng\u1eef li\u1ec7u song song, n\u00ean l\u00e0 "
                                "C\u00d9NG M\u1ed8T n\u1ed9i dung tr\u00ean m\u1ecdi ng\u00f4n ng\u1eef (gi\u00e1 \u00d7 s\u1ed1 token m\u1ed7i "
                                "{unit}); gi\u00e1 ni\u00eam y\u1ebft \u0111\u1ea7u v\u00e0o ({as_of}); VND = USD \u00d7 {rate}",
        "cap_read_against": "\u0110\u1ecdc c\u00f9ng bi\u1ec3u \u0111\u1ed3 t\u00ednh theo k\u00fd t\u1ef1: m\u1ed9t h\u1ec7 ch\u1eef c\u00f4 \u0111\u1ecdng "
                            "(ti\u1ebfng Trung) c\u1ea7n \u00edt k\u00fd t\u1ef1, n\u00ean c\u00e1ch t\u00ednh theo k\u00fd t\u1ef1 th\u1ed5i ph\u1ed3ng "
                            "chi ph\u00ed c\u1ee7a n\u00f3; t\u00ednh theo {unit} th\u00ec n\u00f3 x\u1ebfp th\u1ea5p h\u01a1n h\u1eb3n.",
        "cap_dumbbell_gaps": "Kho\u1ea3ng c\u00e1ch l\u1edbn nh\u1ea5t: {widest}, +{widest_delta} tr\u00ean 1.000 c\u00e2u "
                             "({widest_ratio}\u00d7). T\u1ef7 l\u1ec7 d\u1ed1c nh\u1ea5t: {steepest} \u1edf {steepest_ratio}\u00d7 "
                             "\u2014 nh\u01b0ng ch\u1ec9 +{steepest_delta}, v\u00ec m\u1ed9t m\u00f4 h\u00ecnh r\u1ebb v\u1edbi b\u1ed9 t\u00e1ch token "
                             "k\u00e9m hi\u1ec7u qu\u1ea3 th\u00ec t\u1ed1n k\u00e9m v\u1ec1 t\u1ef7 l\u1ec7 v\u00e0 nh\u1ecf v\u1ec1 ti\u1ec1n.",
        "cap_dumbbell_axes": "\u0110\u00e2y l\u00e0 hai tr\u1ee5c kh\u00e1c nhau: b\u1ed9 t\u00e1ch token \u0111\u1ecbnh ra t\u1ef7 l\u1ec7, c\u00f2n "
                             "b\u1eadc ph\u1ee5c v\u1ee5 \u0111\u1ecbnh ra \u0111\u1ed9 l\u1edbn c\u1ee7a h\u00f3a \u0111\u01a1n m\u00e0 t\u1ef7 l\u1ec7 \u1ea5y \u00e1p l\u00ean.",
    })

LOCALES = (EN, VI)

# The locale in force for the current render — module state for the same reason
# _T is (see above).
_L: Locale = EN


@contextmanager
def _use_locale(locale: Locale):
    """Render under `locale`, restoring the previous one afterwards."""
    global _L
    prev, _L = _L, locale
    try:
        yield locale
    finally:
        _L = prev


def _s(key: str, **kw) -> str:
    """A catalogue string, formatted. Missing keys fail loudly rather than
    falling back to English: a half-translated figure is worse than a build
    error, because nothing downstream can detect it."""
    return _L.t[key].format(**kw)


# Colour follows the TOKENIZER; the tint STEP within that colour follows the
# serving SKU.
#
# One hue per distinct tokenizer, assigned in the tokenizer figures' canonical
# order (ascending Vietnamese premium) and never cycled. Models that SHARE a
# tokenizer share its hue and are separated by lightness instead. Two payoffs:
#
#  * A hue means the same thing in EVERY figure. An earlier cut gave the three
#    dollar-only Claude SKUs their own hues by reusing slots held by counters that
#    never appear beside them. That held within a chart — but across the figure set
#    blue meant Qwen in the cost-driver figure and Sonnet 5 in the dollar figure,
#    which is exactly the cross-figure confusion a stable mapping exists to
#    prevent. Ten counters reach a chart and the palette holds eight, so
#    per-counter hues could never have been collision-free anyway.
#  * The encoding states the argument. The dollar figures exist to show ONE
#    tokenizer priced several ways; same hue + a darker step says that directly,
#    where distinct hues implied unrelated tokenizers and left the caption to
#    argue the reader back out of it.
#
# Which hue lands on which tokenizer is chosen for the RENDERED bar orders, not
# for slot order. The two bar figures order their columns differently — the
# tokenizer figures ascend by Vietnamese premium (_headline_order), the dollar
# figures by price (_priced_order) — so a mapping is only safe if BOTH sequences
# clear the gates. Assigning slots down the tokenizer order alone (the earlier
# cut) left every priced column drawing from the palette's tail: blue and orange
# went to Qwen and Llama, which are unpriced and never reach a dollar figure, so
# the dollar charts were built entirely from aqua/yellow/magenta/green — two of
# the three sub-3:1 slots plus two greens 15.6 apart, i.e. the weakest four hues
# in the set carrying the article's headline chart. The mapping below is one of
# the assignments that passes both orders; among those it was picked to keep the
# priced tokenizers on the strong hues and to leave magenta out entirely.
#
# Palette provenance: validated as a set for this light surface (#fcfcfb) by a
# CIEDE2000 + CVD-simulation check (the validator in Claude Code's bundled
# `dataviz` Agent Skill). That tool is NOT in this repo and a clean checkout
# cannot re-derive these numbers — they are recorded provenance, not a
# reproducible gate. What IS portable is the gate itself, so any equivalent
# checker can reproduce the verdict: for each ADJACENT pair in a rendered bar
# order, OKLab ΔE×100 >= 8 under protanopia/deuteranopia simulation and >= 15
# unsimulated. Measured, on the surface above:
#
#   tokenizer figures (orange,blue,green,yellow,red,violet,aqua):
#     worst adjacent CVD ΔE 15.3 (>=8), normal-vision ΔE 20.8 (>=15)
#     Both were 9.1 / 15.6 under the previous mapping — at the floor, not clear
#     of it. Tritan separation on the red-yellow adjacency is 7.6, the one number
#     that did not improve; tritanopia is vanishingly rare and that pair carries a
#     legend, a direct label on the group extreme, and the committed CSVs as the
#     table view, so hue is not doing the work alone.
#
#   dollar figures: the HUE figures above no longer describe this chart. It draws
#     NINE flat fills — five of them tint steps, not palette slots — in cost order:
#     #59c253 (Gemini 3.1 Flash-Lite) . #36a231 (Gemini 3.5 Flash) . #6c62d2
#     (Haiku 4.5) . #008300 (Gemini 3.1 Pro) . #f06d67 (Sonnet 5) . #eda100
#     (GPT-5.6) . #4a3aa7 (Sonnet 4.6) . #e34948 (Opus 5) . #b51221 (Fable 5).
#     The older "(violet,green,yellow,violet,red)" sequence and its ΔE 16.2 / 30.3
#     were measured before the tint steps replaced hatch, so they describe a chart
#     that is no longer rendered and are not restated here as if they were.
#
#     ⚠️ THE ORDER ABOVE IS THE 2026-08-16 ONE, AND IT MOVED FOR A REASON WORTH
#     KNOWING: Anthropic repriced Sonnet 5 from $3.00 to $2.00, which lifted it from
#     8th-cheapest to 5th. Nothing about the palette changed — but the chart sorts by
#     COST, so a vendor price change silently reorders the fills and creates
#     adjacencies nobody validated. The adjacent-pair gate is a check on a *rendered
#     order*, not on a palette. THAT IS NOW ENFORCED IN CODE, not remembered:
#     tests/test_palette_adjacency.py derives the order from the committed cost data
#     and fails the suite when any cross-family neighbour drops below the floors.
#
#     It caught a real defect on its first run. The reorder put Sonnet 5 next to
#     GPT-5.6, and #fe7a73 -> #eda100 measured normal ΔE 14.1 (floor 15) with CVD 8.5
#     (floor 8) — a genuine FAIL, and a cross-family pair, so unlike a within-family
#     tint step the categorical gate really does apply. Fixed by darkening Sonnet 5
#     one step inside its own red ramp, #fe7a73 -> #f06d67 (OKLCH L 0.732 -> 0.692 at
#     unchanged hue/chroma), which lifts that pair to normal 15.6 / CVD 10.6 while
#     keeping it the lightest step of the family, so "lighter = cheaper" still holds.
#     Two margins narrowed and both still clear: vs #008300 the CVD separation falls
#     11.8 -> 8.9 (floor 8), and the ramp step to Opus 5 falls ΔL 0.109 -> 0.069
#     (floor 0.06). Neither has much room left — if a future price change squeezes
#     this pair again, re-slot a hue rather than darkening Sonnet 5 further.
#
#     The full adjacent-pair re-run was recorded here as outstanding; it was RUN on
#     2026-07-27 against that date's order, and the result needs stating plainly
#     because the headline verdict is a FAIL:
#
#       * All CROSS-family adjacencies pass. On the CURRENT (2026-08-16) order the
#         worst is #f06d67 -> #eda100 at normal ΔE 15.6 / CVD 10.6, then
#         #008300 -> #f06d67 at 33.6 / 8.9 — floors 15 and 8. On the 2026-07-27
#         order the worst was #008300 -> #eda100 at CVD 16.2 (protan) / 30.3 normal,
#         and #4a3aa7 -> #fe7a73 measured 29.2, reproducing the number recorded
#         below and confirming the checker agrees with the original run.
#       * Every WITHIN-family adjacency fails the categorical normal-vision floor,
#         and always did: green 10.0 and 9.9, red 11.3 and 12.9, violet 13.0 — all
#         below 15. Running the categorical gate over the whole sequence therefore
#         reports FAIL, and reported FAIL for the seven-fill chart that shipped
#         before this one (worst #e34948 -> #fe7a73, 11.3). This is not a
#         regression introduced by the Gemini steps; it is what the tint-step
#         design has always measured.
#
#     The right reading is that the categorical gate is the WRONG gate inside a
#     family, not that the chart is broken. That gate assumes hue = identity and
#     asks whether two fills are confusable; here two steps of one hue ARE the same
#     tokenizer, and their similarity is the message the encoding exists to send.
#     Within a family the governing gate is the ordinal one (single hue, monotone
#     lightness, ΔL >= 0.06), which all three ramps pass. Identity is never carried
#     by fill alone in this chart: every bar is in the legend, and the caption names
#     each model with its price.
#
#     One hard limit worth recording so it is not rediscovered as a bug. Within a
#     single hue ΔE is roughly 100x ΔL, so a ΔE-15 step needs ΔL 0.15. The green
#     ramp's base #008300 is fixed at L 0.529 (it is the validated slot-5 hue and
#     must stay Gemini's colour across every figure) and Gemini's flagship is its
#     DEAREST tier, so the ramp can only run lighter — into a ceiling of L 0.752,
#     where contrast hits the 2:1 floor. That leaves 0.223 of lightness for two
#     steps, i.e. ΔL 0.112 and ΔE ~11 at the absolute best. A ΔE-15 green ramp is
#     therefore unreachable without either moving Gemini off #008300 or letting the
#     ramp run darker than its flagship. Neither is worth it; the steps sit at
#     ΔL 0.0995 to keep the light end at 2.21:1 rather than 2.05:1.
#
# Re-check BOTH rendered orders against the gate above before changing any hue —
# slot order alone is not the thing that ships. Folded members reuse a validated
# hue and differ by a LIGHTNESS STEP (see Theme.tints); texture was the
# previous mechanism and is gone.
_SLOT_BY_FLAGSHIP: dict[str, int] = {
    "qwen-3-8": 1, "llama-4": 0, "gemini-3-1-pro": 5, "o200k_base": 3,
    "claude-new": 7, "claude-old": 6, "cl100k_base": 2,
    # slot 4 (#e87ba4, magenta) is deliberately unassigned — headroom for one
    # more tokenizer without disturbing any existing hue. Re-validate both
    # rendered orders when spending it: it is sub-3:1 on this surface and sits
    # ΔE 6.1 from aqua under protanopia, so where it lands is not free.
}

# LIGHTNESS separates SKUs that share a tokenizer, and therefore a hue. Fills
# are flat — no hatch anywhere. Texture was the previous mechanism and is a
# documented last resort (it belongs to the accessibility/print/forced-colors
# path, not to a chart's default look); at ~20px bar width a dense dot or line
# field also muddies the fill it sits on, which is what made these bars read as
# washed out rather than coloured.
#
# The step is ORDERED BY SERVING PRICE — lighter = cheaper, darker = dearer.
# That is legitimate where a flat value-ramp on nominal categories would not be:
# price tier is an ordered variable, and it is NOT what the bar length already
# encodes (length is price x tokens), so the step adds information instead of
# restating it. Each family's flagship keeps the base hue, so a headline column
# is the same colour in the tokenizer figures and the dollar figures.
#
# Steps are explicit validated hexes rather than computed at render time: the
# ramp is a published design decision, and deriving it from matrix position
# would let a MODEL_MATRIX reorder silently repaint a figure. Generated in OKLCH
# at fixed hue and chroma, then validated as ordinal ramps (see the provenance
# note above for the tool and its portability caveat):
#
#   red / claude-new    #f06d67 -> #e34948 -> #b51221   (Sonnet 5 $2, Opus 5 $5, Fable 5 $10)
#   violet / claude-old #6c62d2 -> #4a3aa7              (Haiku 4.5 $1, Sonnet 4.6 $3)
#   green / gemini      #59c253 -> #36a231 -> #008300   (Flash-Lite $0.25, 3.5 Flash $1.50,
#                                                        3.1 Pro $2.00)
#
# All three pass monotone lightness, adjacent OKLCH ΔL >= 0.06, single hue, and a
# light end clearing 2:1 on this surface (red 2.48:1, violet 4.74:1, green 2.21:1).
# The CROSS-family adjacencies these steps create are validated categorically, like
# any other neighbouring pair: worst is #4a3aa7 -> #fe7a73 at CVD ΔE 29.2.
#
# The green ramp was generated the same way as the other two — fixed OKLCH hue and
# chroma (H 142.5, C 0.180, in gamut at every step, so chroma needed no clamping),
# stepping lightness DOWN as price rises: L 0.727 -> 0.628 -> 0.529, ΔL 0.0995 and
# 0.0993. Its light end is the tightest of the three at 2.21:1, which is why
# Flash-Lite sits at L 0.727 rather than lighter: L 0.749 would read as a cleaner
# ramp step and falls to 2.05:1, and this ramp's cheap end is a real bar in the
# published dollar figures rather than a legend swatch. Note the base hue #008300
# is the DEAREST member here, where in the Claude families the base sits mid-ramp —
# Gemini's flagship is its most expensive tier, so the ramp only runs lighter.
# (the per-theme values now live on LIGHT.tints / DARK.tints above)


def _check_style_registries() -> None:
    """Assert the style tables name real counters, at import.

    `_SLOT_BY_FLAGSHIP` already fails loudly when a counter cannot resolve a slot,
    but `Theme.tints` was only ever read through `.get()`. A renamed counter
    therefore lost its tint step silently, fell back to the family base hue, and
    then tripped `_series_style`'s duplicate-fill check — an error naming a colour
    collision when the actual cause was a stale key here.
    """
    unknown_tints = sorted((set(LIGHT.tints) | set(DARK.tints)) - set(config.MATRIX_BY_ID))
    if unknown_tints:
        raise ValueError(f"Theme.tints keys with no counter in MODEL_MATRIX: "
                         f"{unknown_tints} — a rename left the tint step behind, "
                         "and the counter would silently take its family base hue")
    unknown_slots = sorted(set(_SLOT_BY_FLAGSHIP) - set(config.MATRIX_BY_ID))
    if unknown_slots:
        raise ValueError(f"_SLOT_BY_FLAGSHIP keys with no counter in MODEL_MATRIX: "
                         f"{unknown_slots}")
    for loc in LOCALES:
        unknown_units = sorted(set(loc.units) - set(config.CORPORA))
        if unknown_units:
            raise ValueError(f"Locale {loc.name!r} names units for corpora that are "
                             f"not in config.CORPORA: {unknown_units}")
        missing_units = sorted(set(config.CORPORA) - set(loc.units))
        if missing_units:
            raise ValueError(f"Locale {loc.name!r} has no counting unit for corpora "
                             f"{missing_units} — captions would print a word from "
                             "another corpus")
        missing_langs = sorted(set(config.LANGUAGES) - set(loc.languages))
        if missing_langs:
            raise ValueError(f"Locale {loc.name!r} has no display name for "
                             f"{missing_langs} — the axis would lose a language")
        # Every locale carries the WHOLE catalogue. A partial one renders half a
        # figure in English with nothing downstream able to notice.
        missing_keys = sorted(set(EN.t) - set(loc.t))
        extra_keys = sorted(set(loc.t) - set(EN.t))
        if missing_keys or extra_keys:
            raise ValueError(f"Locale {loc.name!r} catalogue does not match EN's: "
                             f"missing {missing_keys}, unknown {extra_keys}")


def _slot(cid: str) -> int:
    """Colour slot for a counter: its own if headline, else its flagship's."""
    c = config.MATRIX_BY_ID[cid]
    key = cid if c.headline else (c.flagship_group or cid)
    if key not in _SLOT_BY_FLAGSHIP:
        raise ValueError(
            f"no colour slot for {cid!r} (resolved to flagship {key!r}) — add the "
            "flagship to _SLOT_BY_FLAGSHIP, or give the counter a flagship_group")
    return _SLOT_BY_FLAGSHIP[key]


def _series_style(counters: list[str]) -> list[str]:
    """Fill colour per counter, guarding that one chart never repeats one.

    The family hue identifies the tokenizer; a tint step within it identifies
    the serving SKU. Counters sharing a tokenizer therefore share a hue — that
    is the whole point — but must land on different steps of it.
    """
    styles: list[str] = []
    for cid in counters:
        s = _slot(cid)
        # Bounds-check per counter, before the duplicate test: a negative index
        # would otherwise wrap silently to the palette tail, and an over-range
        # one would be misreported as a duplicate-hue problem.
        if not 0 <= s < len(_T.series):
            raise ValueError(
                f"slot {s} for {cid!r} is outside the {len(_T.series)}-slot palette "
                "— a 9th categorical hue is not distinguishable under CVD; fold "
                "the tail into 'Other' or facet into small multiples")
        styles.append(_T.tints.get(cid, _T.series[s]))
    if len(set(styles)) != len(styles):
        dupes = sorted({c for c, st in zip(counters, styles)
                        if styles.count(st) > 1})
        raise ValueError(
            f"counters {dupes} resolve to the same fill in one chart — they "
            "would be indistinguishable. Give one a validated step in "
            "Theme.tints; but if this fired because you priced a "
            "same-tokenizer-same-price proxy, read the PRICING note on "
            "claude-opus-4-8 first — the right fix is to leave it unpriced. "
            "Two SKUs at one price cannot take ordered steps honestly, because "
            "the step encodes price")
    return styles


def _style_axes(ax, ylabel: str) -> None:
    """Recessive chrome: the data is the only thing allowed to be loud.

    Hairline horizontal grid behind the bars, no top/right/left frame, muted
    tick text, and a single quiet baseline for the bars to grow from.
    """
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_T.grid, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_T.baseline)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=_T.ink_muted, labelsize=9, length=0)
    ax.set_ylabel(ylabel, fontsize=9, color=_T.ink_2)


def _legend_above(ax, ncol: int) -> None:
    """Legend outside the plot, above it — never floating over the bars.

    Identity is never colour-alone, so the legend is always present for >=2
    series; its text wears secondary ink while the swatch beside it carries the
    series colour.
    """
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol,
                    frameon=False, fontsize=8.5, labelcolor=_T.ink_2,
                    handlelength=0.85, handleheight=0.85, borderpad=0,
                    columnspacing=1.5, handletextpad=0.5)
    return leg


def _grouped_bars(ax, groups: list[str], series: list[str],
                  values, labels: list[str]):
    """Grouped bars with air between them, and a value on the group's extreme.

    Bar thickness is capped below the slot width so the leftover reads as the
    separator — white doing the work, rather than a stroke drawn around each
    bar. Only the tallest bar per group is labelled: a number on all 35 marks
    is chaos and goes unread, while the extreme is the one the reader is
    scanning for.
    """
    if not series:
        raise ValueError("no counters to plot — the caller filtered every series "
                         "out; check the corpus slice reached this figure")
    # NaN must not reach the renderer: it annotates a literal "nan" on the chart,
    # and because every NaN comparison is False the group-extreme search below
    # returns the FIRST bar rather than the tallest — so the single value label
    # the chart carries can point at the wrong bar even when the rest are valid.
    nan_at = [(labels[k], groups[gi]) for k in range(len(series))
              for gi in range(len(groups)) if not np.isfinite(values[k][gi])]
    if nan_at:
        raise ValueError(
            f"non-finite value(s) for {nan_at} — refusing to render. A NaN prints "
            "as 'nan' on the bar and misplaces the group-extreme label; fix the "
            "upstream data instead.")
    x = np.arange(len(groups))
    slot = 0.84 / len(series)
    width = slot * 0.78          # leftover slot = the surface gap between bars
    # A zero-height proxy bar registers the series in the legend; the visible mark
    # is drawn by _rounded_end_bar so the data end is rounded and the baseline
    # stays square. bar() cannot do two-corner rounding.
    for k, color in enumerate(_series_style(series)):
        ax.bar(x + k * slot, np.zeros(len(groups)), width, label=labels[k],
               color=color, linewidth=0)
    ax.autoscale_view()
    ymax = max(max(v) for v in values)
    ax.set_ylim(0, ymax * 1.10)
    for k, color in enumerate(_series_style(series)):
        for gi in range(len(groups)):
            _rounded_end_bar(ax, x=x[gi] + k * slot - width / 2, y=0,
                             w=width, h=values[k][gi], color=color,
                             horizontal=False)
    for gi in range(len(groups)):
        col = [values[k][gi] for k in range(len(series))]
        top = max(range(len(col)), key=lambda k: col[k])
        ax.annotate(_fmt(col[top]), (x[gi] + top * slot, col[top]),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=7.5, color=_T.ink_2)
    ax.set_xticks(x + slot * (len(series) - 1) / 2)
    ax.set_xticklabels([_L.languages[l] for l in groups], fontsize=9)
    ax.set_xlim(-0.5 * slot - 0.12, len(groups) - 1 + slot * len(series) + 0.02)


def _num(v: float, dp: int) -> str:
    """A number in the current locale's convention.

    Python's `,` format spec always emits English separators, so every figure
    number is formatted here and then re-punctuated. Vietnamese swaps both marks
    (1.234,56 for 1,234.56), which is why the swap goes via a sentinel rather
    than two chained replaces — replacing "," first and "." second would undo
    itself on any number carrying both.
    """
    s = f"{v:,.{dp}f}"
    return s.replace(",", "\x00").replace(".", _L.decimal).replace("\x00", _L.group)


def _money(v: float, dp: int | None = None) -> str:
    """A USD amount, punctuated and positioned per locale ($5.00 / 5,00 USD)."""
    if dp is None:
        dp = 0 if float(v).is_integer() else 2
    return _L.money.format(v=_num(v, dp))


def _date(iso: str) -> str:
    """An ISO date in the locale's convention — d/m/yyyy for Vietnamese."""
    if _L.date == "iso":
        return iso
    y, m, d = iso.split("-")
    return f"{int(d)}/{int(m)}/{y}"


def _fmt(v: float) -> str:
    """Compact tick/label text: no trailing noise on round numbers."""
    if v >= 100:
        return _num(v, 0)
    if v >= 10:
        return _num(v, 1)
    return _num(v, 2)

FIG_DIR = config.RESULTS_DIR / "figures"
CONTRAST = [l for l in config.LANGUAGES if l != config.BASELINE_LANG]
LEAD_LANG = "vie_Latn"  # article lead; canonical column order sorts by its premium



def _price_as_of(counters: list[str]) -> str:
    """As-of stamp covering the prices actually used, as a range if they differ.

    Rows carry their own `as_of` (claude-new was re-verified 2026-07-25 when it
    began naming Opus 5, a model that did not exist on the 2026-07-18 ratification
    date). Printing the single global constant would date every price to the
    oldest ratification; printing the newest would claim they were all re-checked
    then. Neither is true, so print what is.
    """
    dates = sorted({config.PRICING[c].as_of for c in counters
                    if config.PRICING.get(c)
                    and config.PRICING[c].input_usd_per_mtok is not None})
    if not dates:
        return _date(config.PRICING_AS_OF)
    return (_date(dates[0]) if len(dates) == 1
            else f"{_date(dates[0])}\u2013{_date(dates[-1])}")


def _price_confidence_note(counters: list[str]) -> list[str]:
    """Name any drawn price the config does not grade `high`. Usually empty.

    `Price.confidence` reached `cost_by_language.price_confidence` but never a
    figure, so a bar computed from a hedged price rendered under a caption reading
    only "input list price" — identical authority to a fully-sourced one, with the
    hedge visible solely to a reader who opened the CSV. Every priced row is `high`
    as of 2026-07-27 (the last `medium`, o200k_base, was resolved by claiming its
    exact SKU rather than by softening the caption), so this line renders nothing
    today. It exists so that the next hedged price cannot silently inherit the
    confident caption: the qualifier appears the moment one is added, without
    anyone remembering to edit prose here.

    Deliberately names the counters rather than counting them — "1 price is
    medium-confidence" tells a reader to distrust the chart; naming it tells them
    which bar.
    """
    hedged = [c for c in counters
              if (p := config.PRICING.get(c)) is not None
              and p.input_usd_per_mtok is not None
              and p.confidence != "high"]
    if not hedged:
        return []
    named = ", ".join(f"{_label(c)} ({config.PRICING[c].confidence})"
                      for c in hedged)
    return [_s("cap_price_confidence", named=named)]


def _same_tokens_diff_price(agg: pd.DataFrame, corpus_id: str,
                            counters: list[str]) -> list[str]:
    """The "same tokens, different price" line — derived AND verified.

    Built from the counters actually on the chart and from `config.PRICING`, so it
    can neither name a model that has no bar nor quote a price the bars weren't
    computed from. An earlier cut hardcoded six model/price pairs: it named an
    Opus 4.8 bar that did not exist, opened a line about differing prices with two
    identical ones, and would have gone stale silently the moment a price was
    re-ratified (Sonnet 5's intro price reverts 2026-09-01).

    The "same tokens" half is now checked against the counts, not taken from
    config. This module's contract says a fold "is re-verified here against the
    counts — never asserted"; `_legend_lines` honoured that via `_identical` while
    this function, making the identical claim on the chart whose entire thesis is
    the fold, grouped on `flagship_group` metadata alone. Perturbing one member's
    totals made the two captions in the same figure set disagree, with the
    unverified one still listing the diverged model.
    """
    groups: dict[str, list[str]] = {}
    for cid in counters:
        c = config.MATRIX_BY_ID[cid]
        price = config.PRICING.get(cid)
        if price is None or price.input_usd_per_mtok is None:
            continue
        key = cid if c.headline else (c.flagship_group or cid)
        # Only claim shared tokens where the counts actually agree in this corpus.
        if key != cid and not _identical(agg, corpus_id, cid, key):
            continue
        groups.setdefault(key, []).append(
            f"{_label(cid)} {_money(price.input_usd_per_mtok)}")
    parts = [" / ".join(v) for v in groups.values() if len(v) > 1]
    if not parts:
        return []
    return [_s("cap_same_tokens", parts="; ".join(parts))]


def _unpriced_line(cost: pd.DataFrame, drawn: list[str]) -> list[str]:
    """Name the headline counters that were MEASURED but carry no serving price.

    Keyed on PRICING, not on "measured but not drawn". The older test —
    `c.id in measured and c.id not in drawn` — was true for two very different
    reasons, and stated only one of them: a counter dropped by the column-ordering
    filter (see `_rank_across_corpora`) was reported as having "no serving list
    price" even when it carried a high-confidence one. That is a false claim in a
    published asset, produced by an ordering bug. A counter that is priced but
    missing from the chart is a defect, not a caption — `_check_drawn_coverage`
    raises on it instead.
    """
    measured = set(cost.counter_id)
    omitted = [c.id for c in config.MODEL_MATRIX
               if c.headline and c.id in measured and c.id not in drawn
               and (config.PRICING.get(c.id) is None
                    or config.PRICING[c.id].input_usd_per_mtok is None)]
    if not omitted:
        return []
    return [_s("cap_unpriced", names=", ".join(_label(c) for c in omitted))]


def _check_drawn_coverage(frame: pd.DataFrame, drawn: list[str], what: str) -> None:
    """Raise if a counter that belongs on this chart was silently dropped.

    The column-ordering functions used to rank on one corpus and hard-filter to
    that rank, so a counter measured only in the *other* corpus vanished from all
    four figures of both corpora while its rows sat in every committed CSV — with
    no error, because the only guard was "no counters at all".
    """
    priced_only = what == "dollar"
    expected = {c.id for c in config.MODEL_MATRIX
                if c.headline and c.id in set(frame.counter_id)
                and (not priced_only
                     or (config.PRICING.get(c.id)
                         and config.PRICING[c.id].input_usd_per_mtok is not None))}
    missing = sorted(expected - set(drawn))
    if missing:
        raise ValueError(
            f"{what} figure would omit measured counter(s) {missing} that have rows "
            "in this corpus. This is a column-ordering defect, not a caption: fix "
            "the order rather than letting the chart disagree with the dataset.")


def _label(cid: str) -> str:
    c = config.MATRIX_BY_ID[cid]
    return c.headline_display or c.display


def _corpus_preference() -> list[str]:
    """Corpora in ranking-preference order: FLORES+ first, then the rest, once each.

    Spelled `("flores", *config.CORPORA)` before, which names flores twice and
    expressed the "prefer FLORES+" intent only by accident — reorder CORPORA and
    the preference silently changes.
    """
    return ["flores"] + [c for c in config.CORPORA if c != "flores"]


def _rank_across_corpora(frame: pd.DataFrame, value_col: str) -> dict[str, float]:
    """Lead-language rank keys, taken from the preferred corpus and COMPLETED from
    the others.

    Ranking on one corpus and hard-filtering to that rank dropped any counter
    absent there from every figure of every corpus. Preference still decides the
    order (so the four figures line up column-for-column); later corpora only
    supply keys for counters the preferred one does not cover, which are appended
    after the ranked ones rather than discarded.
    """
    rank: dict[str, float] = {}
    for corpus in _corpus_preference():
        sub = frame[(frame.corpus == corpus) & (frame.lang == LEAD_LANG)]
        for r in sub.itertuples():
            v = getattr(r, value_col)
            if pd.notna(v):
                rank.setdefault(r.counter_id, float(v))
    return rank


def _headline_order(premium: pd.DataFrame, present: list[str]) -> list[str]:
    """Headline counter ids present in `present`, ordered by the lead language's
    premium (prefer FLORES+ so the order is stable across corpora — the four
    figures then line up column-for-column). Non-headline (folded) ids drop out.

    A counter with no lead-language row anywhere sorts last rather than vanishing.
    """
    ids = [c.id for c in config.MODEL_MATRIX if c.headline and c.id in present]
    rank = _rank_across_corpora(premium, "premium_aggregate")
    return sorted(ids, key=lambda i: (i not in rank, rank.get(i, 0.0), i))


def _identical(agg: pd.DataFrame, corpus_id: str, a: str, b: str) -> bool:
    """True iff counters `a` and `b` produce identical per-language token totals
    in `corpus_id` — the in-dataset proof behind a "shared tokenizer" claim."""
    piv = (agg[agg.corpus == corpus_id]
           .pivot(index="lang", columns="counter_id", values="total_tokens"))
    if a not in piv.columns or b not in piv.columns:
        return False
    return bool((piv[a] == piv[b]).all())


def _legend_lines(agg: pd.DataFrame, corpus_id: str, shown: list[str]) -> list[str]:
    """Human-readable fold notes for the columns actually shown, in column order.
    A "shared" fold is emitted only if re-verified byte-identical in this corpus."""
    lines: list[str] = []
    for flag in shown:
        folded = [c for c in config.MODEL_MATRIX if c.flagship_group == flag]
        # Never list a proxy that measures the SAME endpoint as the column itself —
        # it would render one model twice in an equality chain ("Opus 5 = Opus 5"),
        # which reads as a bug. The test is on `spec` (the endpoint actually
        # measured), NOT on display text: an earlier cut compared labels by
        # substring, which silently drops a genuinely distinct model the moment a
        # label gains a suffix ("Opus 5" ⊂ "Claude Opus 5.1") and fails to fire at
        # all if a `headline_display` is removed. Identity is the real question, so
        # ask it directly. Applied to both fold kinds for symmetry.
        flag_spec = config.MATRIX_BY_ID[flag].spec
        folded = [c for c in folded if c.spec != flag_spec]
        # "shared" is the only supported fold_reason (asserted in
        # config._check_matrix_integrity). A "superseded" branch used to live here
        # and printed "<model> superseded, within X% — omitted", attributing to a
        # tokenizer difference what was entirely a corpus-markup artifact: the one
        # FLORES sentence carrying an HTML tag. It stopped rendering when the last
        # superseded counter was removed, but the string — and the reasoning error
        # in it — survived in the code, ready to re-render. Removed outright; a
        # genuine new tokenizer earns its own column, established by a vocabulary
        # diff rather than by quoting a divergence percentage.
        shared = [m for m in folded
                  if m.fold_reason == "shared" and _identical(agg, corpus_id, m.id, flag)]
        if shared:
            names = " = ".join([_label(flag)] + [_label(m.id) for m in shared])
            note = _s("cap_shared_tokenizer", names=names)
            # Same tokens, different serving price: name the cheapest member from
            # PRICING and from the models actually in this equality chain. This was
            # a hardcoded {"claude-new": "Sonnet 5", ...} keyed on the flagship id
            # alone — so it printed "prices differ (Sonnet 5 cheapest)" even when
            # Sonnet 5 had no bar and no measurement in the corpus, and it could not
            # track a price change (Sonnet 5's intro price reverts 2026-09-01). Same
            # defect, and same fix, as _same_tokens_diff_price above.
            priced = [(config.PRICING[c].input_usd_per_mtok, c)
                      for c in [flag] + [m.id for m in shared]
                      if config.PRICING.get(c)
                      and config.PRICING[c].input_usd_per_mtok is not None]
            if len(priced) > 1:
                lo = min(priced)[0]
                if sum(1 for p, _ in priced if p == lo) == 1:
                    note += _s("cap_prices_differ_named",
                               cheapest=_label(min(priced)[1]))
                else:
                    note += _s("cap_prices_differ")
            lines.append(note)
        if flag == "cl100k_base":
            lines.append(_s("cap_cl100k", label=_label(flag)))
    return lines


def _caption(fig, lines: list[str], ax=None, pad: float = -46) -> None:
    """Footnotes under the plot, anchored to the axes rather than the figure.

    Anchoring to the axes (in offset points below its bottom-left) lets the
    tight bounding box grow to exactly fit the text. The previous version
    reserved a fixed fraction of figure height, which — combined with the tight
    bbox at save time — left a large empty band under every chart.

    `pad` is the offset in points and MUST clear the x tick labels, whose height
    depends on rotation and label length. The default suits the bar charts'
    upright labels; the heatmap rotates its labels 25° and passes a larger value.
    Widen it if tick labels grow — nothing detects the collision automatically.
    """
    if not lines:
        return
    if ax is None:
        if not fig.axes:
            raise ValueError("_caption needs an axes to anchor to — pass `ax` "
                             "explicitly, or call after the plot is drawn")
        ax = fig.axes[0]
    ax.annotate("\n".join(lines), xy=(0, 0), xycoords="axes fraction",
                xytext=(0, pad), textcoords="offset points",
                ha="left", va="top", fontsize=7.5, color=_T.ink_muted,
                linespacing=1.6, annotation_clip=False)


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Paint the surface explicitly. Light's value is matplotlib's own default
    # white, so painting it does not alter the light output; dark would otherwise
    # render its ink on a white ground. (Scoped to the surface: other changes in
    # the same cut — rounded bar ends — did move the light SVGs.)
    fig.patch.set_facecolor(_T.surface)
    for ax in fig.get_axes():
        ax.set_facecolor(_T.surface)
    for ext in ("svg", "png"):
        # Drop the wall-clock Date from SVG metadata so re-runs are byte-identical.
        kw = {"metadata": {"Date": None}} if ext == "svg" else {}
        fig.savefig(FIG_DIR / f"{stem}{_L.suffix}{_T.suffix}.{ext}",
                    bbox_inches="tight", dpi=150,
                    facecolor=_T.surface, **kw)
    plt.close(fig)


def premium_heatmap(premium: pd.DataFrame, agg: pd.DataFrame,
                    corpus_id: str, corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(premium.counter_id)]
    if not counters:
        raise ValueError(
            "premium_heatmap: no headline counters present for corpus "
            f"'{corpus_id}'. Nothing to draw — check that the run measured at "
            "least one headline counter, or that _headline_order is not filtering "
            "everything out.")
    _check_drawn_coverage(premium, counters, "premium")
    langs = CONTRAST
    mat = np.array([[premium[(premium.counter_id == c) & (premium.lang == l)]
                     ["premium_aggregate"].iloc[0] for c in counters] for l in langs])
    # isfinite, not isnan: the sibling guard in _grouped_bars already uses it, and
    # +/-inf passed straight through this one into the norm (span = log(inf)),
    # dying as "ValueError: Invalid vmin or vmax" from inside matplotlib. inf is
    # reachable from _premium_table whenever a baseline total is 0.
    if not np.isfinite(mat).all():
        bad = [(_L.languages[langs[i]], counters[j])
               for i, j in zip(*np.where(~np.isfinite(mat)))]
        raise ValueError(
            f"premium_heatmap: non-finite premium for {bad}. Drawing these prints a "
            "literal 'nan×' in the cell and, because every NaN comparison is "
            "False, silently selects the wrong colour scale for the whole chart. "
            "Fix the data rather than rendering it.")

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(counters), 0.7 + 0.6 * len(langs)))
    # Parity (1.0×) is the meaningful midpoint, not the floor. A sequential scale
    # clamped at vmin=1.0 painted every sub-parity cell the same as parity — which
    # flattened exactly the cells that carry the finding (Qwen ZH 0.89–0.96×,
    # Gemini ZH 0.98× on MASSIVE).
    #
    # The diverging scale is SYMMETRIC IN LOG about parity, not linear. A linear
    # TwoSlopeNorm splits the colormap evenly by ramp, not by data span: with a
    # real slice (min ~0.89, max ~2.6) it packed 0.11 of range into half the ramp
    # and spread 1.6 into the other, ~14x more colour per unit below parity — so a
    # 0.89x saving rendered as visually extreme as a 2.6x penalty. Premium is a
    # ratio, so the honest mapping is multiplicative: 2x dearer and 2x cheaper sit
    # equally far from parity, in opposite directions.
    #
    # Three cases, handled symmetrically. The straddling case is the live one; the
    # two one-sided cases used to share a single `else` that clamped vmin=1.0 and
    # painted with a dearer-side sequential ramp. That was right for an all-dearer
    # slice and exactly backwards for an all-cheaper one — its own comment read "no
    # sub-parity cell to show" in the branch where EVERY cell is sub-parity, and
    # imshow then died inside matplotlib with "minvalue must be less than or equal
    # to maxvalue". Not reachable on the committed data (only Qwen is sub-parity on
    # FLORES), but "reachable if one counter is dropped" is not a guarantee.
    imshow_kw: dict = {}
    if mat.min() < 1.0 < mat.max():
        span = max(abs(np.log(mat.min())), abs(np.log(mat.max())))
        norm = mcolors.FuncNorm(
            (lambda v: 0.5 + np.log(np.clip(v, 1e-9, None)) / (2 * span),
             lambda t: np.exp((np.asarray(t) - 0.5) * 2 * span)),
            vmin=float(np.exp(-span)), vmax=float(np.exp(span)))
        cmap = "RdBu_r"
    elif mat.min() >= 1.0:      # every cell at or above parity — dearer side only
        norm, cmap = None, "OrRd"
        imshow_kw = {"vmin": 1.0}
    else:                       # every cell at or below parity — cheaper side only
        norm, cmap = None, "Blues_r"
        imshow_kw = {"vmin": float(mat.min()), "vmax": 1.0}
    im = ax.imshow(mat, cmap=cmap, norm=norm, **imshow_kw, aspect="auto")
    ax.set_xticks(range(len(counters)))
    ax.set_xticklabels([_label(c) for c in counters], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([_L.languages[l] for l in langs], fontsize=9)
    # The heatmap does not go through _style_axes (it has no gridline or baseline
    # to style), so its chrome had no colour set at all and inherited matplotlib's
    # default black. That was invisible while every figure rendered on white, and
    # unreadable the moment one rendered on #1a1a19. Ink the chrome explicitly —
    # the CELL labels below are a separate rule and correctly follow each cell's
    # own luminance rather than the theme.
    ax.tick_params(colors=_T.ink_muted, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_color(_T.baseline)
    for i in range(len(langs)):
        for j in range(len(counters)):
            # Ink contrast follows the rendered cell luminance, so it stays correct
            # under either scale rather than assuming a light-to-dark ramp.
            r, g, b, _ = im.cmap(im.norm(mat[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{_num(mat[i, j], 2)}\u00d7", ha="center", va="center",
                    color="black" if lum > 0.55 else "white", fontsize=9)
    ax.set_title(_s("title_heatmap", corpus=corpus_name), fontsize=11,
                 color=_T.ink)
    cbar = fig.colorbar(im, ax=ax, label=_s("cbar_premium"))
    cbar.ax.yaxis.label.set_color(_T.ink_2)
    cbar.ax.tick_params(colors=_T.ink_muted)
    cbar.outline.set_edgecolor(_T.baseline)
    # Rotated x tick labels here are taller than the bar charts' upright ones, so
    # the caption needs more clearance than the shared default.
    _caption(fig, _legend_lines(agg, corpus_id, counters), ax, pad=-86)
    _save(fig, stem)



def _rounded_end_bar(ax, *, x, y, w, h, color, horizontal: bool, radius_px=4.0):
    """One bar with its DATA end rounded and its baseline end square.

    matplotlib's bar() is a plain rectangle, and FancyBboxPatch rounds all four
    corners -- which detaches the bar from its own baseline and reads as a
    floating pill. Two corners is the spec, so the path is built by hand.

    The radius is specified in POINTS and converted per axis. A radius in data
    units looks circular only on a square aspect: on this chart x spans ~0.8 USD
    while y spans nine categories, so a data-space radius rendered the long bars
    as stretched lozenges with pointed ends. Converting through transData gives
    the same visual corner on both axes whatever the aspect. The radius is then
    clamped to half the bar's thickness and a third of its length, so a thin or
    short bar degrades to a nearly-square end rather than a lozenge.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    if w == 0 or h == 0:
        return
    # data units per pixel, per axis, at the current limits
    (x0p, y0p), (x1p, y1p) = ax.transData.transform([(0, 0), (1, 1)])
    dx_per_px = 1.0 / abs(x1p - x0p) if x1p != x0p else 0
    dy_per_px = 1.0 / abs(y1p - y0p) if y1p != y0p else 0
    rx, ry = radius_px * dx_per_px, radius_px * dy_per_px
    if horizontal:
        rx = min(rx, abs(w) / 3)
        ry = min(ry, abs(h) / 2)
    else:
        rx = min(rx, abs(w) / 2)
        ry = min(ry, abs(h) / 3)
    if rx <= 0 or ry <= 0:
        return
    x1, y1 = x + w, y + h
    if horizontal:                      # square at x, rounded at x+w
        pts = [(x, y), (x1 - rx, y), (x1, y), (x1, y + ry),
               (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x, y1), (x, y)]
    else:                               # square at y, rounded at y+h
        pts = [(x, y), (x, y1 - ry), (x, y1), (x + rx, y1),
               (x1 - rx, y1), (x1, y1), (x1, y1 - ry), (x1, y), (x, y)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(pts, codes), facecolor=color, edgecolor="none",
                           linewidth=0, clip_on=False))


def vietnamese_cost_bars(cost: pd.DataFrame, agg: pd.DataFrame, corpus_id: str,
                         corpus_name: str, order: list[str], stem: str) -> None:
    """The serving-cost ladder for ONE language, horizontally, every bar labelled.

    The five-language grouped chart answers "how does cost vary by language and
    model" with 45 bars and one label. This answers the narrower question the
    article actually asks -- what does serving Vietnamese cost, and how wide is
    the gap -- and answers it in a glance.

    Horizontal because the categories are long model names, which read straight
    across instead of rotated. Linear rather than log because every bar carries
    its own value: the cheap tiers ARE slivers next to the dear ones, that ~83x
    ratio is the finding, and a log axis would flatter it into looking modest.
    The labels are what make the slivers readable, so the scale can stay honest.
    """
    rows = cost[(cost.lang == "vie_Latn") & cost.cost_usd_per_sentence.notna()]
    counters = [c for c in order if c in set(rows.counter_id)]
    if not counters:
        return
    vals = {c: float(rows[rows.counter_id == c]["cost_usd_per_sentence"].iloc[0]) * 1000
            for c in counters}
    counters.sort(key=lambda c: vals[c])           # cheapest at top, matching the article table
    fills = _series_style(counters)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_facecolor(_T.surface)
    ax.xaxis.grid(True, color=_T.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(_T.baseline)
    ax.tick_params(colors=_T.ink_muted, labelsize=9, length=0)

    ys = np.arange(len(counters))[::-1]            # first counter at the top
    # Limits first: _rounded_end_bar reads transData to size its corner in pixels,
    # so drawing before the limits are final would round against a stale scale.
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(c) for c in counters], fontsize=9)
    ax.set_ylim(-0.7, len(counters) - 0.3)
    ax.set_xlim(0, max(vals.values()) * 1.16)      # headroom for the labels
    for y, c, fill in zip(ys, counters, fills):
        _rounded_end_bar(ax, x=0, y=y - 0.3, w=vals[c], h=0.6,
                         color=fill, horizontal=True)
        ax.annotate(_money(vals[c], 4), (vals[c], y), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center", fontsize=8.5,
                    color=_T.ink_2)
    ax.set_xlabel(_s("xlabel_vi_ladder"), fontsize=9, color=_T.ink_2)
    spread = max(vals.values()) / min(vals.values())
    ax.set_title(_s("title_vi_ladder", corpus=corpus_name, spread=_num(spread, 0)),
                 fontsize=11.5, color=_T.ink, pad=18, loc="left")
    _caption(fig, _cost_per_sentence_legend_lines(cost, agg, corpus_id, counters), ax)
    _save(fig, stem)


def vietnamese_tax_dumbbell(cost: pd.DataFrame, agg: pd.DataFrame, corpus_id: str,
                            corpus_name: str, order: list[str], stem: str) -> None:
    """English vs Vietnamese cost per model -- the tax, in money, per vendor.

    The ladder answers "which vendor is cheapest for Vietnamese" and drops the
    language axis entirely; the heatmaps carry the language comparison but in
    TOKENS, not dollars. Neither shows what the article's thesis is actually
    about: how much more the same content costs in Vietnamese, and how much that
    depends on the vendor rather than the language.

    A dumbbell is the honest form for exactly two values per category. The dots
    give the levels, the LINE LENGTH is the tax, and rows sorted by Vietnamese
    cost keep the vendor ladder readable alongside it. It also separates two
    causes a bar chart conflates: Haiku 4.5's line is long because its tokenizer
    is inefficient, Fable 5's because everything about it is dear. Same visual
    quantity, different reasons -- visible here, invisible in a grouped bar.
    """
    rows = cost[cost.cost_usd_per_sentence.notna()]
    def per_1k(cid, lang):
        r = rows[(rows.counter_id == cid) & (rows.lang == lang)]
        return float(r["cost_usd_per_sentence"].iloc[0]) * 1000 if len(r) else None

    pairs = []
    for c in order:
        en, vi = per_1k(c, "eng_Latn"), per_1k(c, "vie_Latn")
        if en and vi:
            pairs.append((c, en, vi))
    if not pairs:
        return
    pairs.sort(key=lambda t: t[2])                 # cheapest Vietnamese at top
    fills = _series_style([c for c, _, _ in pairs])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_facecolor(_T.surface)
    ax.xaxis.grid(True, color=_T.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(_T.baseline)
    ax.tick_params(colors=_T.ink_muted, labelsize=9, length=0)

    ys = np.arange(len(pairs))[::-1]
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(c) for c, _, _ in pairs], fontsize=9)
    ax.set_ylim(-0.8, len(pairs) - 0.2)
    ax.set_xlim(0, max(v for _, _, v in pairs) * 1.30)
    for y, (cid, en, vi), fill in zip(ys, pairs, fills):
        # The connector IS the tax, so it is drawn at FULL opacity. An earlier cut
        # faded it to alpha 0.55 to let the dots lead; that composited several rows
        # under the 2.0:1 ordinal floor against their own surface (Fable 5 reached
        # 1.50:1 on dark, Gemini 3.1 Flash-Lite 1.56:1 on light) — the chart's
        # primary encoded quantity, sunk into the background. No alpha clears the
        # floor for every row, because o200k_base's yellow is only 2.17:1 solid on
        # the light surface, so ANY fade takes it under. Weight carries the
        # hierarchy instead: a thinner solid line against markersize-8/9 dots.
        ax.plot([en, vi], [y, y], color=fill, linewidth=2.5,
                solid_capstyle="round", zorder=2)
        # Hue means MODEL everywhere in this repo, so it cannot also mean language.
        # Fill state carries the language instead: hollow = English, solid =
        # Vietnamese, both in the row's own hue. The earlier version drew the
        # English dot in muted ink and legended "Vietnamese" with one arbitrary
        # model's red, which asserted "red = Vietnamese" — false, and contradicted
        # by nine rows of green, violet and orange Vietnamese dots.
        ax.plot([en], [y], marker="o", markersize=8, markerfacecolor=_T.surface,
                markeredgecolor=fill, markeredgewidth=2.0, zorder=3)
        ax.plot([vi], [y], marker="o", markersize=9, color=fill, zorder=4)
        ax.annotate(f"{_money(vi, 4)}  ({_num(vi / en, 2)}\u00d7)", (vi, y),
                    textcoords="offset points", xytext=(9, 0), ha="left",
                    va="center", fontsize=8.5, color=_T.ink_2)
    # Legend swatches are neutral: they explain the FILL convention, not a colour.
    ax.plot([], [], marker="o", markersize=8, markerfacecolor=_T.surface,
            markeredgecolor=_T.ink_muted, markeredgewidth=2.0, linestyle="none",
            label=_s("legend_baseline_lang"))
    ax.plot([], [], marker="o", markersize=9, color=_T.ink_muted, linestyle="none",
            label=_s("legend_lead_lang"))
    # Upper right, not lower: the rows are sorted cheapest-first so the short bars
    # are at the top and the long ones at the bottom — a lower-right legend lands
    # squarely on the dearest row's value label, which is the one row a reader is
    # most likely to be looking at.
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5,
                    labelcolor=_T.ink_2, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(_T.ink_2)
    ax.set_xlabel(_s("xlabel_dumbbell"), fontsize=9, color=_T.ink_2)
    # This axis draws DOLLARS, so the title must name the widest dollar gap. The
    # first version named the steepest RATIO (Haiku 4.5, 2.42x) and sent the
    # reader hunting for the longest line, which is one of the shortest on the
    # chart — Haiku is a cheap model with an inefficient tokenizer, so its tax is
    # large in proportion and small in money. Naming both, and labelling which is
    # which, is the article's own point rather than a caption apology.
    widest = max(pairs, key=lambda t: t[2] - t[1])
    steepest = max(pairs, key=lambda t: t[2] / t[1])
    ax.set_title(_s("title_vi_tax", corpus=corpus_name),
                 fontsize=11.5, color=_T.ink, pad=18, loc="left")
    lines = [
        _s("cap_dumbbell_gaps",
           widest=_label(widest[0]),
           widest_delta=_money(widest[2] - widest[1], 4),
           widest_ratio=_num(widest[2] / widest[1], 2),
           steepest=_label(steepest[0]),
           steepest_ratio=_num(steepest[2] / steepest[1], 2),
           steepest_delta=_money(steepest[2] - steepest[1], 4)),
        _s("cap_dumbbell_axes"),
    ] + _cost_per_sentence_legend_lines(cost, agg, corpus_id,
                                        [c for c, _, _ in pairs])
    _caption(fig, lines, ax)
    _save(fig, stem)


def _priced_order(cost: pd.DataFrame) -> list[str]:
    """Priced serving options, ordered by the lead language's USD cost (ascending).

    Unlike the tokenizer figures, this does NOT fold shared-tokenizer models: they
    have identical tokens but *different serving prices*, so each is its own bar —
    that price split is exactly what the dollar figure exists to show. Unpriced
    counters (Llama 4 self-host, cl100k historical) fall out for having no USD
    cost. A priced counter with no lead-language row in the preferred corpus sorts
    last rather than being dropped from the chart.
    """
    priced = [c.id for c in config.MODEL_MATRIX
              if not cost[(cost.counter_id == c.id)
                          & cost.cost_usd_per_1k_chars.notna()].empty]
    rank = _rank_across_corpora(cost, "cost_usd_per_1k_chars")
    return sorted(priced, key=lambda i: (i not in rank, rank.get(i, 0.0), i))


def _cost_legend_lines(cost: pd.DataFrame, agg: pd.DataFrame, corpus_id: str,
                       counters: list[str]) -> list[str]:
    return [
        _s("cap_usd_per_1m_chars", as_of=_price_as_of(counters),
           rate=_num(int(config.USD_TO_VND), 0)),
        *_price_confidence_note(counters),
        *_same_tokens_diff_price(agg, corpus_id, counters),
        *_unpriced_line(cost, counters),
    ]


def dollar_cost_bars(cost: pd.DataFrame, agg: pd.DataFrame, corpus_id: str,
                     corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
    _check_drawn_coverage(cost, counters, "dollar")
    langs = list(config.LANGUAGES)
    vals = [[cost[(cost.counter_id == c) & (cost.lang == l)]
             ["cost_usd_per_1k_chars"].iloc[0] * 1000 for l in langs]  # USD / 1M chars
            for c in counters]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, _s("ylabel_usd_per_1m_chars"))
    _grouped_bars(ax, langs, counters, vals, [_label(c) for c in counters])
    ax.set_title(_s("title_dollar_chars", corpus=corpus_name),
                 fontsize=11.5, color=_T.ink, pad=30, loc="left")
    _legend_above(ax, ncol=min(len(counters), 7))
    _caption(fig, _cost_legend_lines(cost, agg, corpus_id, counters), ax)
    _save(fig, stem)


# Per-sentence dollar figure — same price × tokens, divided by the corpus
# sentence count instead of characters. Parallel corpus, so this is cost for the
# same *meaning* across languages; the honest unit for a message/document
# workload. Deliberately contrasts with dollar_cost_bars (per character): a dense
# script (Chinese) towers per character yet ranks low per sentence.
def _unit(corpus_id: str) -> str:
    """The counting unit for a corpus, in the current locale. Falls back to the
    sentence corpus's word rather than to English, so a locale can never leak an
    untranslated unit into a caption."""
    return _L.units.get(corpus_id, _L.units["flores"])


def _cost_per_sentence_legend_lines(cost: pd.DataFrame, agg: pd.DataFrame,
                                    corpus_id: str,
                                    counters: list[str]) -> list[str]:
    unit = _unit(corpus_id)
    return [
        _s("cap_usd_per_1k_units", unit=unit, as_of=_price_as_of(counters),
           rate=_num(int(config.USD_TO_VND), 0)),
        _s("cap_read_against", unit=unit),
        *_price_confidence_note(counters),
        *_same_tokens_diff_price(agg, corpus_id, counters),
        # Both dollar figures drop the same measured-but-unpriced counters, but
        # only the per-character one said so. Qwen 3.6 — the mildest Vietnamese
        # tax in the matrix — silently vanished from this chart with no note.
        *_unpriced_line(cost, counters),
    ]


def dollar_cost_per_sentence_bars(cost: pd.DataFrame, agg: pd.DataFrame,
                                  corpus_id: str,
                                  corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
    _check_drawn_coverage(cost, counters, "dollar")
    langs = list(config.LANGUAGES)
    unit = _unit(corpus_id)
    vals = [[cost[(cost.counter_id == c) & (cost.lang == l)]
             ["cost_usd_per_sentence"].iloc[0] * 1000 for l in langs]  # USD / 1000 units
            for c in counters]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, _s("ylabel_usd_per_1k_units", unit=unit))
    _grouped_bars(ax, langs, counters, vals, [_label(c) for c in counters])
    ax.set_title(_s("title_dollar_units", corpus=corpus_name, unit=unit),
                 fontsize=11.5, color=_T.ink, pad=30, loc="left")
    _legend_above(ax, ncol=min(len(counters), 7))
    _caption(fig, _cost_per_sentence_legend_lines(cost, agg, corpus_id, counters), ax)
    _save(fig, stem)


def cost_driver_bars(cost: pd.DataFrame, agg: pd.DataFrame,
                     corpus_id: str, corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
    _check_drawn_coverage(cost, counters, "cost-driver")
    langs = list(config.LANGUAGES)
    vals = [[cost[(cost.counter_id == c) & (cost.lang == l)]
             ["tokens_per_1k_chars"].iloc[0] for l in langs] for c in counters]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, _s("ylabel_tokens_per_1k_chars"))
    _grouped_bars(ax, langs, counters, vals, [_label(c) for c in counters])
    ax.set_title(_s("title_cost_driver", corpus=corpus_name),
                 fontsize=11.5, color=_T.ink, pad=30, loc="left")
    _legend_above(ax, ncol=min(len(counters), 7))
    _caption(fig, _legend_lines(agg, corpus_id, counters), ax)
    _save(fig, stem)


# Called here rather than beside the tables: the check covers all three style
# registries plus the locale catalogues in one place.
_check_style_registries()


def make_figures() -> None:
    premium = pd.read_csv(config.RESULTS_DIR / "premium_by_language.csv")
    cost = pd.read_csv(config.RESULTS_DIR / "cost_by_language.csv")
    agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")
    # Tokenizer figures share one canonical column order (see _headline_order); the
    # dollar figure has its own (priced options, unfolded — see _priced_order).
    order = _headline_order(premium, sorted(set(premium.counter_id)))
    dollar_order = _priced_order(cost)
    live: set[str] = set()
    for locale in LOCALES:
        for theme in THEMES:
            with _use_locale(locale), _use_theme(theme):
                live |= _render_all(premium, cost, agg, order, dollar_order)
    _prune_stale(live)
    print(f"wrote figures to {FIG_DIR}/ (svg + png, locales: "
          f"{', '.join(l.name for l in LOCALES)}; themes: "
          f"{', '.join(t.name for t in THEMES)})")
    print(f"  tokenizer columns: {[_label(c) for c in order]}")
    print(f"  dollar columns:    {[_label(c) for c in dollar_order]}")


def _render_all(premium, cost, agg, order, dollar_order) -> set[str]:
    """Emit every figure for the theme in force; return the corpora seen."""
    live: set[str] = set()
    for corpus_id, corpus_name in config.CORPORA.items():
        p = premium[premium.corpus == corpus_id]
        c = cost[cost.corpus == corpus_id]
        if p.empty:
            continue
        live.add(corpus_id)
        premium_heatmap(p, agg, corpus_id, corpus_name, order, f"fig-premium-heatmap-{corpus_id}")
        cost_driver_bars(c, agg, corpus_id, corpus_name, order, f"fig-cost-driver-bars-{corpus_id}")
        dollar_cost_bars(c, agg, corpus_id, corpus_name, dollar_order,
                         f"fig-dollar-cost-{corpus_id}")
        dollar_cost_per_sentence_bars(c, agg, corpus_id, corpus_name, dollar_order,
                                      f"fig-dollar-cost-per-sentence-{corpus_id}")
        vietnamese_cost_bars(c, agg, corpus_id, corpus_name, dollar_order,
                             f"fig-vietnamese-cost-ladder-{corpus_id}")
        vietnamese_tax_dumbbell(c, agg, corpus_id, corpus_name, dollar_order,
                                f"fig-vietnamese-tax-gap-{corpus_id}")
    return live


def _prune_stale(live: set[str]) -> None:
    # Remove figures for corpora this dataset no longer covers. The loop above
    # simply stops emitting them, which left four committed, publishable
    # `fig-*-<retired>.{svg,png}` on disk and in git, silently stale beside CSVs
    # that no longer carried the corpus. analyze.py already unlinks
    # within_vendor_inflation.csv for exactly this reason; this is its counterpart.
    # Keyed on the four stems this module emits, so it can only ever remove files
    # it produced — and on `live` rather than config.CORPORA, since a corpus
    # retired from the config is exactly the case that leaves figures behind.
    _STEMS = ("fig-premium-heatmap-", "fig-cost-driver-bars-",
              "fig-dollar-cost-per-sentence-", "fig-dollar-cost-",
              "fig-vietnamese-cost-ladder-", "fig-vietnamese-tax-gap-")
    if FIG_DIR.exists():
        for path in sorted(FIG_DIR.iterdir()):
            if path.suffix not in (".svg", ".png"):
                continue
            for pre in _STEMS:
                if path.stem.startswith(pre):
                    corpus = path.stem[len(pre):]
                    # Strip the theme and locale suffixes before the corpus test,
                    # in the reverse of the order _save appends them. Without this
                    # every dark figure parses as corpus "<corpus>-dark", matches
                    # nothing in `live`, and is deleted on the run that wrote it —
                    # and the same holds for "<corpus>-vi".
                    for group in (THEMES, LOCALES):
                        for v in group:
                            if v.suffix and corpus.endswith(v.suffix):
                                corpus = corpus[: -len(v.suffix)]
                                break
                    if corpus and corpus not in live:
                        print(f"  removing stale figure for corpus '{corpus}' "
                              f"(no longer in the dataset): {path.name}")
                        path.unlink()
                    break


if __name__ == "__main__":
    make_figures()
