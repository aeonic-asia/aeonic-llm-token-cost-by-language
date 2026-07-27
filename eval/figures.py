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
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_SURFACE = "#fcfcfb"     # chart surface
_INK = "#0b0b0b"         # primary ink (titles)
_INK_2 = "#52514e"       # secondary ink (legend, axis titles)
_INK_MUTED = "#898781"   # muted ink (tick labels, captions)
_GRID = "#e1e0d9"        # hairline gridline, one step off the surface
_BASELINE = "#c3c2b7"    # baseline / axis rule


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
#     (Haiku 4.5) . #008300 (Gemini 3.1 Pro) . #eda100 (GPT-5.6) . #4a3aa7
#     (Sonnet 4.6) . #fe7a73 (Sonnet 5) . #e34948 (Opus 5) . #b51221 (Fable 5).
#     The older "(violet,green,yellow,violet,red)" sequence and its ΔE 16.2 / 30.3
#     were measured before the tint steps replaced hatch, so they describe a chart
#     that is no longer rendered and are not restated here as if they were.
#
#     The full adjacent-pair re-run was recorded here as outstanding; it has now
#     BEEN RUN (2026-07-27), and the result needs stating plainly because the
#     headline verdict is a FAIL:
#
#       * All five CROSS-family adjacencies pass, comfortably. Worst is
#         #008300 -> #eda100 at CVD ΔE 16.2 (protan) / 30.3 normal, against floors
#         of 8 and 15. #4a3aa7 -> #fe7a73 measures 29.2, reproducing the number
#         recorded below and confirming the checker agrees with the original run.
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
# hue and differ by a LIGHTNESS STEP (see _TINT_BY_COUNTER); texture was the
# previous mechanism and is gone.
_SLOT_BY_FLAGSHIP: dict[str, int] = {
    "qwen-3-6": 1, "llama-4": 0, "gemini-3-1-pro": 5, "o200k_base": 3,
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
#   red / claude-new    #fe7a73 -> #e34948 -> #b51221   (Sonnet 5 $3, Opus 5 $5, Fable 5 $10)
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
_TINT_BY_COUNTER: dict[str, str] = {
    "claude-sonnet-5": "#fe7a73",        # $3 — lightest of the newer-Claude family
    "claude-fable-5": "#b51221",         # $10 — darkest
    "claude-haiku-4-5": "#6c62d2",       # $1 — lighter of the older-Claude pair
    "gemini-3-5-flash": "#36a231",       # $1.50 — middle step of the green ramp
    "gemini-3-1-flash-lite": "#59c253",  # $0.25 — lightest; cheapest bar in the chart
}


def _check_style_registries() -> None:
    """Assert the style tables name real counters, at import.

    `_SLOT_BY_FLAGSHIP` already fails loudly when a counter cannot resolve a slot,
    but `_TINT_BY_COUNTER` was only ever read through `.get()`. A renamed counter
    therefore lost its tint step silently, fell back to the family base hue, and
    then tripped `_series_style`'s duplicate-fill check — an error naming a colour
    collision when the actual cause was a stale key here.
    """
    unknown_tints = sorted(set(_TINT_BY_COUNTER) - set(config.MATRIX_BY_ID))
    if unknown_tints:
        raise ValueError(f"_TINT_BY_COUNTER keys with no counter in MODEL_MATRIX: "
                         f"{unknown_tints} — a rename left the tint step behind, "
                         "and the counter would silently take its family base hue")
    unknown_slots = sorted(set(_SLOT_BY_FLAGSHIP) - set(config.MATRIX_BY_ID))
    if unknown_slots:
        raise ValueError(f"_SLOT_BY_FLAGSHIP keys with no counter in MODEL_MATRIX: "
                         f"{unknown_slots}")
    unknown_units = sorted(set(_SENTENCE_UNIT) - set(config.CORPORA))
    if unknown_units:
        raise ValueError(f"_SENTENCE_UNIT keys with no corpus in config.CORPORA: "
                         f"{unknown_units}")


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
        if not 0 <= s < len(_SERIES):
            raise ValueError(
                f"slot {s} for {cid!r} is outside the {len(_SERIES)}-slot palette "
                "— a 9th categorical hue is not distinguishable under CVD; fold "
                "the tail into 'Other' or facet into small multiples")
        styles.append(_TINT_BY_COUNTER.get(cid, _SERIES[s]))
    if len(set(styles)) != len(styles):
        dupes = sorted({c for c, st in zip(counters, styles)
                        if styles.count(st) > 1})
        raise ValueError(
            f"counters {dupes} resolve to the same fill in one chart — they "
            "would be indistinguishable. Give one a validated step in "
            "_TINT_BY_COUNTER; but if this fired because you priced a "
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
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=_INK_MUTED, labelsize=9, length=0)
    ax.set_ylabel(ylabel, fontsize=9, color=_INK_2)


def _legend_above(ax, ncol: int) -> None:
    """Legend outside the plot, above it — never floating over the bars.

    Identity is never colour-alone, so the legend is always present for >=2
    series; its text wears secondary ink while the swatch beside it carries the
    series colour.
    """
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol,
                    frameon=False, fontsize=8.5, labelcolor=_INK_2,
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
    width = slot * 0.84          # leftover slot = the surface gap
    for k, color in enumerate(_series_style(series)):
        ax.bar(x + k * slot, values[k], width, label=labels[k],
               color=color, linewidth=0)
    for gi in range(len(groups)):
        col = [values[k][gi] for k in range(len(series))]
        top = max(range(len(col)), key=lambda k: col[k])
        ax.annotate(_fmt(col[top]), (x[gi] + top * slot, col[top]),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=7.5, color=_INK_2)
    ax.set_xticks(x + slot * (len(series) - 1) / 2)
    ax.set_xticklabels([config.LANGUAGES[l] for l in groups], fontsize=9)
    ax.set_xlim(-0.5 * slot - 0.12, len(groups) - 1 + slot * len(series) + 0.02)


def _fmt(v: float) -> str:
    """Compact tick/label text: no trailing noise on round numbers."""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:,.1f}"
    return f"{v:,.2f}"

FIG_DIR = config.RESULTS_DIR / "figures"
CONTRAST = [l for l in config.LANGUAGES if l != config.BASELINE_LANG]
LEAD_LANG = "vie_Latn"  # article lead; canonical column order sorts by its premium

def _usd(v: float) -> str:
    return f"${v:,.0f}" if float(v).is_integer() else f"${v:,.2f}"


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
        return config.PRICING_AS_OF
    return dates[0] if len(dates) == 1 else f"{dates[0]}–{dates[-1]}"


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
    return [f"Not every price is high-confidence — {named}. See `source` in "
            f"config.PRICING and `price_confidence` in cost_by_language.csv"]


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
            f"{_label(cid)} {_usd(price.input_usd_per_mtok)}")
    parts = [" / ".join(v) for v in groups.values() if len(v) > 1]
    if not parts:
        return []
    return ["Same tokens, different price: " + "; ".join(parts)
            + " (per 1M input tokens)"]


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
    return [", ".join(_label(c) for c in omitted)
            + " omitted — no serving list price"]


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
            note = f"{names} — one shared tokenizer, identical counts (verified here)"
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
                    note += f"; prices differ ({_label(min(priced)[1])} cheapest)"
                else:
                    note += "; prices differ"
            lines.append(note)
        if flag == "cl100k_base":
            lines.append(f"{_label(flag)} — GPT-4/3.5-era baseline, historical anchor")
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
                ha="left", va="top", fontsize=7.5, color=_INK_MUTED,
                linespacing=1.6, annotation_clip=False)


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        # Drop the wall-clock Date from SVG metadata so re-runs are byte-identical.
        kw = {"metadata": {"Date": None}} if ext == "svg" else {}
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=150, **kw)
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
        bad = [(config.LANGUAGES[langs[i]], counters[j])
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
    ax.set_yticklabels([config.LANGUAGES[l] for l in langs], fontsize=9)
    for i in range(len(langs)):
        for j in range(len(counters)):
            # Ink contrast follows the rendered cell luminance, so it stays correct
            # under either scale rather than assuming a light-to-dark ramp.
            r, g, b, _ = im.cmap(im.norm(mat[i, j]))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{mat[i, j]:.2f}×", ha="center", va="center",
                    color="black" if lum > 0.55 else "white", fontsize=9)
    ax.set_title(f"Token premium vs. English ({corpus_name})", fontsize=11)
    fig.colorbar(im, ax=ax, label="× English tokens (1.00 = parity)")
    # Rotated x tick labels here are taller than the bar charts' upright ones, so
    # the caption needs more clearance than the shared default.
    _caption(fig, _legend_lines(agg, corpus_id, counters), ax, pad=-86)
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
        f"USD to serve 1,000,000 input characters — input list price "
        f"({_price_as_of(counters)}); VND = USD × {int(config.USD_TO_VND):,}",
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
    _style_axes(ax, "USD per 1,000,000 input characters")
    _grouped_bars(ax, langs, counters, vals, [_label(c) for c in counters])
    ax.set_title(f"Serving cost: USD per 1M input characters — {corpus_name} "
                 f"(price × tokens; lower = cheaper)",
                 fontsize=11.5, color=_INK, pad=30, loc="left")
    _legend_above(ax, ncol=min(len(counters), 7))
    _caption(fig, _cost_legend_lines(cost, agg, corpus_id, counters), ax)
    _save(fig, stem)


# Per-sentence dollar figure — same price × tokens, divided by the corpus
# sentence count instead of characters. Parallel corpus, so this is cost for the
# same *meaning* across languages; the honest unit for a message/document
# workload. Deliberately contrasts with dollar_cost_bars (per character): a dense
# script (Chinese) towers per character yet ranks low per sentence.
_SENTENCE_UNIT = {"flores": "sentence", "massive": "message"}


def _cost_per_sentence_legend_lines(cost: pd.DataFrame, agg: pd.DataFrame,
                                    corpus_id: str,
                                    counters: list[str]) -> list[str]:
    unit = _SENTENCE_UNIT.get(corpus_id, "sentence")
    return [
        f"USD to serve 1,000 {unit}s — parallel corpus, so the SAME content "
        f"across languages (price × tokens per {unit}); input list price "
        f"({_price_as_of(counters)}); VND = USD × {int(config.USD_TO_VND):,}",
        f"Read against the per-character chart: a dense script (Chinese) needs "
        f"few characters, so per-character overstates its cost; per {unit} it "
        f"ranks far lower.",
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
    unit = _SENTENCE_UNIT.get(corpus_id, "sentence")
    vals = [[cost[(cost.counter_id == c) & (cost.lang == l)]
             ["cost_usd_per_sentence"].iloc[0] * 1000 for l in langs]  # USD / 1000 units
            for c in counters]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, f"USD per 1,000 {unit}s")
    _grouped_bars(ax, langs, counters, vals, [_label(c) for c in counters])
    ax.set_title(f"Serving cost: USD per 1,000 {unit}s — {corpus_name} "
                 f"(price × tokens per {unit}; lower = cheaper)",
                 fontsize=11.5, color=_INK, pad=30, loc="left")
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
    _style_axes(ax, "Tokens per 1,000 NFC characters")
    _grouped_bars(ax, langs, counters, vals, [_label(c) for c in counters])
    ax.set_title(f"Cost driver: tokens per 1,000 characters — {corpus_name} "
                 f"(lower = cheaper)",
                 fontsize=11.5, color=_INK, pad=30, loc="left")
    _legend_above(ax, ncol=min(len(counters), 7))
    _caption(fig, _legend_lines(agg, corpus_id, counters), ax)
    _save(fig, stem)


# Called here rather than beside the tables: _SENTENCE_UNIT is defined further
# down, and the check covers all three style registries in one place.
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
    # Remove figures for corpora this dataset no longer covers. The loop above
    # simply stops emitting them, which left four committed, publishable
    # `fig-*-<retired>.{svg,png}` on disk and in git, silently stale beside CSVs
    # that no longer carried the corpus. analyze.py already unlinks
    # within_vendor_inflation.csv for exactly this reason; this is its counterpart.
    # Keyed on the four stems this module emits, so it can only ever remove files
    # it produced — and on `live` rather than config.CORPORA, since a corpus
    # retired from the config is exactly the case that leaves figures behind.
    _STEMS = ("fig-premium-heatmap-", "fig-cost-driver-bars-",
              "fig-dollar-cost-per-sentence-", "fig-dollar-cost-")
    if FIG_DIR.exists():
        for path in sorted(FIG_DIR.iterdir()):
            if path.suffix not in (".svg", ".png"):
                continue
            for pre in _STEMS:
                if path.stem.startswith(pre):
                    corpus = path.stem[len(pre):]
                    if corpus and corpus not in live:
                        print(f"  removing stale figure for corpus '{corpus}' "
                              f"(no longer in the dataset): {path.name}")
                        path.unlink()
                    break
    print(f"wrote figures to {FIG_DIR}/ (svg + png)")
    print(f"  tokenizer columns: {[_label(c) for c in order]}")
    print(f"  dollar columns:    {[_label(c) for c in dollar_order]}")


if __name__ == "__main__":
    make_figures()
