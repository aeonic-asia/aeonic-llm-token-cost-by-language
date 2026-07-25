"""Figures for the article: premium heatmap + per-language cost-driver bars.

Exports SVG (for later fig-NN-*.svg article assets) and PNG (quick view) to
eval/results/figures/. Driven entirely by the committed result CSVs, so figures
regenerate deterministically from the dataset.

Readability: the full matrix carries proxy/duplicate counters (three models on
one shared Claude tokenizer; both Gemini generations) so the eval can *verify*
equivalence. The headline figures collapse those to **one column per distinct
tokenizer**, named by its flagship model (config `headline` / `flagship_group`),
share **one canonical column order** across all four charts (ascending in the
article's lead language, Vietnamese), and print a legend beneath spelling out the
folds. Which counters fold is config; that a "shared" fold really is byte-identical
is re-verified here against the counts — never asserted.

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
# Hatch separates same-tokenizer SKUs sharing one hue (see _HATCH_BY_COUNTER).
# Bars are ~20px wide at this figure size, so the default 1.0 stroke reads as a
# smear; 0.7 keeps the pattern legible without muddying the fill colour.
matplotlib.rcParams["hatch.linewidth"] = 0.7
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# ── design tokens ────────────────────────────────────────────────────────────
# Categorical hues assigned in FIXED slot order and never cycled (provenance and
# validation notes live with _SLOT_BY_FLAGSHIP below). Three slots (aqua/yellow/
# magenta) fall below 3:1 against the surface, which obliges "relief" — the
# committed cost_by_language.csv / premium_by_language.csv are that table view,
# and the extreme in each group is directly labelled.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_SURFACE = "#fcfcfb"     # chart surface
_INK = "#0b0b0b"         # primary ink (titles)
_INK_2 = "#52514e"       # secondary ink (legend, axis titles)
_INK_MUTED = "#898781"   # muted ink (tick labels, captions)
_GRID = "#e1e0d9"        # hairline gridline, one step off the surface
_BASELINE = "#c3c2b7"    # baseline / axis rule


# Colour follows the TOKENIZER; texture follows the serving SKU.
#
# One hue per distinct tokenizer, assigned in the tokenizer figures' canonical
# order (ascending Vietnamese premium) and never cycled. Models that SHARE a
# tokenizer share its hue and are separated by hatch instead. Two payoffs:
#
#  * A hue means the same thing in EVERY figure. An earlier cut gave the three
#    dollar-only Claude SKUs their own hues by reusing slots held by counters that
#    never appear beside them. That held within a chart — but across the figure set
#    blue meant Qwen in the cost-driver figure and Sonnet 5 in the dollar figure,
#    which is exactly the cross-figure confusion a stable mapping exists to
#    prevent. Ten counters reach a chart and the palette holds eight, so
#    per-counter hues could never have been collision-free anyway.
#  * The encoding states the argument. The dollar figures exist to show ONE
#    tokenizer priced several ways; same hue + different texture says that
#    directly, where distinct hues implied unrelated tokenizers and left the
#    caption to argue the reader back out of it.
#
# Palette provenance: validated as a set for this light surface — all inside the
# lightness band, all above the chroma floor, worst adjacent CVD ΔE 9.1 (>=8) and
# worst adjacent normal-vision ΔE 19.6 (>=15), checked on the RENDERED bar order
# rather than slot order (slot-order checking alone missed a normal-vision ΔE 12.9
# adjacency failure in an earlier mapping). The tool used was external to this
# repo and is NOT committed here, so those figures cannot be re-derived from a
# clean checkout — treat them as recorded provenance, not a reproducible gate, and
# re-validate with an equivalent CIEDE2000 + CVD-simulation check before changing
# any hue. Under the current scheme only the seven headline hues need checking;
# folded members reuse a validated hue and differ by texture.
_SLOT_BY_FLAGSHIP: dict[str, int] = {
    "qwen-3-6": 0, "llama-4": 1, "gemini-3-1-pro": 2, "o200k_base": 3,
    "claude-new": 4, "claude-old": 5, "cl100k_base": 6,
    # slot 7 (#e34948) is deliberately unassigned — headroom for one more
    # tokenizer without disturbing any existing hue.
}

# Texture separates SKUs that share a tokenizer, and therefore a hue. Each
# group's flagship is solid; every folded member takes an explicit pattern.
# Explicit rather than derived from matrix position, so reordering MODEL_MATRIX
# cannot silently repaint a published figure.
# Patterns are chosen to differ in KIND (lines vs dots vs cross), not merely in
# angle: at the ~20px bar width these figures render at, "///" and "\\\" are not
# reliably tellable apart, so opposite diagonals do not count as a distinction.
_HATCH_BY_COUNTER: dict[str, str] = {
    "claude-sonnet-5": "///",
    "claude-fable-5": "...",
    "claude-haiku-4-5": "///",
    # Never charted today (unpriced / superseded), but mapped so a future
    # promotion is distinguishable rather than an exception.
    "claude-opus-4-8": "xxx",
    "gemini-3-pro": "...",
}


def _slot(cid: str) -> int:
    """Colour slot for a counter: its own if headline, else its flagship's."""
    c = config.MATRIX_BY_ID[cid]
    key = cid if c.headline else (c.flagship_group or cid)
    if key not in _SLOT_BY_FLAGSHIP:
        raise ValueError(
            f"no colour slot for {cid!r} (resolved to flagship {key!r}) — add the "
            "flagship to _SLOT_BY_FLAGSHIP, or give the counter a flagship_group")
    return _SLOT_BY_FLAGSHIP[key]


def _series_style(counters: list[str]) -> list[tuple[str, str]]:
    """(hue, hatch) per counter, guarding that one chart never repeats a pair.

    Hue identifies the tokenizer, hatch the serving SKU within it. Two counters
    may legitimately share a hue — that is the whole point — but never a
    (hue, hatch) pair, which would render them indistinguishable.
    """
    styles: list[tuple[str, str]] = []
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
        styles.append((_SERIES[s], _HATCH_BY_COUNTER.get(cid, "")))
    if len(set(styles)) != len(styles):
        dupes = sorted({c for c, st in zip(counters, styles)
                        if styles.count(st) > 1})
        raise ValueError(
            f"counters {dupes} resolve to the same (colour, hatch) in one chart — "
            "they would be indistinguishable. Give one a pattern in "
            "_HATCH_BY_COUNTER; but if this fired because you priced a "
            "same-tokenizer-same-price proxy, read the PRICING note on "
            "claude-opus-4-8 first — the right fix is to leave it unpriced")
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
    x = np.arange(len(groups))
    slot = 0.84 / len(series)
    width = slot * 0.84          # leftover slot = the surface gap
    for k, (cid, (color, hatch)) in enumerate(zip(series, _series_style(series))):
        # Hatch is drawn in the edge colour: the surface tone cuts the pattern out
        # of the fill, so texture reads without adding a second hue or an outline.
        ax.bar(x + k * slot, values[k], width, label=labels[k],
               color=color, hatch=hatch, edgecolor=_SURFACE, linewidth=0)
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


def _same_tokens_diff_price(counters: list[str]) -> list[str]:
    """The "same tokens, different price" line, DERIVED from what is drawn.

    Built from the counters actually on the chart and from `config.PRICING`, so it
    can neither name a model that has no bar nor quote a price the bars weren't
    computed from. An earlier cut hardcoded six model/price pairs: it named an
    Opus 4.8 bar that did not exist, opened a line about differing prices with two
    identical ones, and would have gone stale silently the moment a price was
    re-ratified (Sonnet 5's intro price reverts 2026-09-01). A group is named only
    if two or more of its priced SKUs are present — with one, the claim is vacuous.
    """
    groups: dict[str, list[str]] = {}
    for cid in counters:
        c = config.MATRIX_BY_ID[cid]
        price = config.PRICING.get(cid)
        if price is None or price.input_usd_per_mtok is None:
            continue
        key = cid if c.headline else (c.flagship_group or cid)
        groups.setdefault(key, []).append(
            f"{_label(cid)} {_usd(price.input_usd_per_mtok)}")
    parts = [" / ".join(v) for v in groups.values() if len(v) > 1]
    if not parts:
        return []
    return ["Same tokens, different price: " + "; ".join(parts)
            + " (per 1M input tokens)"]


def _unpriced_line(cost: pd.DataFrame, drawn: list[str]) -> list[str]:
    """Name the headline counters that were MEASURED but carry no serving price.

    Derived from the cost frame rather than hardcoded, so a counter that simply
    never ran is not falsely reported as "omitted — no serving list price".
    """
    measured = set(cost.counter_id)
    omitted = [c.id for c in config.MODEL_MATRIX
               if c.headline and c.id in measured and c.id not in drawn]
    if not omitted:
        return []
    return [", ".join(_label(c) for c in omitted)
            + " omitted — no serving list price"]


def _label(cid: str) -> str:
    c = config.MATRIX_BY_ID[cid]
    return c.headline_display or c.display


def _headline_order(premium: pd.DataFrame, present: list[str]) -> list[str]:
    """Headline counter ids present in `present`, ordered by the lead language's
    premium (prefer FLORES+ so the order is stable across corpora — the four
    figures then line up column-for-column). Non-headline (folded) ids drop out.
    """
    ids = [c.id for c in config.MODEL_MATRIX if c.headline and c.id in present]
    for corpus in ("flores", *config.CORPORA):
        sub = premium[(premium.corpus == corpus) & (premium.lang == LEAD_LANG)]
        if not sub.empty:
            rank = {r.counter_id: r.premium_aggregate for r in sub.itertuples()}
            return sorted([i for i in ids if i in rank], key=lambda i: rank[i])
    return ids


def _identical(agg: pd.DataFrame, corpus_id: str, a: str, b: str) -> bool:
    """True iff counters `a` and `b` produce identical per-language token totals
    in `corpus_id` — the in-dataset proof behind a "shared tokenizer" claim."""
    piv = (agg[agg.corpus == corpus_id]
           .pivot(index="lang", columns="counter_id", values="total_tokens"))
    if a not in piv.columns or b not in piv.columns:
        return False
    return bool((piv[a] == piv[b]).all())


def _max_rel_divergence(agg: pd.DataFrame, corpus_id: str, a: str, b: str):
    """Max per-language relative gap between counters `a` and `b` in `corpus_id`,
    or None if either is absent. Lets the "superseded, within X%" legend quote a
    number computed from the data instead of a hard-coded claim that can go stale."""
    piv = (agg[agg.corpus == corpus_id]
           .pivot(index="lang", columns="counter_id", values="total_tokens"))
    if a not in piv.columns or b not in piv.columns:
        return None
    return float(((piv[a] - piv[b]).abs() / piv[b]).max())


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
        shared = [m for m in folded
                  if m.fold_reason == "shared" and _identical(agg, corpus_id, m.id, flag)]
        superseded = [m for m in folded if m.fold_reason == "superseded"]
        if shared:
            names = " = ".join([_label(flag)] + [_label(m.id) for m in shared])
            note = f"{names} — one shared tokenizer, identical counts (verified here)"
            # same tokens, different serving price — name the cheapest in the group
            cheapest = {"claude-new": "Sonnet 5", "claude-old": "Haiku 4.5"}.get(flag)
            if cheapest:
                note += f"; prices differ ({cheapest} cheapest)"
            lines.append(note)
        for m in superseded:
            div = _max_rel_divergence(agg, corpus_id, m.id, flag)
            if div is None:
                continue  # superseded counter absent in this corpus — nothing to note
            within = "identical" if div == 0 else f"within {div * 100:.2f}%"
            lines.append(
                f"{_label(flag)} shown (current flagship); "
                f"{_label(m.id)} superseded, {within} — omitted")
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
    langs = CONTRAST
    mat = np.array([[premium[(premium.counter_id == c) & (premium.lang == l)]
                     ["premium_aggregate"].iloc[0] for c in counters] for l in langs])

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(counters), 0.7 + 0.6 * len(langs)))
    # Parity (1.0×) is the meaningful midpoint, not the floor. A sequential scale
    # clamped at vmin=1.0 painted every sub-parity cell the same as parity — which
    # flattened exactly the cells that carry the finding (Qwen ZH 0.89–0.96×,
    # Gemini ZH 0.98× on MASSIVE). A diverging scale centred on parity makes
    # "cheaper than English" visible as its own direction.
    if mat.min() < 1.0 < mat.max():
        norm = mcolors.TwoSlopeNorm(vmin=mat.min(), vcenter=1.0, vmax=mat.max())
        cmap, mid = "RdBu_r", None
    else:                       # degenerate slice: no sub-parity cell to show
        norm, cmap, mid = None, "OrRd", 1.0
    im = ax.imshow(mat, cmap=cmap, norm=norm,
                   **({} if norm is not None else {"vmin": mid}), aspect="auto")
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
    that price split is exactly what the dollar figure exists to show. Superseded
    near-duplicates (gemma3) are still dropped, and unpriced counters (Llama 4
    self-host, cl100k historical) fall out for having no USD cost.
    """
    priced = [c.id for c in config.MODEL_MATRIX if c.fold_reason != "superseded"
              and not cost[(cost.counter_id == c.id)
                           & cost.cost_usd_per_1k_chars.notna()].empty]
    for corpus in ("flores", *config.CORPORA):
        sub = cost[(cost.corpus == corpus) & (cost.lang == LEAD_LANG)]
        rank = {r.counter_id: r.cost_usd_per_1k_chars for r in sub.itertuples()
                if pd.notna(r.cost_usd_per_1k_chars)}
        if rank:
            return sorted([i for i in priced if i in rank], key=lambda i: rank[i])
    return priced


def _cost_legend_lines(cost: pd.DataFrame, counters: list[str]) -> list[str]:
    return [
        f"USD to serve 1,000,000 input characters — input list price "
        f"({_price_as_of(counters)}); VND = USD × {int(config.USD_TO_VND):,}",
        *_same_tokens_diff_price(counters),
        *_unpriced_line(cost, counters),
    ]


def dollar_cost_bars(cost: pd.DataFrame, corpus_name: str,
                     order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
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
    _caption(fig, _cost_legend_lines(cost, counters), ax)
    _save(fig, stem)


# Per-sentence dollar figure — same price × tokens, divided by the corpus
# sentence count instead of characters. Parallel corpus, so this is cost for the
# same *meaning* across languages; the honest unit for a message/document
# workload. Deliberately contrasts with dollar_cost_bars (per character): a dense
# script (Chinese) towers per character yet ranks low per sentence.
_SENTENCE_UNIT = {"flores": "sentence", "massive": "message"}


def _cost_per_sentence_legend_lines(corpus_id: str, counters: list[str]) -> list[str]:
    unit = _SENTENCE_UNIT.get(corpus_id, "sentence")
    return [
        f"USD to serve 1,000 {unit}s — parallel corpus, so the SAME content "
        f"across languages (price × tokens per {unit}); input list price "
        f"({_price_as_of(counters)}); VND = USD × {int(config.USD_TO_VND):,}",
        f"Read against the per-character chart: a dense script (Chinese) needs "
        f"few characters, so per-character overstates its cost; per {unit} it "
        f"ranks far lower.",
        *_same_tokens_diff_price(counters),
    ]


def dollar_cost_per_sentence_bars(cost: pd.DataFrame, corpus_id: str,
                                  corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
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
    _caption(fig, _cost_per_sentence_legend_lines(corpus_id, counters), ax)
    _save(fig, stem)


def cost_driver_bars(cost: pd.DataFrame, agg: pd.DataFrame,
                     corpus_id: str, corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
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


def make_figures() -> None:
    premium = pd.read_csv(config.RESULTS_DIR / "premium_by_language.csv")
    cost = pd.read_csv(config.RESULTS_DIR / "cost_by_language.csv")
    agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")
    # Tokenizer figures share one canonical column order (see _headline_order); the
    # dollar figure has its own (priced options, unfolded — see _priced_order).
    order = _headline_order(premium, sorted(set(premium.counter_id)))
    dollar_order = _priced_order(cost)
    for corpus_id, corpus_name in config.CORPORA.items():
        p = premium[premium.corpus == corpus_id]
        c = cost[cost.corpus == corpus_id]
        if p.empty:
            continue
        premium_heatmap(p, agg, corpus_id, corpus_name, order, f"fig-premium-heatmap-{corpus_id}")
        cost_driver_bars(c, agg, corpus_id, corpus_name, order, f"fig-cost-driver-bars-{corpus_id}")
        dollar_cost_bars(c, corpus_name, dollar_order, f"fig-dollar-cost-{corpus_id}")
        dollar_cost_per_sentence_bars(c, corpus_id, corpus_name, dollar_order,
                                      f"fig-dollar-cost-per-sentence-{corpus_id}")
    print(f"wrote figures to {FIG_DIR}/ (svg + png)")
    print(f"  tokenizer columns: {[_label(c) for c in order]}")
    print(f"  dollar columns:    {[_label(c) for c in dollar_order]}")


if __name__ == "__main__":
    make_figures()
