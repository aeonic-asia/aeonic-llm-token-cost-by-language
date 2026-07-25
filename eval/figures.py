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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# ── design tokens ────────────────────────────────────────────────────────────
# Categorical hues assigned in FIXED slot order and never cycled, so a counter
# keeps its colour as the series set changes. Validated as a set for the light
# chart surface: all inside the lightness band, all above the chroma floor,
# worst adjacent CVD ΔE 9.1 (>=8) and worst adjacent normal-vision ΔE 19.6
# (>=15). Three slots (aqua/yellow/magenta) fall below 3:1 against the surface,
# which obliges "relief" — the committed cost_by_language.csv / premium_by_language.csv
# are that table view, and the extreme in each group is directly labelled.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_SURFACE = "#fcfcfb"     # chart surface
_INK = "#0b0b0b"         # primary ink (titles)
_INK_2 = "#52514e"       # secondary ink (legend, axis titles)
_INK_MUTED = "#898781"   # muted ink (tick labels, captions)
_GRID = "#e1e0d9"        # hairline gridline, one step off the surface
_BASELINE = "#c3c2b7"    # baseline / axis rule


# Colour follows the COUNTER, not its position in whichever chart is being
# drawn. Assigning by list index repainted models between figures — Claude Opus 5
# came out green in the dollar chart and magenta in the cost-driver chart — so a
# reader who learned a colour in one figure was misled by the next.
#
# Slots are read off the tokenizer figures' canonical order (ascending Vietnamese
# premium), which fixes the four counters that appear in BOTH figure families
# (Gemini, GPT-5.6, and the two Claude generations) — those keep one identity
# everywhere. The three that appear ONLY in the dollar figures then reuse the
# slots of three that appear only in the tokenizer figures; `_colors` asserts no
# single chart ever draws one slot twice, so the reuse cannot silently collide.
_SLOT_BY_COUNTER: dict[str, int] = {
    # tokenizer figures, in canonical (ascending Vietnamese premium) order
    "qwen-3-6": 0, "llama-4": 1, "gemini-3-1-pro": 2, "o200k_base": 3,
    "claude-new": 4, "claude-old": 5, "cl100k_base": 6,
    # Dollar-only counters reuse the slots of counters that never appear beside
    # them (Qwen, Llama and cl100k are all unpriced, so they never reach a dollar
    # figure). The specific assignment is not free: the dollar figures render in
    # ascending-cost order, and the validator rejected the obvious mapping —
    # Fable 5 on orange landed it next to Opus 5's magenta at normal-vision
    # ΔE 12.9, under the 15 floor. Moving Fable 5 to violet and Haiku to orange
    # clears every hard gate on the order these bars are actually drawn in.
    # Re-run scripts/validate_palette.js on the RENDERED order before changing
    # these — slot-order validation alone would not have caught it.
    "claude-sonnet-5": 0, "claude-fable-5": 6, "claude-haiku-4-5": 1,
    # superseded/duplicate proxies never reach a chart, but keep them mapped so
    # a future promotion doesn't KeyError.
    "gemini-3-pro": 2, "claude-opus-5": 4,
}


def _colors(counters: list[str]) -> list[str]:
    """Stable hue per counter, and a guard that one chart never repeats a slot.

    Never cycles past the palette: a 9th categorical hue is indistinguishable
    under CVD, so fail loudly and let the caller fold the tail or facet instead.
    """
    missing = [c for c in counters if c not in _SLOT_BY_COUNTER]
    if missing:
        raise ValueError(f"no colour slot assigned for {missing} — add them to "
                         "_SLOT_BY_COUNTER (see the reuse rule above)")
    slots = [_SLOT_BY_COUNTER[c] for c in counters]
    if len(set(slots)) != len(slots):
        dupes = sorted({c for c, s in zip(counters, slots)
                        if slots.count(s) > 1})
        raise ValueError(
            f"counters {dupes} share a colour slot in one chart — the slot-reuse "
            "assumption no longer holds; give them distinct slots")
    if max(slots) >= len(_SERIES):
        raise ValueError(
            f"slot {max(slots)} exceeds the {len(_SERIES)}-slot categorical "
            "palette — fold the tail into 'Other' or facet into small multiples")
    return [_SERIES[s] for s in slots]


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
    x = np.arange(len(groups))
    slot = 0.84 / len(series)
    width = slot * 0.84          # leftover slot = the surface gap
    for k, (cid, color) in enumerate(zip(series, _colors(series))):
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

# Shared caption line for BOTH dollar figures: the shared-tokenizer models that
# cost different amounts to serve (same tokens, different price). Kept in one
# place so the two figures can't drift apart.
_SAME_TOKENS_DIFF_PRICE = (
    "Same tokens, different price: Opus 5 $5 / Opus 4.8 $5 / Sonnet 5 $3 / "
    "Fable 5 $10; Sonnet 4.6 $3 / Haiku 4.5 $1 (per 1M tokens)")


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
        shared = [m for m in folded
                  if m.fold_reason == "shared" and _identical(agg, corpus_id, m.id, flag)
                  # Skip a proxy that names the SAME model as the column itself.
                  # A headline column is labelled with its current flagship, and
                  # that flagship may also have its own counter (measured directly
                  # to confirm the fold). Listing both would render the model twice
                  # in one equality chain ("Claude Opus 5 = ... = Opus 5"), which
                  # reads as a bug. The confirmation still happens — it is just not
                  # restated in a caption that already carries the model's name.
                  and _label(m.id).lower() not in _label(flag).lower()]
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


def _caption(fig, lines: list[str], ax=None) -> None:
    """Footnotes under the plot, anchored to the axes rather than the figure.

    Anchoring to the axes (in offset points below its bottom-left) lets the
    tight bounding box grow to exactly fit the text. The previous version
    reserved a fixed fraction of figure height, which — combined with the tight
    bbox at save time — left a large empty band under every chart.
    """
    if not lines:
        return
    target = ax if ax is not None else fig.axes[0]
    target.annotate("\n".join(lines), xy=(0, 0), xycoords="axes fraction",
                    xytext=(0, -46), textcoords="offset points",
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
    im = ax.imshow(mat, cmap="OrRd", vmin=1.0, aspect="auto")
    ax.set_xticks(range(len(counters)))
    ax.set_xticklabels([_label(c) for c in counters], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([config.LANGUAGES[l] for l in langs], fontsize=9)
    for i in range(len(langs)):
        for j in range(len(counters)):
            ax.text(j, i, f"{mat[i, j]:.2f}×", ha="center", va="center",
                    color="black" if mat[i, j] < mat.max() * 0.75 else "white", fontsize=9)
    ax.set_title(f"Token premium vs. English ({corpus_name})", fontsize=11)
    fig.colorbar(im, ax=ax, label="× English tokens")
    _caption(fig, _legend_lines(agg, corpus_id, counters), ax)
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


def _cost_legend_lines() -> list[str]:
    return [
        f"USD to serve 1,000,000 input characters — input list price "
        f"({config.PRICING_AS_OF}); VND = USD × {int(config.USD_TO_VND):,}",
        _SAME_TOKENS_DIFF_PRICE,
        "Llama 4 and Qwen 3.6 (self-host) and cl100k (2023) omitted — no serving list price",
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
    _caption(fig, _cost_legend_lines(), ax)
    _save(fig, stem)


# Per-sentence dollar figure — same price × tokens, divided by the corpus
# sentence count instead of characters. Parallel corpus, so this is cost for the
# same *meaning* across languages; the honest unit for a message/document
# workload. Deliberately contrasts with dollar_cost_bars (per character): a dense
# script (Chinese) towers per character yet ranks low per sentence.
_SENTENCE_UNIT = {"flores": "sentence", "massive": "message"}


def _cost_per_sentence_legend_lines(corpus_id: str) -> list[str]:
    unit = _SENTENCE_UNIT.get(corpus_id, "sentence")
    return [
        f"USD to serve 1,000 {unit}s — parallel corpus, so the SAME content "
        f"across languages (price × tokens per {unit}); input list price "
        f"({config.PRICING_AS_OF}); VND = USD × {int(config.USD_TO_VND):,}",
        f"Read against the per-character chart: a dense script (Chinese) needs "
        f"few characters, so per-character overstates its cost; per {unit} it "
        f"ranks far lower.",
        _SAME_TOKENS_DIFF_PRICE,
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
    _caption(fig, _cost_per_sentence_legend_lines(corpus_id), ax)
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
