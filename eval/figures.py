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

The "cost bars" show tokens-per-1,000-NFC-characters — the measured, price-
independent driver of per-character cost.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
# Determinism: pin the SVG element-id hash salt (else matplotlib re-randomises
# clip-path / marker ids every run) so re-runs are byte-identical. The wall-clock
# <dc:date> is stripped per-save below. Together these keep `make reproduce` from
# churning the committed figures when only the timestamp/ids would differ.
matplotlib.rcParams["svg.hashsalt"] = "aeonic-token-cost-eval"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIG_DIR = config.RESULTS_DIR / "figures"
CONTRAST = [l for l in config.LANGUAGES if l != config.BASELINE_LANG]
LEAD_LANG = "vie_Latn"  # article lead; canonical column order sorts by its premium


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


def _legend_lines(agg: pd.DataFrame, corpus_id: str, shown: list[str]) -> list[str]:
    """Human-readable fold notes for the columns actually shown, in column order.
    A "shared" fold is emitted only if re-verified byte-identical in this corpus."""
    lines: list[str] = []
    for flag in shown:
        folded = [c for c in config.MODEL_MATRIX if c.flagship_group == flag]
        shared = [m for m in folded
                  if m.fold_reason == "shared" and _identical(agg, corpus_id, m.id, flag)]
        superseded = [m for m in folded if m.fold_reason == "superseded"]
        if shared:
            names = " = ".join([_label(flag)] + [_label(m.id) for m in shared])
            note = f"{names} — one shared tokenizer, identical counts (verified here)"
            if flag == "claude-new":
                note += "; prices differ (Sonnet 5 cheapest)"
            lines.append(note)
        for m in superseded:
            lines.append(
                f"{_label(flag)} shown (current flagship); "
                f"{_label(m.id)} superseded, within ~0.01% — omitted")
        if flag == "cl100k_base":
            lines.append(f"{_label(flag)} — GPT-4/3.5-era baseline, historical anchor")
    return lines


def _caption(fig, lines: list[str]) -> None:
    if not lines:
        return
    fig.text(0.01, 0.005, "\n".join(lines), ha="left", va="bottom",
             fontsize=7.5, color="#555", linespacing=1.4)
    # reserve room so the caption doesn't collide with the x-axis labels
    fig.subplots_adjust(bottom=0.30 + 0.03 * len(lines))


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
    _caption(fig, _legend_lines(agg, corpus_id, counters))
    _save(fig, stem)


def cost_driver_bars(cost: pd.DataFrame, agg: pd.DataFrame,
                     corpus_id: str, corpus_name: str, order: list[str], stem: str) -> None:
    counters = [c for c in order if c in set(cost.counter_id)]
    langs = list(config.LANGUAGES)
    x = np.arange(len(langs))
    width = 0.8 / len(counters)

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, c in enumerate(counters):
        vals = [cost[(cost.counter_id == c) & (cost.lang == l)]
                ["tokens_per_1k_chars"].iloc[0] for l in langs]
        ax.bar(x + k * width, vals, width, label=_label(c))
    ax.set_xticks(x + width * (len(counters) - 1) / 2)
    ax.set_xticklabels([config.LANGUAGES[l] for l in langs], rotation=15, ha="right")
    ax.set_ylabel("Tokens per 1,000 NFC characters")
    ax.set_title(f"Cost driver: tokens per 1,000 characters — {corpus_name} (lower = cheaper)")
    ax.legend(fontsize=9, ncol=2)
    _caption(fig, _legend_lines(agg, corpus_id, counters))
    _save(fig, stem)


def make_figures() -> None:
    premium = pd.read_csv(config.RESULTS_DIR / "premium_by_language.csv")
    cost = pd.read_csv(config.RESULTS_DIR / "cost_by_language.csv")
    agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")
    # One canonical column order, shared by every figure (see _headline_order).
    order = _headline_order(premium, sorted(set(premium.counter_id)))
    # One heatmap + one cost-bar chart per corpus (the premium differs by register).
    for corpus_id, corpus_name in config.CORPORA.items():
        p = premium[premium.corpus == corpus_id]
        c = cost[cost.corpus == corpus_id]
        if p.empty:
            continue
        premium_heatmap(p, agg, corpus_id, corpus_name, order, f"fig-premium-heatmap-{corpus_id}")
        cost_driver_bars(c, agg, corpus_id, corpus_name, order, f"fig-cost-driver-bars-{corpus_id}")
    print(f"wrote figures to {FIG_DIR}/ (svg + png); columns: {[_label(c) for c in order]}")


if __name__ == "__main__":
    make_figures()
