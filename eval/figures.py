"""Figures for the article: premium heatmap + per-language cost-driver bars.

Exports SVG (for later fig-NN-*.svg article assets) and PNG (quick view) to
eval/results/figures/. Driven entirely by the committed result CSVs, so figures
regenerate deterministically from the dataset.

The "cost bars" show tokens-per-1,000-NFC-characters — the measured, price-
independent driver of per-character cost. A dollar-denominated variant slots in
once config.PRICING is ratified; until then the figure reports the real token
side rather than an invented dollar figure.
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


def _save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        # Drop the wall-clock Date from SVG metadata so re-runs are byte-identical.
        kw = {"metadata": {"Date": None}} if ext == "svg" else {}
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=150, **kw)
    plt.close(fig)


def premium_heatmap(premium: pd.DataFrame) -> None:
    counters = list(premium.counter_id.unique())
    langs = CONTRAST
    mat = np.array([[premium[(premium.counter_id == c) & (premium.lang == l)]
                     ["premium_aggregate"].iloc[0] for c in counters] for l in langs])

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(counters), 0.7 + 0.6 * len(langs)))
    im = ax.imshow(mat, cmap="OrRd", vmin=1.0, aspect="auto")
    ax.set_xticks(range(len(counters)))
    ax.set_xticklabels([config.MATRIX_BY_ID[c].display for c in counters],
                       rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([config.LANGUAGES[l] for l in langs], fontsize=9)
    for i in range(len(langs)):
        for j in range(len(counters)):
            ax.text(j, i, f"{mat[i, j]:.2f}×", ha="center", va="center",
                    color="black" if mat[i, j] < mat.max() * 0.75 else "white", fontsize=9)
    ax.set_title("Token premium vs. English (FLORES+)", fontsize=11)
    fig.colorbar(im, ax=ax, label="× English tokens")
    _save(fig, "fig-premium-heatmap")


def cost_driver_bars(cost: pd.DataFrame) -> None:
    counters = list(cost.counter_id.unique())
    langs = list(config.LANGUAGES)
    x = np.arange(len(langs))
    width = 0.8 / len(counters)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, c in enumerate(counters):
        vals = [cost[(cost.counter_id == c) & (cost.lang == l)]
                ["tokens_per_1k_chars"].iloc[0] for l in langs]
        ax.bar(x + k * width, vals, width, label=config.MATRIX_BY_ID[c].display)
    ax.set_xticks(x + width * (len(counters) - 1) / 2)
    ax.set_xticklabels([config.LANGUAGES[l] for l in langs], rotation=15, ha="right")
    ax.set_ylabel("Tokens per 1,000 NFC characters")
    ax.set_title("Cost driver: tokens per 1,000 characters (lower = cheaper)")
    ax.legend(fontsize=9)
    _save(fig, "fig-cost-driver-bars")


def make_figures() -> None:
    premium = pd.read_csv(config.RESULTS_DIR / "premium_by_language.csv")
    cost = pd.read_csv(config.RESULTS_DIR / "cost_by_language.csv")
    premium_heatmap(premium)
    cost_driver_bars(cost)
    print(f"wrote figures to {FIG_DIR}/ (svg + png)")


if __name__ == "__main__":
    make_figures()
