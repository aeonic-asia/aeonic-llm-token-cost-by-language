"""Analysis: premium distribution, tokens-per-1k-chars, and cost tables.

Reads the committed raw dataset (eval/results/*.csv) and writes:
  premium_by_language.csv  aggregate premium + per-sentence distribution
                           (median, p10/p25/p75/p90, mean) per counter × language
  cost_by_language.csv     tokens-per-1k-NFC-chars (real) + cost-per-1k-chars in
                           USD and VND where a dated price exists (else blank —
                           never estimated)
  summary.json            headline premiums, oracle check, shared-tokenizer
                           coincidence check, and provenance

The premium is always measured vs. English. The token side of cost is real
measurement; the dollar side comes from config.PRICING and is emitted only when
present, so an unpriced (e.g. forward-dated) model reports premium-only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def _premium_table(agg: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = config.BASELINE_LANG
    for counter_id in agg["counter_id"].unique():
        a = agg[agg.counter_id == counter_id].set_index("lang")
        base_tokens_total = a.loc[base, "total_tokens"]
        r = raw[raw.counter_id == counter_id]
        base_per_sent = (r[r.lang == base]
                         .set_index("sentence_idx")["n_tokens"])
        for lang, name in config.LANGUAGES.items():
            # aggregate premium (paper method)
            agg_prem = a.loc[lang, "total_tokens"] / base_tokens_total
            # per-sentence premium distribution
            lang_per_sent = (r[r.lang == lang]
                             .set_index("sentence_idx")["n_tokens"])
            ratios = (lang_per_sent / base_per_sent).replace(
                [np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "counter_id": counter_id, "lang": lang, "language": name,
                "premium_aggregate": round(agg_prem, 4),
                "premium_median": round(float(ratios.median()), 4),
                "premium_p10": round(float(ratios.quantile(0.10)), 4),
                "premium_p25": round(float(ratios.quantile(0.25)), 4),
                "premium_p75": round(float(ratios.quantile(0.75)), 4),
                "premium_p90": round(float(ratios.quantile(0.90)), 4),
                "premium_mean": round(float(ratios.mean()), 4),
            })
    return pd.DataFrame(rows)


def _cost_table(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in agg.iterrows():
        cid = row["counter_id"]
        tokens_per_1k = row["total_tokens"] / row["total_chars"] * 1000
        price = config.PRICING.get(cid)
        usd = vnd = None
        if price and price.input_usd_per_mtok is not None:
            usd_per_token = price.input_usd_per_mtok / 1_000_000
            usd = round(tokens_per_1k * usd_per_token, 6)
            vnd = round(usd * config.USD_TO_VND, 2)
        rows.append({
            "counter_id": cid, "lang": row["lang"],
            "language": config.LANGUAGES[row["lang"]],
            "tokens_per_1k_chars": round(tokens_per_1k, 2),
            "price_usd_per_mtok": price.input_usd_per_mtok if price else None,
            "price_confidence": price.confidence if price else "unknown",
            "cost_usd_per_1k_chars": usd,
            "cost_vnd_per_1k_chars": vnd,
        })
    return pd.DataFrame(rows)


def _coincidence_check(agg: pd.DataFrame) -> list[list[str]]:
    """Counter pairs with identical per-language token totals → shared tokenizer.

    A real signal in this data: two models that share a tokenizer produce the
    exact same counts on every language. The article claims Fable 5 / Opus 4.8 /
    Sonnet 5 share one tokenizer; this is the machinery that verifies such
    claims once those counters are run.
    """
    pivot = agg.pivot(index="lang", columns="counter_id", values="total_tokens")
    cols = list(pivot.columns)
    shared = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if (pivot[cols[i]] == pivot[cols[j]]).all():
                shared.append([cols[i], cols[j]])
    return shared


def analyze() -> None:
    agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")
    raw = pd.read_csv(config.RESULTS_DIR / "raw_counts.csv")

    premium = _premium_table(agg, raw)
    cost = _cost_table(agg)
    premium.to_csv(config.RESULTS_DIR / "premium_by_language.csv", index=False)
    cost.to_csv(config.RESULTS_DIR / "cost_by_language.csv", index=False)

    # headline: aggregate premium per counter × language (contrast langs only)
    headline = {}
    for cid in premium.counter_id.unique():
        p = premium[premium.counter_id == cid].set_index("lang")
        headline[cid] = {config.LANGUAGES[l]: p.loc[l, "premium_aggregate"]
                         for l in config.LANGUAGES if l != config.BASELINE_LANG}

    # oracle: our cl100k vs the paper's Table 1
    paper = {"Vietnamese": 2.45, "Chinese (Simplified)": 1.91, "German": 1.58}
    ours = headline.get("cl100k_base", {})
    oracle = {name: {"paper": v, "ours": ours.get(name),
                     "delta": None if ours.get(name) is None else round(ours[name] - v, 4)}
              for name, v in paper.items()}

    summary = {
        "dataset_as_of": config.PRICING_AS_OF,
        "premium_vs": "English (eng_Latn)",
        "headline_premiums_aggregate": headline,
        "cl100k_oracle_vs_paper_table1": oracle,
        "shared_tokenizer_pairs": _coincidence_check(agg),
        "pricing_as_of": config.PRICING_AS_OF,
        "usd_to_vnd": {"rate": config.USD_TO_VND, "as_of": config.USD_TO_VND_AS_OF},
        "cost_note": "tokens_per_1k_chars is measured; USD/VND cost emitted only "
                     "where config.PRICING has a dated price. Forward-dated "
                     "flagships are premium-only until price ratification (5-3).",
    }
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # console view
    print("\nAggregate premium vs. English:")
    view = premium.pivot(index="language", columns="counter_id", values="premium_aggregate")
    print(view.to_string())
    print("\ncl100k oracle vs. paper Table 1:")
    for name, o in oracle.items():
        print(f"  {name:22s} ours {o['ours']}  paper {o['paper']}  Δ {o['delta']}")
    print(f"\nwrote premium_by_language.csv, cost_by_language.csv, summary.json")


if __name__ == "__main__":
    analyze()
