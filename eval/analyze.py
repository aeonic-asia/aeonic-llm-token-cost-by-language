"""Analysis: premium distribution, tokens-per-1k-chars, and cost tables.

Reads the committed raw dataset (eval/results/*.csv) and writes:
  premium_by_language.csv  aggregate premium + per-sentence distribution
                           (median, p10/p25/p75/p90, mean) per counter × language
  cost_by_language.csv     token density + cost under TWO denominators —
                           per-1k-NFC-chars and per-sentence — in USD and VND
                           where a dated price exists (else blank, never
                           estimated). The two denominators rank languages
                           differently for dense scripts (see _cost_table).
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

from . import config, corpora


def _premium_table(agg: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = config.BASELINE_LANG
    # Per-sentence premium uses the envelope-stripped content count so a
    # wrapped-message counter (Claude) is comparable to the bare-text offline
    # counters; the fixed frame otherwise compresses short-sentence premiums
    # toward 1.0. Older datasets without the column fall back to raw n_tokens.
    tok_col = "n_tokens_content" if "n_tokens_content" in raw.columns else "n_tokens"
    for corpus_id in agg["corpus"].unique():
        ac = agg[agg.corpus == corpus_id]
        rc = raw[raw.corpus == corpus_id]
        for counter_id in ac["counter_id"].unique():
            a = ac[ac.counter_id == counter_id].set_index("lang")
            base_tokens_total = a.loc[base, "total_tokens"]
            r = rc[rc.counter_id == counter_id]
            base_per_sent = (r[r.lang == base]
                             .set_index("sentence_idx")[tok_col])
            for lang, name in config.LANGUAGES.items():
                # aggregate premium (paper method) — envelope-inclusive raw total
                agg_prem = a.loc[lang, "total_tokens"] / base_tokens_total
                # per-sentence premium distribution — envelope-stripped
                lang_per_sent = (r[r.lang == lang]
                                 .set_index("sentence_idx")[tok_col])
                ratios = (lang_per_sent / base_per_sent).replace(
                    [np.inf, -np.inf], np.nan).dropna()
                row = {
                    "corpus": corpus_id, "counter_id": counter_id,
                    "lang": lang, "language": name,
                    "premium_aggregate": round(agg_prem, 4),
                }
                # distribution only when per-sentence data exists (API counters
                # are aggregate-only); otherwise leave the columns blank.
                if len(ratios):
                    q = ratios.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                    row.update({
                        "premium_median": round(float(q[0.50]), 4),
                        "premium_p10": round(float(q[0.10]), 4),
                        "premium_p25": round(float(q[0.25]), 4),
                        "premium_p75": round(float(q[0.75]), 4),
                        "premium_p90": round(float(q[0.90]), 4),
                        "premium_mean": round(float(ratios.mean()), 4),
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def _within_vendor_inflation(agg: pd.DataFrame) -> pd.DataFrame:
    """Absolute token inflation of the newer tokenizer over the older one.

    This is the article's headline hook and a *different* metric from premium:
    tokens(new)/tokens(old) for the SAME language. Premium is a ratio vs.
    English; this is a ratio vs. the previous tokenizer generation. They can
    move in opposite directions (if English inflates fastest, the non-English
    premium falls even as absolute non-English tokens rise). Reported per corpus.
    """
    new_id, old_id = "claude-new", "claude-old"
    if new_id not in set(agg.counter_id) or old_id not in set(agg.counter_id):
        return pd.DataFrame()
    rows = []
    for corpus_id in agg["corpus"].unique():
        sub = agg[agg.corpus == corpus_id]
        if new_id not in set(sub.counter_id) or old_id not in set(sub.counter_id):
            continue
        p = sub.pivot(index="lang", columns="counter_id", values="total_tokens")
        for lang, name in config.LANGUAGES.items():
            old, new = int(p.loc[lang, old_id]), int(p.loc[lang, new_id])
            rows.append({
                "corpus": corpus_id, "lang": lang, "language": name,
                "tokens_old": old, "tokens_new": new,
                "inflation": round(new / old, 4),
                "inflation_pct": round((new / old - 1) * 100, 1),
            })
    return pd.DataFrame(rows)


def _cost_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Cost under two denominators, because they tell different stories.

    * per-1k-NFC-chars — the per-character cost that token density drives. A dense
      script (CJK) reads high here: many tokens per character.
    * per-sentence — cost for the same *meaning*. The corpora are parallel (one
      aligned line per language), so a CJK sentence carries identical content in
      far fewer characters; per sentence it costs far less than the per-character
      view implies. For a serving workload counted in messages/documents this is
      the honest unit, and per-character overstates CJK.

    tokens_per_sentence is the paper-style concatenated total divided by the
    corpus sentence count (mean tokens per sentence) — consistent with the
    aggregate premium, which is built from the same concatenated total.
    """
    n_sent = {c: corpora.corpus_size(c) for c in agg["corpus"].unique()}
    rows = []
    for _, row in agg.iterrows():
        cid = row["counter_id"]
        tokens_per_1k = row["total_tokens"] / row["total_chars"] * 1000
        tokens_per_sent = row["total_tokens"] / n_sent[row["corpus"]]
        price = config.PRICING.get(cid)
        usd_1k = vnd_1k = usd_sent = vnd_sent = None
        if price and price.input_usd_per_mtok is not None:
            usd_per_token = price.input_usd_per_mtok / 1_000_000
            usd_1k = round(tokens_per_1k * usd_per_token, 6)
            vnd_1k = round(usd_1k * config.USD_TO_VND, 2)
            usd_sent = round(tokens_per_sent * usd_per_token, 8)
            vnd_sent = round(usd_sent * config.USD_TO_VND, 4)
        rows.append({
            "corpus": row["corpus"], "counter_id": cid, "lang": row["lang"],
            "language": config.LANGUAGES[row["lang"]],
            "tokens_per_1k_chars": round(tokens_per_1k, 2),
            "tokens_per_sentence": round(tokens_per_sent, 2),
            "price_usd_per_mtok": price.input_usd_per_mtok if price else None,
            "price_confidence": price.confidence if price else "unknown",
            "cost_usd_per_1k_chars": usd_1k,
            "cost_vnd_per_1k_chars": vnd_1k,
            "cost_usd_per_sentence": usd_sent,
            "cost_vnd_per_sentence": vnd_sent,
        })
    return pd.DataFrame(rows)


def _coincidence_check(agg: pd.DataFrame) -> dict[str, list[list[str]]]:
    """Per corpus, counter pairs with identical per-language token totals.

    A real signal: two models that share a tokenizer produce the exact same
    counts on every language. The article claims Fable 5 / Opus 4.8 / Sonnet 5
    share one tokenizer; this verifies it in-dataset. Checked per corpus (a
    genuine shared tokenizer coincides in *both* registers).
    """
    out: dict[str, list[list[str]]] = {}
    for corpus_id in agg["corpus"].unique():
        pivot = (agg[agg.corpus == corpus_id]
                 .pivot(index="lang", columns="counter_id", values="total_tokens"))
        cols = list(pivot.columns)
        shared = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if (pivot[cols[i]] == pivot[cols[j]]).all():
                    shared.append([cols[i], cols[j]])
        out[corpus_id] = shared
    return out


def analyze() -> None:
    agg = pd.read_csv(config.RESULTS_DIR / "aggregate_counts.csv")
    raw = pd.read_csv(config.RESULTS_DIR / "raw_counts.csv")

    premium = _premium_table(agg, raw)
    cost = _cost_table(agg)
    inflation = _within_vendor_inflation(agg)
    premium.to_csv(config.RESULTS_DIR / "premium_by_language.csv", index=False)
    cost.to_csv(config.RESULTS_DIR / "cost_by_language.csv", index=False)
    # Inflation needs both Claude generations. Write it when present; otherwise
    # remove any prior copy so a stale file can't disagree with summary.json
    # (which would show no inflation) after a run that dropped the Claude pair.
    infl_path = config.RESULTS_DIR / "within_vendor_inflation.csv"
    if not inflation.empty:
        inflation.to_csv(infl_path, index=False)
    elif infl_path.exists():
        infl_path.unlink()

    # headline: aggregate premium per corpus × counter × language (contrast langs)
    headline: dict[str, dict] = {}
    for corpus_id in premium.corpus.unique():
        pc = premium[premium.corpus == corpus_id]
        headline[corpus_id] = {}
        for cid in pc.counter_id.unique():
            p = pc[pc.counter_id == cid].set_index("lang")
            headline[corpus_id][cid] = {
                config.LANGUAGES[l]: p.loc[l, "premium_aggregate"]
                for l in config.LANGUAGES if l != config.BASELINE_LANG}

    # oracle: our cl100k vs the paper's Table 1 — the paper used FLORES, so the
    # oracle is defined on the FLORES+ corpus only.
    paper = {"Vietnamese": 2.45, "Chinese (Simplified)": 1.91, "German": 1.58}
    ours = headline.get("flores", {}).get("cl100k_base", {})
    oracle = {name: {"paper": v, "ours": ours.get(name),
                     "delta": None if ours.get(name) is None else round(ours[name] - v, 4)}
              for name, v in paper.items()}

    infl: dict[str, dict] = {}
    if not inflation.empty:
        for _, r in inflation.iterrows():
            infl.setdefault(r["corpus"], {})[r["language"]] = {
                "inflation": r["inflation"], "pct": r["inflation_pct"]}

    summary = {
        "dataset_as_of": config.DATASET_AS_OF,
        "premium_vs": "English (eng_Latn)",
        "corpora": config.CORPORA,
        "headline_premiums_aggregate_by_corpus": headline,
        "within_vendor_claude_inflation_new_over_old_by_corpus": infl,
        "cl100k_oracle_vs_paper_table1_flores": oracle,
        "shared_tokenizer_pairs_by_corpus": _coincidence_check(agg),
        "per_sentence_premium_note": "The per-sentence premium DISTRIBUTION "
                     "(median/p10..p90 in premium_by_language.csv) is measured on "
                     "envelope-stripped content tokens (raw_counts.n_tokens_content): "
                     "Anthropic count_tokens counts a wrapped chat message, so a "
                     "fixed frame (see run_manifest.envelope_tokens_by_counter) is "
                     "subtracted to compare like-for-like with the bare-text offline "
                     "counters. The AGGREGATE premium is the raw paper-style "
                     "concatenated count (frame negligible over ~10^5 tokens), so "
                     "aggregate and median premia are on consistent bare-text bases.",
        "pricing_as_of": config.PRICING_AS_OF,
        "usd_to_vnd": {"rate": config.USD_TO_VND, "as_of": config.USD_TO_VND_AS_OF},
        "cost_note": "token density is measured; USD/VND cost emitted only where "
                     "config.PRICING has a dated price. Forward-dated flagships are "
                     "premium-only until price ratification. Cost is reported under "
                     "two denominators — per-1k-NFC-chars and per-sentence (parallel "
                     "corpus, so per-sentence = same meaning across languages); they "
                     "rank dense scripts (CJK) very differently.",
    }
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # console view — per corpus
    for corpus_id, corpus_name in config.CORPORA.items():
        if corpus_id not in set(premium.corpus):
            continue
        print(f"\nAggregate premium vs. English — {corpus_name}:")
        view = (premium[premium.corpus == corpus_id]
                .pivot(index="language", columns="counter_id", values="premium_aggregate"))
        print(view.to_string())
    if not inflation.empty:
        print("\nWithin-vendor Claude inflation (newer/older tokenizer, same language):")
        for _, r in inflation.iterrows():
            print(f"  [{r['corpus']:7s}] {r['language']:22s} "
                  f"{r['tokens_old']:>7d} -> {r['tokens_new']:>7d}  "
                  f"= {r['inflation']}x  (+{r['inflation_pct']}%)")
    print("\ncl100k oracle vs. paper Table 1 (FLORES+):")
    for name, o in oracle.items():
        print(f"  {name:22s} ours {o['ours']}  paper {o['paper']}  Δ {o['delta']}")
    print(f"\nshared-tokenizer pairs by corpus: {_coincidence_check(agg)}")
    print("wrote premium_by_language.csv, cost_by_language.csv, "
          "within_vendor_inflation.csv, summary.json")


if __name__ == "__main__":
    analyze()
