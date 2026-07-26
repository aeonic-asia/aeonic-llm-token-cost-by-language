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
    # Choose the column PER ROW, not once for the frame. Testing only for column
    # presence broke on a mixed-schema carry-forward: rows predating the envelope
    # work get NaN when concatenated with newer ones, the fallback never fires
    # because the column exists, every ratio is NaN, and the distribution columns
    # come out blank — which looks exactly like the legitimate "aggregate-only"
    # state, so a previously published median vanishes with no warning.
    if "n_tokens_content" in raw.columns:
        raw = raw.copy()
        missing = raw["n_tokens_content"].isna()
        if missing.any():
            affected = sorted(raw.loc[missing, "counter_id"].unique())
            print(f"  NOTE  n_tokens_content absent for {affected} (rows predate the "
                  "envelope correction) — falling back to raw n_tokens for those.")
            raw.loc[missing, "n_tokens_content"] = raw.loc[missing, "n_tokens"]
        tok_col = "n_tokens_content"
    else:
        tok_col = "n_tokens"
    # Corpus sizes, for the sample-basis label below. Resolved once; a corpus in
    # the data but no longer registered reports 0, which makes every slice of it a
    # "head slice" rather than a false "full sweep".
    corpus_n: dict[str, int] = {}
    for cid in agg["corpus"].unique():
        try:
            corpus_n[cid] = corpora.corpus_size(cid)
        except KeyError:
            corpus_n[cid] = 0
    for corpus_id in agg["corpus"].unique():
        ac = agg[agg.corpus == corpus_id]
        rc = raw[raw.corpus == corpus_id]
        for counter_id in ac["counter_id"].unique():
            a = ac[ac.counter_id == counter_id].set_index("lang")
            missing_langs = [l for l in config.LANGUAGES if l not in a.index]
            if missing_langs or base not in a.index:
                # A language added to config.LANGUAGES without a full re-measure
                # used to surface as a bare KeyError from deep in the loop.
                raise SystemExit(
                    f"{counter_id} has no rows for {missing_langs or [base]} in "
                    f"corpus '{corpus_id}'. config.LANGUAGES changed without a "
                    f"re-measure — run `make reproduce COUNTERS={counter_id}`, or "
                    "revert the language set.")
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
                        # Sample basis, in the CSV itself. API counters measure a
                        # deterministic 200-sentence HEAD slice while offline
                        # counters sweep all 2009/2033 — previously both landed in
                        # identical percentile columns with nothing to tell them
                        # apart, and this CSV is the extractable table citers read.
                        #
                        # Compare against the corpus, not against a config constant.
                        # The old `>= API_PER_SENTENCE_SUBSAMPLE * 2` heuristic
                        # mislabelled in BOTH directions whenever the constant moved
                        # without a re-measure: at 100 it called a carried 200-row
                        # head slice a "full sweep", and at 1005 it called a genuine
                        # 2009-sentence sweep a head slice. The corpus size is the
                        # only honest reference, and it is already in scope.
                        "n_sentences": int(len(ratios)),
                        "sample_basis": ("full sweep"
                                         if len(ratios) >= corpus_n.get(corpus_id, 0)
                                         else f"head slice [:{len(ratios)}], "
                                              "topically clustered — not a random sample"),
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
            if lang not in p.index:
                continue   # language added since the Claude pair was measured
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
    empty = [c for c, n in n_sent.items() if not n]
    if empty:
        raise SystemExit(f"corpus/corpora {empty} report 0 sentences — cannot form "
                         "a per-sentence denominator. Check the corpus files.")
    rows = []
    for _, row in agg.iterrows():
        cid = row["counter_id"]
        if not row["total_chars"]:
            raise SystemExit(f"{cid}/{row['corpus']}/{row['lang']} has total_chars=0 "
                             "— refusing to divide by zero. The corpus is empty or "
                             "the row is corrupt; re-run `make reproduce`.")
        tokens_per_1k = row["total_tokens"] / row["total_chars"] * 1000
        tokens_per_sent = row["total_tokens"] / n_sent[row["corpus"]]
        price = config.PRICING.get(cid)
        usd_1k = vnd_1k = usd_sent = vnd_sent = None
        if price and price.input_usd_per_mtok is not None:
            usd_per_token = price.input_usd_per_mtok / 1_000_000
            # Round each output once, from the full-precision value. Deriving VND
            # from the already-rounded USD compounded the quantization — at
            # $1/Mtok the 6-dp USD rounding is ~0.4% of the value on its own.
            usd_1k_exact = tokens_per_1k * usd_per_token
            usd_sent_exact = tokens_per_sent * usd_per_token
            usd_1k = round(usd_1k_exact, 6)
            vnd_1k = round(usd_1k_exact * config.USD_TO_VND, 2)
            usd_sent = round(usd_sent_exact, 8)
            vnd_sent = round(usd_sent_exact * config.USD_TO_VND, 4)
        rows.append({
            "corpus": row["corpus"], "counter_id": cid, "lang": row["lang"],
            # `.get` with a visible fallback: a language RETIRED from
            # config.LANGUAGES leaves carried rows behind (carry-forward keys on
            # counters, not languages), and a bare lookup died with an unhelpful
            # KeyError naming only the code.
            "language": config.LANGUAGES.get(row["lang"], f"{row['lang']} (retired)"),
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


def _coincidence_check(agg: pd.DataFrame) -> dict:
    """Counter pairs whose per-language token totals coincide.

    Equality is evidence of a shared tokenizer, not proof of one — two genuinely
    different tokenizers can agree on a corpus that happens not to exercise their
    differences. So a pair is only reported as `shared` when it coincides in
    EVERY corpus, which is what this function's contract always said and what the
    output previously did not enforce: the two Gemini counters agreed on MASSIVE
    and differed on FLORES (by one HTML-bearing sentence) and were published as a
    "shared tokenizer pair" on the strength of the register that happened not to
    contain markup — in the machine-readable file readers are told to cite.

    Pairs that coincide in some corpora but not all are reported separately under
    `coincident_in_some_corpora`, with the corpora named, so the near-miss is
    still visible without being labelled an identity.
    """
    per_corpus: dict[str, set[tuple[str, str]]] = {}
    incomparable: list[dict] = []
    for corpus_id in agg["corpus"].unique():
        pivot = (agg[agg.corpus == corpus_id]
                 .pivot(index="lang", columns="counter_id", values="total_tokens"))
        cols = list(pivot.columns)
        pairs = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = pivot[cols[i]], pivot[cols[j]]
                # Ragged language coverage must not read as inequality. If one
                # counter was re-measured after a language was added (or before one
                # was retired) its partner's cell is NaN, and NaN == NaN is False —
                # so a genuine shared-tokenizer pair silently dropped out of
                # `shared_in_all_corpora` and did not even appear as a near-miss,
                # taking the article's "identical counts (verified here)" evidence
                # with it. Compare only where both are present, and refuse to
                # conclude anything when the covered sets differ.
                both = a.notna() & b.notna()
                if a.notna().sum() != b.notna().sum() or not both.all():
                    if both.any() and (a[both] == b[both]).all():
                        incomparable.append({
                            "pair": [cols[i], cols[j]], "corpus": corpus_id,
                            "identical_on": sorted(map(str, pivot.index[both])),
                            "reason": "counters do not cover the same language set; "
                                      "identical where both are present, but this is "
                                      "not evidence of a shared tokenizer",
                        })
                    continue
                if (a == b).all():
                    pairs.add((cols[i], cols[j]))
        per_corpus[corpus_id] = pairs

    corpora_ids = list(per_corpus)
    everywhere = set.intersection(*per_corpus.values()) if per_corpus else set()
    somewhere = set.union(*per_corpus.values()) if per_corpus else set()
    partial = sorted(somewhere - everywhere)
    return {
        "definition": "a pair is 'shared' only if its token totals are identical "
                      "in every corpus; equality in one register is not identity",
        "corpora_checked": corpora_ids,
        "shared_in_all_corpora": [list(p) for p in sorted(everywhere)],
        "coincident_in_some_corpora": [
            {"pair": list(p),
             "corpora": sorted(c for c, s in per_corpus.items() if p in s)}
            for p in partial],
        # Pairs that could not be compared because the two counters cover
        # different language sets — surfaced rather than silently treated as
        # "not identical".
        "incomparable_coverage": incomparable,
        # Retained per corpus for auditability — this is the raw equality result
        # the `shared` verdict is derived from, not a finding in its own right.
        "raw_equality_by_corpus": {c: [list(p) for p in sorted(s)]
                                   for c, s in per_corpus.items()},
    }


def _coverage(agg: pd.DataFrame, oracle: dict) -> dict:
    """Provenance the headline numbers depend on, carried INTO summary.json.

    summary.json is the file CLAUDE.md points readers at first and the one built
    to be machine-extracted, yet it used to carry a single `dataset_as_of` and no
    indication of which counters were actually measured in that pass versus
    carried forward from an earlier one, nor whether the oracle passed. A citer
    could read one date and cite a count last measured weeks earlier against a
    different endpoint. The manifest had the facts; the summary did not read it.
    """
    manifest_path = config.RESULTS_DIR / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    # float() the deltas: they arrive as numpy scalars, and a numpy bool_ derived
    # from them is not JSON-serializable.
    if not manifest:
        # An absent manifest yields empty counter lists and no versions — which is
        # indistinguishable from a genuine "nothing measured" state in the very
        # block CLAUDE.md calls the one that says what the rest of the file is
        # worth. Say so out loud rather than emitting a confident-looking vacuum.
        print("  WARN  run_manifest.json is absent — summary.json's `coverage` block "
              "will report no measured counters and no library versions. That is a "
              "missing file, not a finding. Restore it with "
              "`git checkout -- eval/results/run_manifest.json`.")
    deltas = [float(abs(o["delta"])) for o in oracle.values() if o.get("delta") is not None]
    return {
        "counters_in_dataset": sorted(set(agg.counter_id)),
        "counters_measured_last_pass": manifest.get("counters_ran", []),
        "counters_carried_forward": manifest.get("counters_carried_forward", []),
        "counters_skipped": [s.get("counter") for s in manifest.get("counters_skipped", [])],
        "measured_on_by_counter": manifest.get("measured_on_by_counter", {}),
        "corpora_fingerprint": manifest.get("corpora_fingerprint", {}),
        "manifest_present": bool(manifest),
        "versions": manifest.get("versions", {}),
        "oracle_max_abs_delta": round(max(deltas), 6) if deltas else None,
        "oracle_tolerance": config.ORACLE_TOL,
        "oracle_passes": bool(max(deltas) < config.ORACLE_TOL) if deltas else None,
        "note": "`counters_carried_forward` were NOT re-measured in the last pass; "
                "their counts come from an earlier one. Cross-check "
                "`measured_on_by_counter` before citing a specific counter's date.",
    }


def _check_corpora_unchanged() -> None:
    """Refuse to analyze rows against a corpus that has changed since measurement.

    `_cost_table` divides carried token totals by a corpus size re-read from disk,
    so a rebuilt or edited corpus silently re-normalizes every per-sentence cost of
    every counter that was NOT re-measured — exit 0, no warning, wrong numbers in a
    published figure. run.py records a fingerprint per corpus; this is the reader
    side of that contract.
    """
    manifest_path = config.RESULTS_DIR / "run_manifest.json"
    if not manifest_path.exists():
        return
    try:
        recorded = json.loads(manifest_path.read_text()).get("corpora_fingerprint", {})
    except (json.JSONDecodeError, OSError):
        return
    if not recorded:
        return   # dataset predates fingerprinting; nothing to compare against
    langs = list(config.LANGUAGES)
    for cid, was in recorded.items():
        if cid not in config.CORPORA:
            continue
        now = corpora.corpus_fingerprint(cid, langs)
        if was.get("sha256") != now["sha256"]:
            raise SystemExit(
                f"corpus '{cid}' has changed since the dataset was measured "
                f"({was.get('n_sentences')} sentences -> {now['n_sentences']}). "
                "Per-sentence and per-character denominators are read live, so "
                "analyzing would re-normalize every carried counter against a corpus "
                "it was never measured on. Re-measure "
                f"(`make reproduce CORPORA={cid}`) or restore the corpus.")


def analyze() -> None:
    _check_corpora_unchanged()
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
    # oracle is defined on the FLORES+ corpus only. Resolved by LANGUAGE CODE from
    # the shared constant in config (tests/test_oracle.py reads the same one), then
    # rendered under the display name. Keying the lookup on the display string meant
    # renaming a language dropped it from the gate, which then reported a tighter
    # max delta over two of three checks and still passed.
    # Computed from the unrounded totals, not from premium_aggregate: that column
    # is rounded to 4 dp, so a true delta of 0.00496 was presented to the gate as
    # 0.0050 and failed a tolerance it actually met.
    ours_by_code = {}
    fl_agg = agg[(agg.corpus == "flores") & (agg.counter_id == "cl100k_base")]
    if not fl_agg.empty and config.BASELINE_LANG in set(fl_agg.lang):
        base_total = float(fl_agg.loc[fl_agg.lang == config.BASELINE_LANG,
                                      "total_tokens"].iloc[0])
        for _, r in fl_agg.iterrows():
            ours_by_code[r["lang"]] = float(r["total_tokens"]) / base_total
    oracle = {config.LANGUAGES[code]: {
                  "paper": v,
                  "ours": None if ours_by_code.get(code) is None
                  else round(ours_by_code[code], 4),
                  "delta": None if ours_by_code.get(code) is None
                  else round(ours_by_code[code] - v, 6)}
              for code, v in config.PAPER_CL100K_FLORES.items()}
    absent = sorted(c for c in config.PAPER_CL100K_FLORES if c not in ours_by_code)
    if absent:
        # Never let a shrunken oracle report a better margin than a full one.
        raise SystemExit(
            f"oracle cannot be evaluated: cl100k_base has no FLORES premium for "
            f"{absent}. The validation gate would silently check fewer languages "
            "and report a tighter delta. Re-run `make reproduce COUNTERS=cl100k_base`.")

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
        # Renamed from `shared_tokenizer_pairs_by_corpus`: the old key promised an
        # identity finding while holding a per-corpus equality result, and a
        # machine reading it could not tell the two apart.
        "tokenizer_coincidence_check": _coincidence_check(agg),
        "coverage": _coverage(agg, oracle),
        "per_sentence_premium_note": "The per-sentence premium DISTRIBUTION "
                     "(median/p10..p90 in premium_by_language.csv) is measured on "
                     "envelope-stripped content tokens (raw_counts.n_tokens_content): "
                     "Anthropic count_tokens counts a wrapped chat message, so a "
                     "fixed frame (see run_manifest.envelope_tokens_by_counter) is "
                     "subtracted to compare like-for-like with the bare-text offline "
                     "counters. The AGGREGATE premium is the raw paper-style "
                     "concatenated count and is left envelope-INCLUSIVE, so the two "
                     "are on near-identical rather than identical bases. The frame's "
                     "share of the call is corpus-dependent — 0.0073%/0.0124% of the "
                     "English total on FLORES (newer/older Claude) but 0.0286%/0.0463% "
                     "on MASSIVE, whose totals are ~1.5-2.1x10^4 tokens rather than "
                     "~10^5. Worst resulting bias on a published aggregate premium is "
                     "+0.0007 (massive/claude-old/vie), i.e. below the 4th decimal.",
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
                  # sign from the value, not hardcoded: deflation printed "(+-3.2%)"
                  f"= {r['inflation']}x  ({r['inflation_pct']:+}%)")
    print("\ncl100k oracle vs. paper Table 1 (FLORES+):")
    for name, o in oracle.items():
        print(f"  {name:22s} ours {o['ours']}  paper {o['paper']}  Δ {o['delta']}")
    _coin = _coincidence_check(agg)
    print(f"\nshared tokenizers (identical in ALL corpora): "
          f"{_coin['shared_in_all_corpora']}")
    if _coin["coincident_in_some_corpora"]:
        print(f"  coincident in some corpora only (NOT an identity): "
              f"{_coin['coincident_in_some_corpora']}")
    print("wrote premium_by_language.csv, cost_by_language.csv, "
          "within_vendor_inflation.csv, summary.json")


if __name__ == "__main__":
    analyze()
