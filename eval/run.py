"""Driver: run the matrix over the corpora and write the committed raw dataset.

Outputs (eval/results/):
  raw_counts.csv       long/tidy: corpus, counter_id, lang, sentence_idx,
                       n_tokens (raw counter output), n_tokens_content
                       (n_tokens minus the message envelope — the bare-text
                       count, source for the premium *distribution*),
                       envelope_tokens (the frame subtracted; 0 for offline
                       counters), n_chars
  aggregate_counts.csv corpus, counter_id, lang, total_tokens, total_chars
                       (paper-style concatenated count — source for the aggregate
                       premium and cost-per-1k-chars)
  run_manifest.json    which counters ran vs. skipped (and why), which corpora
                       were measured vs. carried forward, versions, "as of" date.

Corpus selection + carry-forward. By default all corpora in config.CORPORA are
measured. Set `EVAL_CORPORA=massive` (comma-separated) to measure only a subset;
rows for the corpora NOT selected are carried forward verbatim from the existing
committed CSVs. This avoids needlessly re-measuring a deterministic-but-key-gated
corpus (e.g. the Claude FLORES counts) when adding a new one — and preserves
those numbers byte-for-byte. A full `make reproduce` (no selection) is still a
clean from-scratch rebuild.

Determinism: token counts are deterministic and the manifest carries no
wall-clock timestamp (it uses config's dated `as_of`), so a full re-run produces
byte-identical files — `make reproduce` yields no git churn.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import config
from .corpora import load_corpus
from .measure import build_counter, char_count


def _selected_corpora() -> list[str]:
    """Corpora to measure this pass (env `EVAL_CORPORA`, default: all)."""
    raw = os.environ.get("EVAL_CORPORA", "").strip()
    if not raw:
        return list(config.CORPORA)
    sel = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in sel if c not in config.CORPORA]
    if unknown:
        raise SystemExit(f"unknown corpora in EVAL_CORPORA: {unknown}; "
                         f"known: {list(config.CORPORA)}")
    return sel


def _selected_counters() -> list[str]:
    """Counter ids to measure this pass (env `EVAL_COUNTERS`, default: all).

    The counter analogue of `EVAL_CORPORA`: measure only these counters and
    carry the rest forward verbatim from the committed dataset. Same rationale
    as corpus carry-forward — adding a new counter (Gemini, Llama) must not force
    a re-measure of the deterministic-but-key-gated Claude rows, which would risk
    perturbing numbers that should stay byte-identical.
    """
    raw = os.environ.get("EVAL_COUNTERS", "").strip()
    known = [c.id for c in config.MODEL_MATRIX]
    if not raw:
        return known
    sel = [c.strip() for c in raw.split(",") if c.strip()]
    unknown = [c for c in sel if c not in known]
    if unknown:
        raise SystemExit(f"unknown counters in EVAL_COUNTERS: {unknown}; "
                         f"known: {known}")
    return sel


def run() -> None:
    langs = list(config.LANGUAGES)
    selected = _selected_corpora()
    selected_counters = _selected_counters()
    # Load only the selected corpora (each asserts its own line-alignment).
    corpora = {cid: load_corpus(cid, langs) for cid in selected}
    n_by_corpus = {cid: len(c[langs[0]]) for cid, c in corpora.items()}

    # Carry-forward preserves every committed (corpus × counter) cell that is NOT
    # successfully re-measured this pass. Load the committed dataset here; the
    # carried set is computed AFTER the run because it keys on what actually
    # *ran* (see below) — not on the mere selection. Loading regardless of subset
    # is what makes a keyless/tokenless `make reproduce` CONVERGE on the committed
    # dataset (preserving the rows it can't re-measure) instead of erasing them.
    raw_path = config.RESULTS_DIR / "raw_counts.csv"
    agg_path = config.RESULTS_DIR / "aggregate_counts.csv"
    ex_raw = ex_agg = None
    if raw_path.exists() and agg_path.exists():
        ex_raw, ex_agg = pd.read_csv(raw_path), pd.read_csv(agg_path)
        for df, nm in ((ex_raw, "raw_counts.csv"), (ex_agg, "aggregate_counts.csv")):
            if "corpus" not in df.columns:
                raise SystemExit(f"existing {nm} lacks a 'corpus' column — delete "
                                 "eval/results/ and run a full `make reproduce` once")
    is_subset = (set(selected) != set(config.CORPORA)
                 or set(selected_counters) != {c.id for c in config.MODEL_MATRIX})
    if is_subset and ex_raw is None:
        raise SystemExit(
            "EVAL_CORPORA/EVAL_COUNTERS selects a subset, but there is no committed "
            "dataset to carry the rest forward from — run a full `make reproduce` "
            "first (or unset the selectors).")

    # Build only the selected counters, then run each over every selected corpus.
    # Non-selected counters are carried forward (below), not skipped-for-cause.
    built: list[tuple] = []
    skipped: list[dict] = []
    for spec in config.MODEL_MATRIX:
        if spec.id not in selected_counters:
            continue
        counter, reason = build_counter(spec)
        if counter is None:
            skipped.append({"counter": spec.id, "status": spec.status, "reason": reason})
            print(f"  skip  {spec.id:16s} — {reason}")
            continue
        built.append((spec, counter))

    per_sentence: list[dict] = []
    aggregate: list[dict] = []
    ran: list[str] = []
    envelope_by_counter: dict[str, int] = {}

    for spec, counter in built:
        ran.append(spec.id)
        # API counters are rate-limited: aggregate always, per-sentence only up
        # to a subsample. Offline counters do the full per-sentence sweep.
        is_api = spec.kind in config.API_KINDS
        note = " [aggregate + %d-sentence subsample/corpus]" % config.API_PER_SENTENCE_SUBSAMPLE
        print(f"  run   {spec.id:16s} ({spec.display}){note if is_api else ''}")
        try:
            # Message-envelope floor (0 for bare-text counters): subtract it from
            # per-sentence counts so a wrapped-message counter (Claude) is
            # comparable to the offline counters. Measured once per counter.
            envelope = counter.envelope_tokens()
            envelope_by_counter[spec.id] = envelope
            for cid, corpus in corpora.items():
                max_per_sentence = (config.API_PER_SENTENCE_SUBSAMPLE
                                    if is_api else n_by_corpus[cid])
                for lang in langs:
                    sentences = corpus[lang]
                    # aggregate: paper-style — concatenate, count once. Left
                    # envelope-inclusive: the fixed frame is <0.01% of a single
                    # ~10^5-token call, so it is negligible and this stays the
                    # raw paper-style total.
                    joined = " ".join(sentences)
                    aggregate.append({
                        "corpus": cid, "counter_id": spec.id, "lang": lang,
                        "total_tokens": counter.count(joined),
                        "total_chars": char_count(joined),
                    })
                    # per-sentence: for the premium distribution. n_tokens is the
                    # raw counter output; n_tokens_content strips the envelope so
                    # the distribution compares like with like.
                    for idx, sent in enumerate(sentences[:max_per_sentence]):
                        raw_tok = counter.count(sent)
                        per_sentence.append({
                            "corpus": cid, "counter_id": spec.id, "lang": lang,
                            "sentence_idx": idx,
                            "n_tokens": raw_tok,
                            "n_tokens_content": raw_tok - envelope,
                            "envelope_tokens": envelope,
                            "n_chars": char_count(sent),
                        })
        except Exception as exc:  # noqa: BLE001 — one bad counter must not kill the run
            # roll back any partial rows for this counter (across all corpora)
            aggregate[:] = [r for r in aggregate if r["counter_id"] != spec.id]
            per_sentence[:] = [r for r in per_sentence if r["counter_id"] != spec.id]
            envelope_by_counter.pop(spec.id, None)
            ran.remove(spec.id)
            skipped.append({"counter": spec.id, "status": spec.status,
                            "reason": f"errored at run: {type(exc).__name__}: {exc}"})
            print(f"  ERROR {spec.id:16s} — {type(exc).__name__}: {exc}")
            continue

    # Carry forward every committed cell NOT successfully measured this pass —
    # keyed on `ran`, so a counter that was skipped for cause, errored mid-run, or
    # simply not selected keeps its committed rows rather than being dropped.
    carried_raw = carried_agg = None
    carried_corpora: list[str] = []
    carried_counters: list[str] = []
    if ex_raw is not None:
        def _carry(df):
            measured = df.corpus.isin(selected) & df.counter_id.isin(ran)
            return df[~measured]
        carried_raw, carried_agg = _carry(ex_raw), _carry(ex_agg)
        carried_counters = sorted(set(carried_agg.counter_id) - set(ran))
        carried_corpora = sorted(set(carried_agg.corpus) - set(selected))

    new_raw, new_agg = pd.DataFrame(per_sentence), pd.DataFrame(aggregate)
    final_raw = (pd.concat([carried_raw, new_raw], ignore_index=True)
                 if carried_raw is not None and not carried_raw.empty else new_raw)
    final_agg = (pd.concat([carried_agg, new_agg], ignore_index=True)
                 if carried_agg is not None and not carried_agg.empty else new_agg)

    # Safety: never overwrite the committed dataset with an empty result (e.g.
    # every counter skipped on a fresh checkout with no keys/SDK) — that would
    # destroy the committed data the run was meant to preserve.
    if final_agg.empty:
        raise SystemExit("measured 0 counters and nothing to carry forward — "
                         "refusing to write an empty dataset (check ANTHROPIC_API_KEY / "
                         "HF_TOKEN / google-genai, or `make clean` for a true reset).")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_raw.to_csv(raw_path, index=False)
    final_agg.to_csv(agg_path, index=False)

    import tiktoken
    manifest = {
        "dataset_as_of": config.DATASET_AS_OF,
        "corpora": {cid: name for cid, name in config.CORPORA.items()},
        "corpora_measured_this_pass": {cid: n_by_corpus[cid] for cid in selected},
        "corpora_carried_forward": carried_corpora,
        "languages": config.LANGUAGES,
        "baseline_language": config.BASELINE_LANG,
        "counters_ran": ran,
        "counters_carried_forward": carried_counters,
        "counters_skipped": skipped,
        "envelope_tokens_by_counter": envelope_by_counter,
        "versions": {"tiktoken": tiktoken.__version__, "pandas": pd.__version__},
        "notes": "Dual-register: FLORES+ (formal) + MASSIVE (assistant utterances); "
                 "premium reported per corpus. Anthropic counters run when "
                 "ANTHROPIC_API_KEY is present (aggregate + per-sentence subsample "
                 "per corpus); Gemini via offline google-genai LocalTokenizer (no "
                 "key); Llama 4 via HF AutoTokenizer (gated repo, needs HF_TOKEN). "
                 "`corpora_carried_forward` / `counters_carried_forward` list "
                 "slices preserved verbatim from the prior dataset (not re-measured "
                 "this pass); the measured set is their complement. "
                 "`envelope_tokens_by_counter` is the per-message chat frame "
                 "(measured, 0 for bare-text counters) subtracted from "
                 "raw_counts.n_tokens_content so the per-sentence premium "
                 "distribution compares Claude's wrapped counts like-for-like with "
                 "the offline counters; the aggregate is left envelope-inclusive. "
                 "API per-sentence rows are a DETERMINISTIC head-slice "
                 "(sentences[:api_per_sentence_subsample] of the FLORES+ dev split / "
                 "MASSIVE), not a random sample — so its percentiles describe that "
                 "fixed slice; offline counters do the full per-sentence sweep.",
        "api_per_sentence_subsample": config.API_PER_SENTENCE_SUBSAMPLE,
    }
    with open(config.RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  measured {len(ran)} counter(s) over corpora "
          f"[{', '.join(f'{c}={n}' for c, n in n_by_corpus.items())}]; "
          f"carried corpora: {carried_corpora or 'none'}; "
          f"carried counters: {carried_counters or 'none'}; skipped {len(skipped)}")
    print(f"  wrote {config.RESULTS_DIR}/raw_counts.csv, aggregate_counts.csv, run_manifest.json")


if __name__ == "__main__":
    run()
