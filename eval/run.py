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
                       premium and cost-per-1k-chars). `total_chars` is the sum of
                       the per-sentence NFC character counts, NOT len(joined): the
                       n-1 spaces inserted by " ".join exist nowhere in the corpus,
                       and their share of the denominator scales inversely with
                       sentence length (measured: 0.77% of English FLORES chars vs
                       2.30% of Chinese; 2.79% vs 8.74% on MASSIVE). Counting them
                       cancels in any within-language comparison but understates
                       dense scripts by several percent in the cross-language
                       tokens-per-1k-chars figure. Tokens are still counted over the
                       joined string — that IS the paper's method.
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
wall-clock timestamp (it uses config's dated `as_of`), so a re-run produces
byte-identical files. Rows are written in a canonical sort order (corpus,
counter_id, lang[, sentence_idx]) rather than in carry-then-measure order, so a
selective pass and a full rebuild of the same data produce the same bytes — not
merely the same content. Before the sort was added, re-measuring one counter
moved its rows within the file and `git diff` reported deletions plus
reinsertions; the guarantee then held only for full-rebuild-after-full-rebuild.
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


def _versions() -> dict[str, str]:
    """Installed versions of every library that can move a number or a byte.

    Absent optional deps are recorded as "not installed" rather than omitted, so
    the manifest distinguishes "this counter ran without it" from "unknown".
    """
    import importlib.metadata as md
    out: dict[str, str] = {}
    for pkg in ("tiktoken", "pandas", "numpy", "matplotlib",
                "anthropic", "google-genai", "transformers", "sentencepiece"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run() -> None:
    langs = list(config.LANGUAGES)
    selected = _selected_corpora()
    selected_counters = _selected_counters()
    # Load only the selected corpora (each asserts its own line-alignment).
    corpora = {cid: load_corpus(cid, langs) for cid in selected}
    n_by_corpus = {cid: len(c[langs[0]]) for cid, c in corpora.items()}
    # A corpus truncated to 0 bytes in every language passes load_corpus's
    # equal-length check (all zero) and would otherwise be "measured" as 0 tokens /
    # 0 chars, overwriting the committed rows with zeros; only the downstream cost
    # division then fails, with a bare ZeroDivisionError and the data already gone.
    empty = [cid for cid, n in n_by_corpus.items() if n == 0]
    if empty:
        raise SystemExit(f"corpus/corpora {empty} loaded 0 sentences — refusing to "
                         "overwrite committed measurements with zeros. Check the "
                         "corpus files (restore with `git checkout -- <path>`).")

    # Carry-forward preserves every committed (corpus × counter) cell that is NOT
    # successfully re-measured this pass. Load the committed dataset here; the
    # carried set is computed AFTER the run because it keys on what actually
    # *ran* (see below) — not on the mere selection. Loading regardless of subset
    # is what makes a keyless/tokenless `make reproduce` CONVERGE on the committed
    # dataset (preserving the rows it can't re-measure) instead of erasing them.
    raw_path = config.RESULTS_DIR / "raw_counts.csv"
    agg_path = config.RESULTS_DIR / "aggregate_counts.csv"
    # Load each file independently. Gating both on the joint existence of the pair
    # meant a missing raw_counts.csv (5.7 MB — a partial checkout or a truncated
    # write) silently discarded a perfectly loadable aggregate_counts.csv: a keyless
    # full pass then rewrote the aggregate with only the offline counters, erasing
    # every key-gated Claude row, and the empty-dataset guard below did not fire
    # because the surviving offline rows are non-empty.
    ex_raw = pd.read_csv(raw_path) if raw_path.exists() else None
    ex_agg = pd.read_csv(agg_path) if agg_path.exists() else None
    for df, nm in ((ex_raw, "raw_counts.csv"), (ex_agg, "aggregate_counts.csv")):
        if df is not None and "corpus" not in df.columns:
            # Never advise deleting eval/results/: on a keyless machine — the
            # documented default — a full rebuild cannot re-measure the six
            # key-gated Claude counters, so that advice permanently destroys them.
            raise SystemExit(
                f"existing {nm} predates the 'corpus' column. Do NOT delete "
                "eval/results/ — a keyless rebuild cannot re-measure the Claude "
                "counters and they would be lost. Restore the file from git "
                "(`git checkout -- eval/results/`), or migrate it by adding a "
                "'corpus' column set to 'flores'.")
    if (ex_raw is None) != (ex_agg is None):
        missing = "raw_counts.csv" if ex_raw is None else "aggregate_counts.csv"
        print(f"  WARN  {missing} is absent; carrying forward from the other file "
              "only. Restore it from git if this was not intentional.")
    is_subset = (set(selected) != set(config.CORPORA)
                 or set(selected_counters) != {c.id for c in config.MODEL_MATRIX})
    if is_subset and (ex_raw is None or ex_agg is None):
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
        # Construction is guarded too, not just measurement. build_counter is
        # contracted to return (None, reason) rather than raise, but it sits
        # OUTSIDE the per-counter try below, so any kind that breaks that
        # contract takes down the whole pass — no manifest, no CSVs, no
        # carry-forward. One counter must never be able to do that.
        try:
            counter, reason = build_counter(spec)
        except Exception as exc:  # noqa: BLE001
            counter, reason = None, (f"failed to construct: {type(exc).__name__} "
                                     "(see console output for detail)")
            print(f"  ERROR {spec.id:16s} — construction: {type(exc).__name__}: {exc}")
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
                        "spec": spec.spec, "measured_on": config.DATASET_AS_OF,
                        "total_tokens": counter.count(joined),
                        # Sum the per-sentence character counts rather than measuring
                        # the joined string: the n-1 join spaces are not corpus
                        # content, and counting them biases the per-character
                        # denominator ~3x harder for dense scripts (see module
                        # docstring). Tokens stay measured over `joined` — the
                        # cross-sentence BPE merges are part of the paper's method.
                        "total_chars": sum(char_count(s) for s in sentences),
                    })
                    # per-sentence: for the premium distribution. n_tokens is the
                    # raw counter output; n_tokens_content strips the envelope so
                    # the distribution compares like with like.
                    for idx, sent in enumerate(sentences[:max_per_sentence]):
                        raw_tok = counter.count(sent)
                        per_sentence.append({
                            "corpus": cid, "counter_id": spec.id, "lang": lang,
                            "spec": spec.spec, "measured_on": config.DATASET_AS_OF,
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
            # Only the exception TYPE goes into the committed manifest. SDK error
            # strings routinely embed request URLs, response bodies and header
            # echoes (huggingface_hub can embed a resolved, token-bearing URL), and
            # run_manifest.json is committed to a public repo. The full text still
            # goes to the console for the operator.
            skipped.append({"counter": spec.id, "status": spec.status,
                            "reason": f"errored at run: {type(exc).__name__} "
                                      "(see console output for detail)"})
            print(f"  ERROR {spec.id:16s} — {type(exc).__name__}: {exc}")
            continue

    # Carry forward every committed cell NOT successfully measured this pass —
    # keyed on `ran`, so a counter that was skipped for cause, errored mid-run, or
    # simply not selected keeps its committed rows rather than being dropped.
    carried_raw = carried_agg = None
    carried_corpora: list[str] = []
    carried_counters: list[str] = []
    # Either file on its own is enough to carry from. Gating this on ex_raw alone
    # reintroduced the very bug the independent loads above fix: with
    # raw_counts.csv missing, a loadable aggregate_counts.csv was silently
    # discarded and the pass wrote only what it measured.
    if ex_raw is not None or ex_agg is not None:
        # Rows for a counter id that no longer exists in MODEL_MATRIX are dropped
        # rather than carried. Without this a renamed or retired counter leaves
        # ghost rows that no pass can ever purge (carry-forward keys on `ran`, and
        # a counter absent from the matrix can never be in `ran`): they survive
        # every subsequent run, flow through analyze into the committed CSVs, and
        # are invisible in the figures because those iterate MODEL_MATRIX. The
        # matrix is the declaration of what this dataset contains; anything else
        # is residue.
        known = {c.id for c in config.MODEL_MATRIX}
        dropped = sorted(set(ex_agg.counter_id) - known)
        if dropped:
            print(f"  purge {', '.join(dropped)} — no longer in MODEL_MATRIX")

        def _carry(df):
            if df is None:
                return None
            measured = df.corpus.isin(selected) & df.counter_id.isin(ran)
            return df[~measured & df.counter_id.isin(known)]
        carried_raw, carried_agg = _carry(ex_raw), _carry(ex_agg)
        ref = carried_agg if carried_agg is not None else carried_raw
        carried_counters = sorted(set(ref.counter_id) - set(ran)) if ref is not None else []
        carried_corpora = sorted(set(ref.corpus) - set(selected)) if ref is not None else []

        # Provenance gate. A headline column must be measured on the model it names
        # (see CLAUDE.md). Without this, repointing a counter's `spec` to a new
        # flagship and then running ANY unrelated selective pass republishes rows
        # measured on the old endpoint under the new model's name — the exact
        # label/measurement split this repo has already had to correct once.
        spec_by_id = {c.id: c.spec for c in config.MODEL_MATRIX}
        for df in (carried_raw, carried_agg):
            if df is None or df.empty or "spec" not in df.columns:
                continue
            for cid, grp in df.groupby("counter_id"):
                recorded = set(grp["spec"].dropna().unique())
                stale = {s for s in recorded if s != spec_by_id.get(cid)}
                if stale:
                    raise SystemExit(
                        f"refusing to carry forward {cid}: rows were measured on "
                        f"{sorted(stale)} but MODEL_MATRIX now specs "
                        f"'{spec_by_id.get(cid)}'. Re-measure it "
                        f"(`make reproduce COUNTERS={cid}`) so the column is backed "
                        "by the model it names, or restore the previous spec.")
        unrecorded = [c for c in carried_counters
                      if carried_agg is None or "spec" not in carried_agg.columns
                      or carried_agg.loc[carried_agg.counter_id == c, "spec"].isna().all()]
        if unrecorded:
            print(f"  NOTE  provenance unrecorded for carried counter(s) "
                  f"{', '.join(unrecorded)} — measured before spec tracking existed; "
                  "a credentialed re-measure will stamp them.")

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

    # Canonical order, so byte-identity does not depend on which counters this pass
    # happened to re-measure. pd.concat writes carried rows first and newly measured
    # rows last, so without this a selective pass and a full rebuild of identical
    # data differ physically and `git diff` reports mass deletions + reinsertions.
    def _sorted(df, keys):
        keys = [k for k in keys if k in df.columns]
        return df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    final_raw = _sorted(final_raw, ["corpus", "counter_id", "lang", "sentence_idx"])
    final_agg = _sorted(final_agg, ["corpus", "counter_id", "lang"])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_raw.to_csv(raw_path, index=False)
    final_agg.to_csv(agg_path, index=False)

    # Carry the measured message-envelope forward for counters this pass did not
    # re-measure, exactly like their rows. Without this the map documented every
    # counter in the dataset but, after a one-counter pass, described only that
    # counter — so `envelope_tokens_by_counter` silently stopped covering the data
    # it sits beside, and the CLAUDE.md/README statement that the frame is recorded
    # there stopped being true for the carried rows. The per-row `envelope_tokens`
    # column in raw_counts.csv is the source of truth for the carried values.
    if "envelope_tokens" in final_raw.columns:
        for cid in carried_counters:
            if cid in envelope_by_counter:
                continue
            vals = final_raw.loc[final_raw.counter_id == cid, "envelope_tokens"].dropna()
            if not vals.empty:
                envelope_by_counter[cid] = int(vals.iloc[0])
    envelope_by_counter = dict(sorted(envelope_by_counter.items()))

    # Per-counter measurement vintage. `dataset_as_of` is one date for what is
    # actually a merge of many passes, which is the same false-provenance failure
    # DATASET_AS_OF was split from PRICING_AS_OF to avoid — one stamp cannot
    # describe a carried row measured weeks earlier. Counters measured before
    # vintage tracking report "unrecorded" rather than inheriting today's date.
    measured_on_by_counter: dict[str, str] = {}
    if "measured_on" in final_agg.columns:
        for cid, grp in final_agg.groupby("counter_id"):
            vals = sorted(grp["measured_on"].dropna().unique())
            measured_on_by_counter[str(cid)] = vals[0] if len(vals) == 1 else (
                "unrecorded" if not vals else f"mixed: {', '.join(map(str, vals))}")
    for cid in sorted(set(final_agg.counter_id)):
        measured_on_by_counter.setdefault(str(cid), "unrecorded")

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
        "measured_on_by_counter": dict(sorted(measured_on_by_counter.items())),
        "envelope_tokens_by_counter": envelope_by_counter,
        # Record every library that can move a number or a committed byte, not just
        # the two direct imports: matplotlib determines the figure bytes, and
        # anthropic / transformers / google-genai determine the counts themselves.
        # google-genai's version is load-bearing for the Gemini column (its
        # model->tokenizer map is what the counter resolves through), so omitting it
        # from the manifest of the run that used it defeated the point.
        "versions": _versions(),
        "notes": "Dual-register: FLORES+ (formal) + MASSIVE (assistant utterances); "
                 "premium reported per corpus. Anthropic counters run when "
                 "ANTHROPIC_API_KEY is present (aggregate + per-sentence subsample "
                 "per corpus); Gemini via offline google-genai LocalTokenizer (no "
                 "key); open-weight via HF AutoTokenizer — Llama 4 (gated, needs "
                 "HF_TOKEN) and Qwen 3.6 (ungated Apache-2.0, no token). "
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
