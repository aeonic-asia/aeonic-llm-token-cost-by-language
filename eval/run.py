"""Driver: run the matrix over the corpora and write the committed raw dataset.

Outputs (eval/results/):
  raw_counts.csv       long/tidy: corpus, counter_id, lang, sentence_idx,
                       n_tokens, n_chars (per-sentence — source for the premium
                       *distribution*)
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


def run() -> None:
    langs = list(config.LANGUAGES)
    selected = _selected_corpora()
    # Load only the selected corpora (each asserts its own line-alignment).
    corpora = {cid: load_corpus(cid, langs) for cid in selected}
    n_by_corpus = {cid: len(c[langs[0]]) for cid, c in corpora.items()}

    # Carry-forward: when measuring a subset, preserve the other corpora's rows
    # from the existing committed dataset instead of re-measuring them.
    carried_raw = carried_agg = None
    carried_corpora: list[str] = []
    raw_path = config.RESULTS_DIR / "raw_counts.csv"
    agg_path = config.RESULTS_DIR / "aggregate_counts.csv"
    if set(selected) != set(config.CORPORA) and raw_path.exists() and agg_path.exists():
        ex_raw, ex_agg = pd.read_csv(raw_path), pd.read_csv(agg_path)
        if "corpus" not in ex_raw.columns or "corpus" not in ex_agg.columns:
            raise SystemExit(
                "existing results lack a 'corpus' column — run a full "
                "`make reproduce` once before selecting a subset with EVAL_CORPORA")
        carried_raw = ex_raw[~ex_raw.corpus.isin(selected)]
        carried_agg = ex_agg[~ex_agg.corpus.isin(selected)]
        carried_corpora = sorted(set(carried_agg.corpus))

    # Build counters once, then run each over every selected corpus.
    built: list[tuple] = []
    skipped: list[dict] = []
    for spec in config.MODEL_MATRIX:
        counter, reason = build_counter(spec)
        if counter is None:
            skipped.append({"counter": spec.id, "status": spec.status, "reason": reason})
            print(f"  skip  {spec.id:16s} — {reason}")
            continue
        built.append((spec, counter))

    per_sentence: list[dict] = []
    aggregate: list[dict] = []
    ran: list[str] = []

    for spec, counter in built:
        ran.append(spec.id)
        # API counters are rate-limited: aggregate always, per-sentence only up
        # to a subsample. Offline counters do the full per-sentence sweep.
        is_api = spec.kind in config.API_KINDS
        note = " [aggregate + %d-sentence subsample/corpus]" % config.API_PER_SENTENCE_SUBSAMPLE
        print(f"  run   {spec.id:16s} ({spec.display}){note if is_api else ''}")
        try:
            for cid, corpus in corpora.items():
                max_per_sentence = (config.API_PER_SENTENCE_SUBSAMPLE
                                    if is_api else n_by_corpus[cid])
                for lang in langs:
                    sentences = corpus[lang]
                    # aggregate: paper-style — concatenate, count once
                    joined = " ".join(sentences)
                    aggregate.append({
                        "corpus": cid, "counter_id": spec.id, "lang": lang,
                        "total_tokens": counter.count(joined),
                        "total_chars": char_count(joined),
                    })
                    # per-sentence: for the premium distribution
                    for idx, sent in enumerate(sentences[:max_per_sentence]):
                        per_sentence.append({
                            "corpus": cid, "counter_id": spec.id, "lang": lang,
                            "sentence_idx": idx,
                            "n_tokens": counter.count(sent), "n_chars": char_count(sent),
                        })
        except Exception as exc:  # noqa: BLE001 — one bad counter must not kill the run
            # roll back any partial rows for this counter (across all corpora)
            aggregate[:] = [r for r in aggregate if r["counter_id"] != spec.id]
            per_sentence[:] = [r for r in per_sentence if r["counter_id"] != spec.id]
            ran.remove(spec.id)
            skipped.append({"counter": spec.id, "status": spec.status,
                            "reason": f"errored at run: {type(exc).__name__}: {exc}"})
            print(f"  ERROR {spec.id:16s} — {type(exc).__name__}: {exc}")
            continue

    # Union freshly-measured rows with any carried-forward corpora.
    new_raw, new_agg = pd.DataFrame(per_sentence), pd.DataFrame(aggregate)
    final_raw = (pd.concat([carried_raw, new_raw], ignore_index=True)
                 if carried_raw is not None else new_raw)
    final_agg = (pd.concat([carried_agg, new_agg], ignore_index=True)
                 if carried_agg is not None else new_agg)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_raw.to_csv(raw_path, index=False)
    final_agg.to_csv(agg_path, index=False)

    import tiktoken
    manifest = {
        "dataset_as_of": config.PRICING_AS_OF,
        "corpora": {cid: name for cid, name in config.CORPORA.items()},
        "corpora_measured_this_pass": {cid: n_by_corpus[cid] for cid in selected},
        "corpora_carried_forward": carried_corpora,
        "languages": config.LANGUAGES,
        "baseline_language": config.BASELINE_LANG,
        "counters_ran": ran,
        "counters_skipped": skipped,
        "versions": {"tiktoken": tiktoken.__version__, "pandas": pd.__version__},
        "notes": "Dual-register: FLORES+ (formal) + MASSIVE (assistant utterances); "
                 "premium reported per corpus. Anthropic counters run when "
                 "ANTHROPIC_API_KEY is present (aggregate + per-sentence subsample "
                 "per corpus); Gemini/Llama slots deferred. `corpora_carried_forward` "
                 "lists corpora preserved verbatim from the prior dataset (not "
                 "re-measured this pass).",
        "api_per_sentence_subsample": config.API_PER_SENTENCE_SUBSAMPLE,
    }
    with open(config.RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  measured {len(ran)} counter(s) over corpora "
          f"[{', '.join(f'{c}={n}' for c, n in n_by_corpus.items())}]; "
          f"carried forward: {carried_corpora or 'none'}; skipped {len(skipped)}")
    print(f"  wrote {config.RESULTS_DIR}/raw_counts.csv, aggregate_counts.csv, run_manifest.json")


if __name__ == "__main__":
    run()
