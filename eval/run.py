"""Driver: run the matrix over the corpus and write the committed raw dataset.

Outputs (eval/results/):
  raw_counts.csv       long/tidy: counter_id, lang, sentence_idx, n_tokens, n_chars
                       (per-sentence — the source for the premium *distribution*)
  aggregate_counts.csv counter_id, lang, total_tokens, total_chars
                       (paper-style concatenated count — the source for the
                       aggregate premium and cost-per-1k-chars)
  run_manifest.json    which counters ran vs. were skipped (and why), corpus,
                       versions, and the dataset "as of" date.

Determinism: token counts are deterministic and the manifest carries no
wall-clock timestamp (it uses config's dated `as_of`), so re-running produces
byte-identical files — `make reproduce` yields no git churn.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .corpora import load_parallel
from .measure import build_counter, char_count


def run() -> None:
    langs = list(config.LANGUAGES)
    corpus = load_parallel(langs)  # asserts line-alignment
    n_sentences = len(corpus[langs[0]])

    per_sentence: list[dict] = []
    aggregate: list[dict] = []
    ran: list[str] = []
    skipped: list[dict] = []

    for spec in config.MODEL_MATRIX:
        counter, reason = build_counter(spec)
        if counter is None:
            skipped.append({"counter": spec.id, "status": spec.status, "reason": reason})
            print(f"  skip  {spec.id:16s} — {reason}")
            continue
        ran.append(spec.id)
        # API counters are rate-limited: aggregate always, per-sentence only up
        # to a subsample. Offline counters do the full per-sentence sweep.
        is_api = spec.kind in config.API_KINDS
        max_per_sentence = config.API_PER_SENTENCE_SUBSAMPLE if is_api else n_sentences
        print(f"  run   {spec.id:16s} ({spec.display})"
              f"{' [aggregate + %d-sentence subsample]' % max_per_sentence if is_api else ''}")
        try:
            for lang in langs:
                sentences = corpus[lang]
                # aggregate: paper-style — concatenate, count once
                joined = " ".join(sentences)
                aggregate.append({
                    "counter_id": spec.id, "lang": lang,
                    "total_tokens": counter.count(joined),
                    "total_chars": char_count(joined),
                })
                # per-sentence: for the premium distribution
                for idx, sent in enumerate(sentences[:max_per_sentence]):
                    per_sentence.append({
                        "counter_id": spec.id, "lang": lang, "sentence_idx": idx,
                        "n_tokens": counter.count(sent), "n_chars": char_count(sent),
                    })
        except Exception as exc:  # noqa: BLE001 — one bad counter must not kill the run
            # roll back any partial rows for this counter so the dataset stays clean
            aggregate[:] = [r for r in aggregate if r["counter_id"] != spec.id]
            per_sentence[:] = [r for r in per_sentence if r["counter_id"] != spec.id]
            ran.pop()
            skipped.append({"counter": spec.id, "status": spec.status,
                            "reason": f"errored at run: {type(exc).__name__}: {exc}"})
            print(f"  ERROR {spec.id:16s} — {type(exc).__name__}: {exc}")
            continue

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_sentence).to_csv(config.RESULTS_DIR / "raw_counts.csv", index=False)
    pd.DataFrame(aggregate).to_csv(config.RESULTS_DIR / "aggregate_counts.csv", index=False)

    import tiktoken
    manifest = {
        "dataset_as_of": config.PRICING_AS_OF,
        "corpus": {"name": "FLORES+", "splits": ["dev", "devtest"],
                   "languages": config.LANGUAGES, "sentences_per_language": n_sentences},
        "baseline_language": config.BASELINE_LANG,
        "counters_ran": ran,
        "counters_skipped": skipped,
        "versions": {"tiktoken": tiktoken.__version__, "pandas": pd.__version__},
        "notes": "MASSIVE second-domain corpus deferred. Anthropic counters run "
                 "when ANTHROPIC_API_KEY is present (aggregate + per-sentence "
                 "subsample); Gemini/Llama slots deferred. See counters_ran / "
                 "counters_skipped for what actually ran this pass.",
        "api_per_sentence_subsample": config.API_PER_SENTENCE_SUBSAMPLE,
    }
    with open(config.RESULTS_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  ran {len(ran)} counter(s), skipped {len(skipped)}; "
          f"{n_sentences} sentences × {len(langs)} languages")
    print(f"  wrote {config.RESULTS_DIR}/raw_counts.csv, aggregate_counts.csv, run_manifest.json")


if __name__ == "__main__":
    run()
