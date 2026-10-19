# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Aeonic fork.** This is Aeonic's fork of `AleksandarPetrov/tokenization-fairness` (MIT; `upstream` remote retained). The upstream research measures how the same text tokenizes into wildly different token counts across FLORES-200 languages. Aeonic extends it into a **current-generation, five-language token-cost eval** (Vietnamese lead + English baseline + Chinese/Russian/German) across each vendor's current models and **two registers** — FLORES+ (formal prose) and MASSIVE (short virtual-assistant utterances; CC BY 4.0, committed slice built by `eval/build_massive.py`) — the reproducible companion to the *Vietnamese Token Tax* article. The upstream pipeline is left intact; the Aeonic eval lives in its own `eval/` package.

## Reference paper (read first)

Petrov, La Malfa, Torr, Bibi, *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023), **arXiv:2305.15425** — kept locally at `paper/2305.15425v2.pdf` (git-ignored: it's a third-party PDF, cited by arXiv link rather than redistributed in this public repo; fetch from arXiv if not present). This is the methodological foundation for both the upstream code and the Aeonic eval. Load-bearing facts:

- **Premium definition (§3):** `premium(A vs B) = |tokens(A)| / |tokens(B)|` over the same sentence, translated. The Aeonic eval measures every premium **relative to English**.
- **Corpus:** FLORES-200 parallel corpus (same 2000 sentences human-translated to 200 languages); dev + devtest joined. Committed under `flores200_dataset/`.
- **UNK gate (§4.1):** report a (language, tokenizer) pair only when <10% of characters map to UNK. Immaterial for modern byte-level BPE (o200k/cl100k/Claude/Gemini/Qwen have no true UNK) but kept for honesty.
- **Validation oracle (Table 1):** upstream `cl100k_base` (ChatGPT/GPT-4) premiums vs. English — **Vietnamese 2.45, Chinese-Simplified 1.91, German 1.58**. The Aeonic eval reproduces `cl100k_base` offline over the same corpus, so these are the correctness target: if the eval's cl100k premiums don't land on those, the pipeline is wrong.

## The Aeonic eval (`eval/`) — start here

A self-contained package, deliberately separate from the upstream pipeline below. If you are working on the token-cost eval, this is the whole world.

```bash
make setup       # venv + pinned deps (requirements-eval.txt); optional counter deps are listed there
make reproduce   # counters -> premium/cost analysis -> figures
make test        # correctness gates: the paper oracle, palette, carry-forward, envelope, figure stems, shared tokenizer
```

API counters need `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (all three count-tokens endpoints are free; Anthropic's still requires a positive credit balance). Without a key a counter's committed rows are carried forward.

**Where the results are** — the canonical, in-repo record; do not restate numbers elsewhere:

- `eval/results/summary.json` — headline premiums per corpus/counter. Read **`coverage`** before citing a date (per-counter measurement dates, library versions, oracle margin) and **`tokenizer_coincidence_check`** before claiming two models share a tokenizer.
- `premium_by_language.csv` (per-sentence distribution — **check `n_sentences` / `sample_basis` before quoting a percentile**: API counters use a deterministic 200-sentence head slice, topically clustered; offline counters the full sweep), `cost_by_language.csv` (USD + VND, per 1k chars *and* per sentence — the two rank dense scripts very differently), `aggregate_counts.csv`, `raw_counts.csv`, `run_manifest.json` (what a pass actually did), `figures/`.
- The editorial narrative lives in Aeonic's private workshop, not here. This repo is the authority only for what was measured and how.

**The matrix** (`config.MODEL_MATRIX`) is each vendor's current lineup plus `cl100k_base` as the single historical anchor and paper oracle. One **headline column per distinct tokenizer**, each measured on the model it names; other price tiers and offline proofs **fold** into it (`flagship_group`, `fold_reason="shared"`). Rules:

- **A fold is a hypothesis until the data confirms it.** The coincidence check compares **content totals** (aggregate minus each counter's measured envelope, `analyze.content_totals`) and reports `shared_in_all_corpora` only when every corpus agrees; figure captions claim "one shared tokenizer" only under the same test. Equality is evidence, not proof — before adding a column for a "new" tokenizer, diff the vocabularies when the artifact is downloadable.
- **Superseded models are dropped, not kept as proxies.** A new model on a measured tokenizer is a price row (folded counter + `PRICING` entry), not a new column.
- **Never assign a tokenizer by inference.** tiktoken and the google-genai offline table lag vendor releases; an unmapped model is measured through its vendor's count-tokens API (`kind="openai"` / `"gemini_api"` / `"anthropic"`), never hand-assigned. The offline counters `o200k_base` and `gemma4` stay as proofs that tie API models to published tokenizers.
- **Anthropic publishes no tokenizer**; it is measurable only through `count_tokens`.
- **A changed `spec` refuses carry-forward** (`run.py`), so a relabelled column can never keep another model's numbers. Retired counter ids are purged automatically.

**Message-envelope correction.** Count-tokens endpoints count a wrapped request, so a fixed frame sits on top of the content (measured per counter in `run_manifest.envelope_tokens_by_counter`; it differs by vendor and model and has changed between Claude generations). `measure.ApiCounter.envelope_tokens` probes eight single characters across classes and takes the **floor** — a character is never fewer than one token, so a multi-token probe can only push its frame up (on the Claude tokenizer, uppercase `Z` costs two). At least `ENVELOPE_MIN_AGREEING` probes must reach the floor. The subtrahend (a lone character is one token) is an assumption, tested rather than verified. Per-sentence counts are envelope-stripped (`n_tokens_content`); the aggregate stays the raw paper-style total, envelope-inclusive (<0.25% of any corpus total).

**Runs.** `make reproduce COUNTERS=<id,...> CORPORA=<name>` re-measures only what is selected and carries every other row forward byte-for-byte. A credentialed counter that was available but produced no rows (e.g. retries exhausted) keeps its rows but makes `run.py` exit non-zero. API per-sentence calls run on a thread pool (`API_CONCURRENCY`); `map()` preserves order, so output is byte-identical to a sequential pass. Never run two passes at once — both rewrite the same result files.

**Prices** (`config.PRICING`) are today's input list prices, each row dated and sourced; the dataset is a dated snapshot and is not re-priced later; a vendor's announced dated step-up is noted in the row, not tracked. Unpriced = no list price exists (self-hosted, historical, or an offline proof whose API model carries the price).

**Figures render once per (locale, theme)**: `fig-<stem>-<corpus>[-vi][-dark].svg`. Labels are localised at source from whole-sentence message catalogues (`figures.LOCALES`) — matplotlib draws text as paths, so there is nothing to translate afterwards; `_check_style_registries` fails at import on a missing key, language or unit.

**Figure encoding: hue = tokenizer, lightness step = price tier** (lighter = cheaper; `figures._SLOT_BY_FLAGSHIP`, `Theme.tints`). Rendered orders follow cost, so **a vendor price change can create an inaccessible adjacency with no colour edited** — `tests/test_palette_adjacency.py` derives every rendered order from the committed data and gates cross-family neighbours (normal ΔE ≥ 15, protan/deutan ΔE ≥ 8), within-family ramps (monotone, ΔL ≥ 0.06) and fill contrast (≥ 2:1). Its ΔE maths is pinned by an 11-pair calibration case; if that fails, every colour number is suspect. When it fails, search slot assignments against all orders and both themes rather than nudging one hex.

**Corpus quirks worth knowing.** `total_chars` sums per-sentence NFC characters (join spaces excluded — they bias dense scripts in the per-character view). FLORES `vie_Latn` 429 is the only line with HTML (`km<sup>2</sup>`) and tokenizes differently via API and offline artifacts (Gemini Flash API 104 vs gemma4 112) — a markup artifact, never a language finding; `vie_Latn` 745 carries an undecoded `&amp;`.

**Corpora on disk.** Loaded only through `eval/corpora.py` (paths in `config.py`). FLORES+ stays at the repo root (`flores200_dataset/`, inherited from upstream, which references it by a root-relative path); the Aeonic-built MASSIVE slice lives in `eval/massive/` (CC BY 4.0, `PROVENANCE.md`). Keep Aeonic-authored data under `eval/`.

**`eval/` file map:** `config.py` (matrix, pricing, knobs) · `measure.py` (`TokenCounter`, `ApiCounter` + vendor counters, offline counters) · `corpora.py` · `run.py` (driver, carry-forward) · `analyze.py` (premium, cost, coincidence check) · `figures.py` · `build_massive.py` · `tests/` · `results/` (committed dataset) · `tiktoken_cache/` (offline BPE ranks).

## What the upstream code is

Research code + project page for the paper above. It measures how the same text, translated across the FLORES-200 languages, tokenizes into wildly different token counts across ~28 tokenizers — the source of cost/latency/context unfairness between language communities.

## Commands (upstream pipeline only — for the Aeonic eval see "The Aeonic eval" above)

```bash
pip install -r requirements.txt              # needs a HF login for gated models (Llama-2, Qwen)
python compute/compute_tokenizations.py      # regenerate all assets/*.csv and assets/examples/*.json
```

Run from the repo root — `compute_tokenizations.py` uses paths relative to it (`flores200_dataset/`, `compute/`, `assets/`). There is no test suite; correctness is validated by inspecting the output CSVs and the rendered page.

The project page is static: open `index.html` directly or serve the repo root (it fetches `assets/tokenization_lengths_validated.csv` and `assets/examples/*.json`).

## Upstream architecture (not the Aeonic eval)

The whole upstream pipeline hangs off one abstraction: **`TokenizerInterface`** in `compute/tokenizer_interface.py`. Every tokenizer is a subclass exposing a uniform `encode` / `decode` / `pretty_name` / `count_unknown`. This uniformity is what lets one loop run all tokenizers over all languages.

- **Base + family subclasses** (`tokenizer_interface.py`): `OpenAITokenizer` (tiktoken), `HuggingFaceTokenizer` (transformers `AutoTokenizer`), plus special cases `UTF32_Tokenizer`, `FacebookAI_SeamlessM4T`, and NLLB. Concrete tokenizers are usually a two-line subclass setting `tokenizer` (model id) and `tokenizer_name`. **`ALL_TOKENIZERS` at the bottom of the file is the registry** — the driver iterates exactly this list, so adding a tokenizer means: subclass the right base, set the model id, append to `ALL_TOKENIZERS`.
- **`count_unknown`** estimates how many tokens map to UNK. Byte/char-level tokenizers (UTF-32, ByT5, CANINE) and tiktoken return 0; HF and Seamless estimate it from length lost after stripping UNK tokens. This drives validation.
- **`align_tokens_to_text`** greedily merges consecutive tokens until they decode to a complete grapheme (using `NOT_COMPLETE_SYMBOL_ORD`, the U+FFFD replacement char), producing the per-token highlighting shown on the page. `latex_pretty` is the LaTeX-figure equivalent used for the paper.

**Driver** `compute/compute_tokenizations.py`: for each of the ~200 FLORES languages, joins all dev+devtest sentences, encodes with every tokenizer, and writes:
- `assets/tokenization_lengths.csv` — raw token counts (rows=languages, cols=tokenizers)
- `assets/tokenization_unknown.csv` / `_unknown_fraction.csv` — UNK counts and fractions
- `assets/tokenization_lengths_validated.csv` — lengths with any (language, tokenizer) pair over 10% UNK replaced by `–––`. **This is the file the web page consumes.**
- `assets/examples/{lang}.json` — 10 fixed example sentences (hard-coded indices) with per-token alignment, for the interactive highlighting.

The first language is processed serially before the loop so gated models download once (the commented-out `multiprocessing.Pool` is the intended parallel path).

**Reference data** (`compute/`): `flores_language_map.csv` (code → full name, joined into every output), `flores_family_map.csv` (language family), `flores_rtl.txt` (RTL languages, controls the `reverse` flag in alignment).

**Notebooks** are analysis/figure generation on top of the CSVs, not part of the compute pipeline: `prepare_tables.ipynb` (paper plots/tables), `how_much_will_english_lose.ipynb` (adds `compute/` to `sys.path` and reuses the tokenizer classes for a follow-up analysis).

## Upstream conventions

- Tokenizer identity everywhere is `pretty_name`, which becomes the CSV column header — keep names stable and unique or you break the CSVs and the page.
- The FLORES dataset is committed under `flores200_dataset/`; example-sentence selection is by fixed line index, so it's tied to that dataset version.
