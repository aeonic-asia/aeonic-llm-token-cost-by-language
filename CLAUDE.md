# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Aeonic fork.** This is Aeonic's fork of `AleksandarPetrov/tokenization-fairness` (MIT; `upstream` remote retained). The upstream research measures how the same text tokenizes into wildly different token counts across FLORES-200 languages. Aeonic extends it into a **current-generation, five-language token-cost eval** (Vietnamese lead + English baseline + Chinese/Russian/German) across the 2026 model matrix and **two registers** — FLORES+ (formal prose) and MASSIVE (short virtual-assistant utterances; CC BY 4.0, committed slice built by `eval/build_massive.py`) — the reproducible companion to the *Vietnamese Token Tax* article. The upstream pipeline is left intact; the Aeonic eval lives in its own `eval/` package.

## Reference paper (read first)

Petrov, La Malfa, Torr, Bibi, *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023), **arXiv:2305.15425** — kept locally at `paper/2305.15425v2.pdf` (git-ignored: it's a third-party PDF, cited by arXiv link rather than redistributed in this public repo; fetch from arXiv if not present). This is the methodological foundation for both the upstream code and the Aeonic eval. Load-bearing facts:

- **Premium definition (§3):** `premium(A vs B) = |tokens(A)| / |tokens(B)|` over the same sentence, translated. The Aeonic eval measures every premium **relative to English**.
- **Corpus:** FLORES-200 parallel corpus (same 2000 sentences human-translated to 200 languages); dev + devtest joined. Committed under `flores200_dataset/`.
- **UNK gate (§4.1):** report a (language, tokenizer) pair only when <10% of characters map to UNK. Immaterial for modern byte-level BPE (o200k/cl100k/Claude/Gemini/Llama have no true UNK) but kept for honesty.
- **Validation oracle (Table 1):** upstream `cl100k_base` (ChatGPT/GPT-4) premiums vs. English — **Vietnamese 2.45, Chinese-Simplified 1.91, German 1.58**. The Aeonic eval reproduces `cl100k_base` offline over the same corpus, so these are the correctness target: if the eval's cl100k premiums don't land on those, the pipeline is wrong.

## The Aeonic eval (`eval/`) — start here

The Aeonic work is a **self-contained `eval/` package**, deliberately separate from the upstream pipeline below (which is left intact). If you're working on the token-cost eval, this is the whole world; ignore the upstream sections except as method reference.

**Run it** (offline, deterministic — full command detail in `README.md`):

```bash
make setup       # venv + pinned deps (requirements-eval.txt)
make reproduce   # counters -> premium/cost analysis -> figures
make test        # correctness gate: reproduce the paper's cl100k premiums (<0.005)
```

**Where the results are** (this is the canonical, in-repo record — do not restate the numbers elsewhere; read them here):

- `eval/results/summary.json` — headline premiums per corpus/counter, machine-readable, with `dataset_as_of`. **Read this first for the current numbers.**
- `eval/results/premium_by_language.csv` (per-sentence distribution), `cost_by_language.csv` (USD+VND under **two denominators** — per-1k-chars *and* per-sentence; the parallel corpus makes per-sentence a same-meaning comparison, which ranks dense scripts like Chinese very differently from per-character), `within_vendor_inflation.csv` (the newer-vs-older-Claude hook), `aggregate_counts.csv`, `raw_counts.csv`, `run_manifest.json` (which counters actually ran — nothing is estimated), `figures/`.
- The **narrative + interpretation** (why the numbers reshaped the thesis, the ratified/parked angle) lives in the *workshop* repo's `epic-5-vietnamese-token-tax.md` decision log — not duplicated here.

**Key architectural difference from upstream.** The `eval/` package hangs off a narrower **`TokenCounter`** contract (`count(text) -> int`) in `eval/measure.py`, **not** upstream's `TokenizerInterface` (encode/decode/alignment). This is load-bearing: closed models (Claude via `count_tokens`) return a *count only*, never token strings — so nothing fakes `encode` on a count-only model. Offline tokenizers and the Anthropic API both satisfy `TokenCounter`.

**Message-envelope correction (Claude per-sentence counts).** Anthropic's `count_tokens` counts the fully-rendered chat prompt, so a fixed turn/role frame sits on top of the content tokens (**measured: 6 for the newer Claude tokenizer, 7 for the older**; verified in-dataset — `count("x")` = 7 newer / 8 older). The offline counters count bare text with no frame. Left uncorrected this inflates Claude's *per-sentence* counts by that fixed floor — negligible on the whole-corpus aggregate (one call over ~10⁵ tokens) but material on short sentences: a true 2.0× premium reads ~1.6× when +6 lands on both sides of the ratio. So each API counter *measures* its own frame (`TokenCounter.envelope_tokens()`, a single-token probe — never estimated) and `run.py` subtracts it: `raw_counts.csv` keeps both `n_tokens` (raw output) and `n_tokens_content` (frame-stripped); the **per-sentence premium distribution** in `premium_by_language.csv` is built from `n_tokens_content`, while the **aggregate premium** stays the raw paper-style total. Frame per counter is recorded in `run_manifest.json` → `envelope_tokens_by_counter`. Consequence for the article: on MASSIVE the newer-Claude Vietnamese median rises 1.59×→2.00× (= the aggregate) once the frame is removed — the register finding is real, not a measurement wrinkle.

**Adding a counter or corpus without re-measuring the rest.** Runs are selectable with **carry-forward**: `make reproduce COUNTERS=<id>` re-measures only that counter (and `CORPORA=<name>` only that corpus), preserving every other row **byte-for-byte**. This is what lets the key-gated Claude rows stay fixed while a new offline counter is added. New counters register in `build_counter` (`eval/measure.py`); new corpora load via `eval/corpora.py`.

**`eval/` file map:** `config.py` (model matrix, pricing, subsample knobs) · `measure.py` (`TokenCounter` + concrete counters) · `corpora.py` (FLORES+/MASSIVE loaders, NFC gate) · `run.py` (driver + carry-forward) · `analyze.py` (premium/cost/inflation) · `figures.py` (deterministic SVG+PNG) · `build_massive.py` (one-time MASSIVE slice builder) · `tests/test_oracle.py` (the paper oracle) · `results/` (committed dataset) · `tiktoken_cache/` (offline BPE ranks).

**Corpora on disk — the two datasets live at different levels, by design.** Both are loaded *only* through `eval/corpora.py`; nothing else in `eval/` reads a corpus path directly (paths are centralized in `config.py`: `FLORES_DIR`, `MASSIVE_DIR`).

- **FLORES+** → `flores200_dataset/` at the **repo root** — the copy *inherited from upstream*, reused (not duplicated) by the eval. It stays at root because upstream's `compute/compute_tokenizations.py` references it by a root-relative path; moving it would break upstream and diff against `upstream`.
- **MASSIVE** → `eval/massive/` — Aeonic-built (CC BY 4.0 slice), so it lives *with* its owning package.

This root-vs-`eval/` asymmetry is a deliberate consequence of the "leave upstream intact" fork rule, not an inconsistency. If you add a corpus, register its loader + `config` path here and keep Aeonic-authored data under `eval/`.

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
