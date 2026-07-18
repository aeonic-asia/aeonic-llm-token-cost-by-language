# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Aeonic fork.** This is Aeonic's fork of `AleksandarPetrov/tokenization-fairness` (MIT; `upstream` remote retained). The upstream research measures how the same text tokenizes into wildly different token counts across FLORES-200 languages. Aeonic extends it into a **current-generation, five-language token-cost eval** (Vietnamese lead + English baseline + Chinese/Russian/German) across the 2026 model matrix — the reproducible companion to the *Vietnamese Token Tax* article. The upstream pipeline is left intact; the Aeonic eval lives in its own `eval/` package.

## Reference paper (read first)

Petrov, La Malfa, Torr, Bibi, *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023), **arXiv:2305.15425** — kept locally at `paper/2305.15425v2.pdf` (git-ignored: it's a third-party PDF, cited by arXiv link rather than redistributed in this public repo; fetch from arXiv if not present). This is the methodological foundation for both the upstream code and the Aeonic eval. Load-bearing facts:

- **Premium definition (§3):** `premium(A vs B) = |tokens(A)| / |tokens(B)|` over the same sentence, translated. The Aeonic eval measures every premium **relative to English**.
- **Corpus:** FLORES-200 parallel corpus (same 2000 sentences human-translated to 200 languages); dev + devtest joined. Committed under `flores200_dataset/`.
- **UNK gate (§4.1):** report a (language, tokenizer) pair only when <10% of characters map to UNK. Immaterial for modern byte-level BPE (o200k/cl100k/Claude/Gemini/Llama have no true UNK) but kept for honesty.
- **Validation oracle (Table 1):** upstream `cl100k_base` (ChatGPT/GPT-4) premiums vs. English — **Vietnamese 2.45, Chinese-Simplified 1.91, German 1.58**. The Aeonic eval reproduces `cl100k_base` offline over the same corpus, so these are the correctness target: if the eval's cl100k premiums don't land on those, the pipeline is wrong.

## What the upstream code is

Research code + project page for the paper above. It measures how the same text, translated across the FLORES-200 languages, tokenizes into wildly different token counts across ~28 tokenizers — the source of cost/latency/context unfairness between language communities.

## Commands

```bash
pip install -r requirements.txt              # needs a HF login for gated models (Llama-2, Qwen)
python compute/compute_tokenizations.py      # regenerate all assets/*.csv and assets/examples/*.json
```

Run from the repo root — `compute_tokenizations.py` uses paths relative to it (`flores200_dataset/`, `compute/`, `assets/`). There is no test suite; correctness is validated by inspecting the output CSVs and the rendered page.

The project page is static: open `index.html` directly or serve the repo root (it fetches `assets/tokenization_lengths_validated.csv` and `assets/examples/*.json`).

## Architecture

The whole pipeline hangs off one abstraction: **`TokenizerInterface`** in `compute/tokenizer_interface.py`. Every tokenizer is a subclass exposing a uniform `encode` / `decode` / `pretty_name` / `count_unknown`. This uniformity is what lets one loop run all tokenizers over all languages.

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

## Conventions

- Tokenizer identity everywhere is `pretty_name`, which becomes the CSV column header — keep names stable and unique or you break the CSVs and the page.
- The FLORES dataset is committed under `flores200_dataset/`; example-sentence selection is by fixed line index, so it's tied to that dataset version.
