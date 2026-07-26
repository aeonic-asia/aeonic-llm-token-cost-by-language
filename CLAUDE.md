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
- `eval/results/premium_by_language.csv` (per-sentence distribution — **check `n_sentences` / `sample_basis` before quoting a percentile**: API counters use a deterministic 200-sentence head slice, offline counters the full 2009/2033 sweep), `cost_by_language.csv` (USD+VND under **two denominators** — per-1k-chars *and* per-sentence; the parallel corpus makes per-sentence a same-meaning comparison, which ranks dense scripts like Chinese very differently from per-character), `within_vendor_inflation.csv` (the newer-vs-older-Claude hook), `aggregate_counts.csv`, `raw_counts.csv`, `run_manifest.json` (which counters actually ran — nothing is estimated), `figures/`.
- **`summary.json` → `coverage`** says what the rest of the file is worth: which counters were measured in the last pass versus carried forward from an earlier one, per-counter measurement dates, the library versions used, and the oracle's pass/fail with its margin. One `dataset_as_of` cannot describe a dataset that is a merge of several passes — read `coverage` before citing a date.
- **`summary.json` → `tokenizer_coincidence_check`** (renamed from `shared_tokenizer_pairs_by_corpus`). A pair is reported as `shared_in_all_corpora` only when its totals are identical in **every** corpus; pairs that agree in one register only are listed separately under `coincident_in_some_corpora` and are **not** an identity claim. Equality is evidence of a shared tokenizer, never proof: two different tokenizers can agree on a corpus that does not exercise their differences, which is exactly what happened with the two Gemini counters before the fold.
- The **narrative + interpretation** (why the numbers reshaped the thesis, the editorial angle) lives in Aeonic's private editorial workspace, not in this repo. Nothing here should restate it, and this repo is the authority only for what was measured and how.

**Key architectural difference from upstream.** The `eval/` package hangs off a narrower **`TokenCounter`** contract (`count(text) -> int`) in `eval/measure.py`, **not** upstream's `TokenizerInterface` (encode/decode/alignment). This is load-bearing: closed models (Claude via `count_tokens`) return a *count only*, never token strings — so nothing fakes `encode` on a count-only model. Offline tokenizers and the Anthropic API both satisfy `TokenCounter`.

**Message-envelope correction (Claude per-sentence counts).** Anthropic's `count_tokens` counts the fully-rendered chat prompt, so a fixed turn/role frame sits on top of the content tokens (**measured: 6 for the newer Claude tokenizer, 7 for the older**). The offline counters count bare text with no frame. Left uncorrected this inflates Claude's *per-sentence* counts by that fixed floor — small on the whole-corpus aggregate but material on short sentences: a true 2.0× premium reads ~1.6× when +6 lands on both sides of the ratio. So each API counter *measures* its own frame (`TokenCounter.envelope_tokens()`) and `run.py` subtracts it: `raw_counts.csv` keeps both `n_tokens` (raw output) and `n_tokens_content` (frame-stripped); the **per-sentence premium distribution** in `premium_by_language.csv` is built from `n_tokens_content`, while the **aggregate premium** stays the raw paper-style total. Frame per counter is recorded in `run_manifest.json` → `envelope_tokens_by_counter`. Consequence for the article: on MASSIVE the newer-Claude Vietnamese median rises 1.59×→2.00× (= the aggregate) once the frame is removed — the register finding is real, not a measurement wrinkle.

Two honesty caveats on that frame, both worth stating precisely because the numbers are load-bearing. **It is measured-minus-an-assumption, not purely measured:** `count(probe)` is measured, but the subtrahend — that a lone ASCII character is one token — cannot be checked directly against a tokenizer whose vocabulary Anthropic does not publish. `measure.py` probes three distinct single characters and requires agreement, which rules out a class-specific surprise but *cannot* distinguish "each probe is one token" from "each is two". No probe row exists in `raw_counts.csv`, so this is not verifiable from the committed data. **And the aggregate's exposure is corpus-dependent, not uniformly negligible:** the frame is 0.0073% / 0.0124% of the English total on FLORES (newer / older) but **0.0286% / 0.0463% on MASSIVE**, whose totals are ~1.5–2.1×10⁴ tokens rather than ~10⁵. Worst resulting bias on a published aggregate premium is **+0.0007** (massive/claude-old/vie) — below the 4th decimal, so the numbers stand. But the aggregate is envelope-*inclusive* while the distribution is envelope-*stripped*: near-identical bases, not identical ones.

**Adding a counter or corpus without re-measuring the rest.** Runs are selectable with **carry-forward**: `make reproduce COUNTERS=<id>` re-measures only that counter (and `CORPORA=<name>` only that corpus), preserving every other row **byte-for-byte**. This is what lets the key-gated Claude rows stay fixed while a new offline counter is added. New counters register in `build_counter` (`eval/measure.py`); new corpora load via `eval/corpora.py`.

**Carry-forward is silent only when the pass *could not* measure.** No credential, missing dependency, counter not selected — those keep their committed rows and the run exits 0, which is the whole point on a keyless machine. A counter that *was* available and still produced no rows (a rate limit outlasting the retry budget is the realistic case) also keeps its rows, but `run.py` then exits **non-zero**: a credentialed pass must never republish a stale counter behind a green exit. Re-measure it with `make reproduce COUNTERS=<id>`, and note that the failing exit stops `make reproduce` before `analyze`/`figures`, so the derived artifacts stay one pass behind until you do. The Anthropic client's retry budget and per-request timeout are set explicitly in `config.py` (`ANTHROPIC_MAX_RETRIES` / `ANTHROPIC_TIMEOUT_S`) — the SDK's own retry machinery handles `retry-after`; only its default sizing was wrong for ~12,000 sequential calls.

**Counter identity is the TOKENIZER, not the model.** Headline columns are named per distinct tokenizer; models that share one fold into it (`flagship_group` + `fold_reason="shared"`) and exist to *prove* the sharing in-dataset via the coincidence check. Two consequences worth knowing before editing `config.MODEL_MATRIX`:

- **Anthropic publishes no tokenizer name.** Unlike OpenAI (`cl100k_base`/`o200k_base`) and Google (`gemma3`/`gemma4`), Anthropic ships no named or downloadable tokenizer — it is measurable only through `count_tokens`. Hence the coined, release-stable generation ids `claude-new` / `claude-old`: the **id** stays put across releases while the **spec and label together** track the model being measured. (Say "Anthropic's recommended default" rather than "flagship" for Opus 5 — Anthropic describes Fable 5 as its most capable widely released model.) (The only tokenizer artifact Anthropic ever named is `claude.json` in `@anthropic-ai/tokenizer` — legacy, pre-Claude-3, and disclaimed by Anthropic as inaccurate for Claude 3+. Community vocab estimates disagree; never state one as fact.)
- **A headline column must be measured on the model it names.** `claude-new` is labelled "Claude Opus 5" *and* specs `claude-opus-5`. When the flagship rotates, repoint the spec and re-measure, then keep the outgoing model as a fold proxy (that is why `claude-opus-4-8` exists). An earlier cut moved the label alone and left the spec on the prior endpoint — the numbers were right, but the measured model appeared nowhere in the published figure and the caption asserted a verification the reader could not see. Don't reintroduce that split.
- **A "tokenizer split" needs a vocabulary diff, not an equality test.** The matrix carried two Gemini columns on the strength of the SDK mapping 3.0 and 3.1 to different tokenizer artifacts, plus a single differing sentence. Both were true and neither established a split: the artifacts ship the same 262,144-entry text vocabulary (only chat-control tokens differ), Google's own Gemma 3 and Gemma 4 reports state the tokenizer is unchanged from Gemini 2.0, and the differing sentence turned out to be the corpus's one line of HTML. Before adding a column for a "new" tokenizer, diff the vocabularies and check the vendor's technical report — counts agreeing or disagreeing on one corpus proves neither direction.
- **A new flagship is usually a price row, not a run.** If a released model shares an already-measured tokenizer, add it as a folded proxy + a `PRICING` entry; no new headline column. Only a genuinely new tokenizer earns its own column. Anthropic's docs place the Claude 4.7+ line on the newer tokenizer, but treat that as the *hypothesis a proxy tests*, never as the finding — every fold in this repo is confirmed in-dataset by the coincidence check before it is stated.

**Figure encoding: hue = tokenizer, tint step = serving SKU.** One hue per distinct tokenizer (`figures._SLOT_BY_FLAGSHIP`), inherited by every model folded into it; models that share a tokenizer but differ in price take different *lightness steps* of that hue (`_TINT_BY_COUNTER`), ordered lighter=cheaper, never a different hue. Fills are flat — no hatch (texture belongs to the a11y/print path, not to a chart's default look). So a colour means the same thing in every figure, and the dollar charts *show* the "same tokens, different price" point instead of merely captioning it. `_series_style` raises if two counters in one chart resolve to the same fill — which is the correct outcome for two SKUs at one price, since the step encodes price. Which hue each tokenizer holds is chosen against the **rendered** bar orders (the two bar figures sort columns differently, so both must pass), and the gates plus their provenance caveat are documented at `_SLOT_BY_FLAGSHIP`; re-check both orders before changing a hue or adding a step.

**HF counters: gated vs. ungated.** `kind="hf"` counters set `gated=True` only when the repo requires accepting a license (Meta Llama → needs `HF_TOKEN`). Ungated Apache-2.0 repos (Qwen) leave it `False` and download with no credentials — `measure.py` demands a token only for gated ones, so an open repo isn't wrongly skipped.

**`eval/` file map:** `config.py` (model matrix, pricing, subsample knobs) · `measure.py` (`TokenCounter` + concrete counters) · `corpora.py` (FLORES+/MASSIVE loaders, NFC gate) · `run.py` (driver + carry-forward) · `analyze.py` (premium/cost/inflation) · `figures.py` (deterministic SVG+PNG) · `build_massive.py` (one-time MASSIVE slice builder) · `tests/test_oracle.py` (the paper oracle) · `results/` (committed dataset) · `tiktoken_cache/` (offline BPE ranks).

**Two things to know about the character denominator.** `aggregate_counts.total_chars` is the **sum of the per-sentence NFC character counts**, not the length of the `" ".join(...)` string the tokens are counted over. The n-1 join spaces are not corpus content, and their share of the denominator scales inversely with sentence length — measured at 0.77% of English FLORES characters against 2.30% of Chinese, and 2.79% against 8.74% on MASSIVE. Counting them cancels in any within-language model comparison but understates dense scripts by up to ~6pp in the cross-language `tokens_per_1k_chars` view. Tokens are still counted over the joined string; the cross-sentence merges are part of the paper's method. (Per-sentence cost is unaffected either way.)

Separately, **one FLORES sentence contains an HTML tag** — `vie_Latn` index 429, `km<sup>2</sup>` — and it is the only line carrying a tag in any of the ten corpus files (6 tags across **~1.52M NFC characters**; the earlier "~260K" was one file's count, not ten). One further line, `vie_Latn` 745, carries an undecoded `&amp;` entity. Every counter tokenizes both as content, which is faithful to upstream FLORES and immaterial to every headline number. Line 429 is worth knowing only because it was the single sentence on which the two Gemini tokenizer artifacts disagreed, and it briefly looked like a language finding.

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
