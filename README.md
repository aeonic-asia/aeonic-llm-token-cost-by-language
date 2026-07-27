# LLM token cost by language

> **Aeonic fork** of [`AleksandarPetrov/tokenization-fairness`](https://github.com/AleksandarPetrov/tokenization-fairness) (MIT). The upstream research (below) measured 2023-era tokenizers. This fork extends the same method to a **current-generation, five-language token-cost eval** — the reproducible companion to Aeonic's *Vietnamese Token Tax* analysis.

## Aeonic eval (`eval/`)

Measures how many tokens the same content costs across **English (baseline), Vietnamese, Chinese, Russian, German** on the 2026 model matrix, across **two committed parallel corpora** — FLORES+ (formal prose) and MASSIVE (short virtual-assistant utterances) — so the premium is reported in both a formal and a conversational register. Premium is always measured relative to English.

```bash
make setup       # create the venv, install pinned deps (requirements-eval.txt)
make reproduce   # counters -> premium/cost analysis -> figures (fully offline)
make test        # reproduce the paper's cl100k premiums (correctness gate)
```

`make reproduce` is offline: the tiktoken BPE ranks are committed under `eval/tiktoken_cache/`. Outputs land in `eval/results/` (raw counts, premium and cost tables, figures) with a `run_manifest.json` recording exactly which counters ran.

Both corpora are loaded through `eval/corpora.py`, but they sit at different levels: **FLORES+** is the upstream copy at the repo root (`flores200_dataset/`, reused not duplicated — the committed files are the FLORES-200 release that FLORES+ continues; CC BY-SA 4.0, see `flores200_dataset/ATTRIBUTION.md`), while the Aeonic-built **MASSIVE** slice lives under `eval/massive/` (CC BY 4.0, see its `PROVENANCE.md`). This is deliberate — see `CLAUDE.md` ("Corpora on disk") for why.

**Correctness is anchored to the paper.** The eval reproduces the upstream `cl100k_base` premiums (Vietnamese 2.45, Chinese 1.91, German 1.58) to within 0.005 — see `make test`.

**Like-for-like across vendors.** Anthropic's `count_tokens` counts a wrapped chat message, so its counts carry a fixed turn/role frame (measured: 6 tokens for the newer Claude tokenizer, 7 for the older) that the offline tokenizers — which count bare text — do not. The eval measures that frame per Claude counter and subtracts it from the **per-sentence** counts (`raw_counts.csv` keeps both the raw `n_tokens` and the frame-stripped `n_tokens_content`; the premium distribution uses the latter). The whole-corpus aggregate is one call, so its frame is negligible and left in. See `CLAUDE.md` → "Message-envelope correction".

**Coverage as committed — every counter below is already measured and in `eval/results/`.** What follows is what you need in order to *re-measure* each one, not a list of gaps.

OpenAI `o200k_base` and `cl100k_base` run offline from the committed tiktoken cache, so a plain `make setup && make reproduce` re-measures them with no credentials. The six Anthropic `count_tokens` counters (both tokenizer generations — the newer measured on the `claude-opus-5` endpoint — plus the shared-tokenizer models that confirm the fold: Opus 4.8, Sonnet 5, Fable 5, Haiku 4.5) need an `ANTHROPIC_API_KEY` to re-measure; without one they are carried forward from the committed dataset rather than dropped. The three **Gemini** counters (`gemini-3.1-pro-preview`, Google's current Pro model, plus the `gemini-3.5-flash` and `gemini-3.1-flash-lite` serving tiers that share its `gemma4` tokenizer) run offline via the google-genai `LocalTokenizer` — no key — once `google-genai[local-tokenizer]` is installed.

*One* Gemini column, deliberately — three counters, but one tokenizer: `gemma3` and `gemma4` are two distribution artifacts of the same 262,144-entry text vocabulary (they differ only in chat-control tokens), Google's Gemma 3 and Gemma 4 technical reports both state the tokenizer is unchanged from Gemini 2.0, and the earlier second column measured `gemini-3-pro-preview`, which Google shut down on 2026-03-09 and now aliases to 3.1 Pro. The two Flash counters exist for the *dollar* figures rather than the tokenizer ones: identical token counts at $1.50 and $0.25 against Pro's $2.00, so one tokenizer is drawn across an 8× price range instead of a single bar. Two newer GA models — Gemini 3.6 Flash and 3.5 Flash-Lite — are deliberately absent: the SDK's model→tokenizer table maps neither (checked against the pinned 2.12.1 and `main`, which are identical), and this eval will not assign a tokenizer the vendor has not published. The **open-weight** counters need `transformers`: **Llama 4** (`meta-llama/Llama-4-Scout-17B-16E`) is a *gated* repo and additionally needs an `HF_TOKEN`; **Qwen 3.6** (`Qwen/Qwen3.6-27B`, Apache-2.0) is *ungated* and downloads with no credentials. Add a single counter without re-measuring the rest via `make reproduce COUNTERS=<id>` (per-counter carry-forward, the counter analogue of `CORPORA=`). Both corpora (FLORES+ and MASSIVE) are committed and run every pass; the MASSIVE slice is a CC BY 4.0 derivative built once by `python -m eval.build_massive` (see `eval/massive/PROVENANCE.md`). `run_manifest.json` records what a pass actually did — which counters ran, which were carried forward, which were skipped and why; nothing is estimated. One caveat about the *committed* copy: the dataset's last pass applied deterministic transforms rather than re-measuring, so that file carries a hand-authored `migration_2026_07_25` note, an empty `counters_ran`, and a `versions` map from an earlier run. It describes the dataset's history honestly, but it is not the output of a single `make reproduce` — the first credentialed full pass regenerates it (and widens the CSV schema with the `spec` / `measured_on` provenance columns, a large diff in which no measured value changes).

---

## Upstream research: tokenization unfairness between languages

This repository holds the code for experiments and the project page for the [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) paper.

**Abstract** (as published in [arXiv v2](https://arxiv.org/abs/2305.15425v2); the current arXiv landing page shows a later-edited abstract. The PDF is not redistributed here — `paper/` is git-ignored, so a clean checkout will not contain it):


_Recent language models have shown impressive multilingual performance, even when not explicitly trained for it. Despite this, concerns have been raised about the quality of their outputs across different languages. In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked. The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases. These disparities persist across the 17 tokenizers we evaluate, even if they are intentionally trained for multilingual support. Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs. This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the models. Therefore, we make the case that we should train future language models using multilingually fair tokenizers._


**Repository Structure:**

The tokenizers are defined in `compute/tokenizer_interface.py` and the computation of the parity tables is done in `compute/compute_tokenizations.py`.

The FLORES-200 dataset is provided.

We also provide the computed table in the `assets` directory:

- `tokenization_unknown_fraction.csv`: What fraction of input characters are mapped to the UNK token for all tokenizers and languages.
- `tokenization_lengths.csv`: The tokenization lengths for all tokenizers and languages.
- `tokenization_lengths_validated.csv`: Tokenization lengths where the language-tokenizer pairs with more than 10% characters mapped to UNK tokens removed.

**Cite as:**

```
@inproceedings{petrov2023token_unfairness,
    title = {Language Model Tokenizers Introduce Unfairness Between Languages},
    author = {Petrov, Aleksandar and La Malfa, Emanuele and H. S. Torr, Philip and Bibi, Adel},    
    booktitle = {Advances in Neural Information Processing Systems},
    url = {https://arxiv.org/abs/2305.15425},
    year = {2023}
}
```
