# LLM token cost by language

> **Aeonic fork** of [`AleksandarPetrov/tokenization-fairness`](https://github.com/AleksandarPetrov/tokenization-fairness) (MIT). The upstream research (below) measured 2023-era tokenizers. This fork extends the same method to a **current-generation, five-language token-cost eval** — the reproducible companion to Aeonic's *Vietnamese Token Tax* analysis.

## Aeonic eval (`eval/`)

Measures how many tokens the same content costs across **English (baseline), Vietnamese, Chinese, Russian, German** on each vendor's current models, across **two committed parallel corpora** — FLORES+ (formal prose) and MASSIVE (short virtual-assistant utterances) — so the premium is reported in both a formal and a conversational register. Premium is always measured relative to English.

```bash
make setup       # create the venv, install pinned deps (requirements-eval.txt)
make reproduce   # counters -> premium/cost analysis -> figures
make test        # reproduce the paper's cl100k premiums (correctness gate)
```

The offline counters need no network once set up (the tiktoken BPE ranks are committed under `eval/tiktoken_cache/`). Outputs land in `eval/results/` (raw counts, premium and cost tables, figures) with a `run_manifest.json` recording exactly which counters ran.

Both corpora are loaded through `eval/corpora.py`, but they sit at different levels: **FLORES+** is the upstream copy at the repo root (`flores200_dataset/`, reused not duplicated — the committed files are the FLORES-200 release that FLORES+ continues; CC BY-SA 4.0, see `flores200_dataset/ATTRIBUTION.md`), while the Aeonic-built **MASSIVE** slice lives under `eval/massive/` (CC BY 4.0, see its `PROVENANCE.md`). This is deliberate — see `CLAUDE.md` ("Corpora on disk") for why.

**Correctness is anchored to the paper.** The eval reproduces the upstream `cl100k_base` premiums (Vietnamese 2.45, Chinese 1.91, German 1.58) to within 0.005 — see `make test`.

**What is measured.** Each vendor's current models, plus OpenAI's 2023-era `cl100k_base` as a historical anchor (as of 2026-10-19):

| Vendor | Models | How it is counted |
|---|---|---|
| Anthropic | Claude Opus 5.5, Fable 5.1, Sonnet 5.5, Haiku 5.5 | `count_tokens` API (`ANTHROPIC_API_KEY`) |
| OpenAI | GPT-6 Sol, Astra, Luna | Responses API input-token count (`OPENAI_API_KEY`); counts match the offline `o200k_base` on every sampled sentence |
| Google | Gemini 4 Argon; Gemini 3.8 Flash, 3.5 Flash-Lite | Gemini API `countTokens` (`GEMINI_API_KEY`); the Flash line matches the offline `gemma4` tokenizer on every sampled sentence, Argon does not |
| Alibaba | Qwen 3.8 (open-weight) | HF `AutoTokenizer`, offline |

The three count-tokens endpoints are free. Models that share a tokenizer are measured separately anyway, and `summary.json` → `tokenizer_coincidence_check` records which ones the data confirms as identical. Without a key, a counter's committed rows are carried forward rather than dropped; `run_manifest.json` records what each pass actually measured. `make reproduce COUNTERS=<id>` re-measures a single counter.

**Like-for-like across vendors.** The count-tokens APIs count a wrapped request, so each carries a fixed frame the offline tokenizers do not. The eval measures that frame per API counter and subtracts it from the per-sentence counts (`raw_counts.csv` keeps both `n_tokens` and `n_tokens_content`). See `CLAUDE.md` → "Message-envelope correction".

Both corpora are committed and run every pass; the MASSIVE slice is a CC BY 4.0 derivative built once by `python -m eval.build_massive` (see `eval/massive/PROVENANCE.md`).

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
