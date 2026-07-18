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

**Correctness is anchored to the paper.** The eval reproduces the upstream `cl100k_base` premiums (Vietnamese 2.45, Chinese 1.91, German 1.58) to within 0.005 — see `make test`.

**Coverage as committed:** OpenAI `o200k_base` and `cl100k_base` counters run offline today. The Anthropic `count_tokens` counters (both Claude tokenizer generations) are implemented and run once an `ANTHROPIC_API_KEY` is provided. The **Gemini** counter runs offline via the google-genai `LocalTokenizer` (`gemini-3-pro-preview` → `gemma3` sentencepiece; no key) once `google-genai[local-tokenizer]` is installed — it is kept out of the lean default deps, so a plain `make setup && make reproduce` skips it cleanly. The **Llama 4** open-weight counter is implemented and runs once `transformers` + an `HF_TOKEN` for the gated `meta-llama/Llama-4-Scout-17B-16E` repo are present. Add a single counter without re-measuring the rest via `make reproduce COUNTERS=<id>` (per-counter carry-forward, the counter analogue of `CORPORA=`). Both corpora (FLORES+ and MASSIVE) are committed and run every pass; the MASSIVE slice is a CC BY 4.0 derivative built once by `python -m eval.build_massive` (see `eval/massive/PROVENANCE.md`). `run_manifest.json` always reflects the true state of a given run — nothing is estimated.

---

## Upstream research: tokenization unfairness between languages

This repository holds the code for experiments and the project page for the [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) paper (`paper/2305.15425v2.pdf`).

**Abstract:**


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
