# Tokenizer provenance and terms

No tokenizer artifact is redistributed in this repository. Every counter resolves
its tokenizer at run time — from a committed BPE-ranks cache (OpenAI), a vendor
API (Anthropic), or a first-party download (Google, Meta, Alibaba). This note
records where each comes from and under what terms, so a reader can reproduce the
eval without guessing, and so the licence position is stated rather than assumed.

| Counter | Tokenizer | Obtained from | Terms |
|---|---|---|---|
| `o200k_base`, `cl100k_base` | tiktoken BPE ranks | committed under `eval/tiktoken_cache/` | [tiktoken](https://github.com/openai/tiktoken), MIT |
| `claude-*` (6 counters) | none published | Anthropic `count_tokens` API | [Anthropic usage policies](https://www.anthropic.com/legal/aup); the endpoint is free and unbilled |
| `gemini-3-1-pro`, `gemini-3-5-flash`, `gemini-3-1-flash-lite` | `gemma4` (one artifact, all three) | `google/gemma-4-E4B-it` via the google-genai SDK | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| `llama-4` | Llama 4 tokenizer | `meta-llama/Llama-4-Scout-17B-16E` (**gated**; needs `HF_TOKEN` and licence acceptance) | [Llama 4 Community License](https://www.llama.com/llama4/license/) |
| `qwen-3-8`, `qwen-3-6` | Qwen tokenizer (one artifact, both counters — vocabulary and merge rules verified byte-identical 2026-08-16) | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-27B` (both ungated) | Apache-2.0 |

**Only token counts are published here** — never vocabularies, merges, or decoded
token strings. The `TokenCounter` contract is `count(text) -> int` precisely
because closed models expose nothing more, and the open ones are held to the same
narrow surface.

**Corpora** have their own notices: `flores200_dataset/ATTRIBUTION.md` (FLORES-200,
CC BY-SA 4.0) and `eval/massive/PROVENANCE.md` (MASSIVE slice, CC BY 4.0).
