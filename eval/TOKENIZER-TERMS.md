# Tokenizer provenance and terms

No tokenizer artifact is redistributed in this repository. Every counter resolves
its tokenizer at run time — from a committed BPE-ranks cache (OpenAI), a vendor
count-tokens API (Anthropic, OpenAI, Google), or a first-party download (Google,
Alibaba). This note records where each comes from and under what terms, so a
reader can reproduce the eval without guessing, and so the licence position is
stated rather than assumed.

| Counter | Tokenizer | Obtained from | Terms |
|---|---|---|---|
| `o200k_base`, `cl100k_base` | tiktoken BPE ranks | committed under `eval/tiktoken_cache/` | [tiktoken](https://github.com/openai/tiktoken), MIT |
| `claude`, `claude-*` (4 counters) | none published | Anthropic `count_tokens` API | [Anthropic usage policies](https://www.anthropic.com/legal/aup); the endpoint is free and unbilled |
| `gpt`, `gpt-6-*` (3 counters) | not named by OpenAI (totals match `o200k_base`) | OpenAI Responses API input-token count | [OpenAI usage policies](https://openai.com/policies/usage-policies); the endpoint is free |
| `gemini`, `gemini-3-*` (3 counters) | not published | Gemini API `countTokens` | [Gemini API terms](https://ai.google.dev/gemini-api/terms); the endpoint is free |
| `gemma4` | `gemma4` (offline proof for the Gemini Flash line) | `google/gemma-4-E4B-it` via the google-genai SDK | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| `qwen` | Qwen tokenizer | `Qwen/Qwen3.8-27B` (ungated) | Apache-2.0 |

**Only token counts are published here** — never vocabularies, merges, or decoded
token strings. The `TokenCounter` contract is `count(text) -> int` precisely
because closed models expose nothing more, and the open ones are held to the same
narrow surface.

**Corpora** have their own notices: `flores200_dataset/ATTRIBUTION.md` (FLORES-200,
CC BY-SA 4.0) and `eval/massive/PROVENANCE.md` (MASSIVE slice, CC BY 4.0).
