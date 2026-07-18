"""Aeonic token-cost eval — configuration (languages, model matrix, pricing).

This is the single binding spec for the eval. Everything downstream (run.py,
analyze.py, figures.py) reads scope from here. Scope is locked for the
*Vietnamese Token Tax* article: five languages, premium measured vs. English,
current-generation model matrix, dual-corpus (FLORES+ now, MASSIVE deferred).

Honesty rules baked in:
  * Every counter declares an availability `status`; nothing is fabricated.
    Offline counters run now; API/gated counters are plumbed but skipped until
    credentials/IDs are supplied (see status values below).
  * Every price is dated and carries a `confidence`. Unverified prices are
    `None` and cost is simply not reported for them — never estimated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
FLORES_DIR = REPO_ROOT / "flores200_dataset"
MASSIVE_DIR = REPO_ROOT / "eval" / "massive"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
# tiktoken BPE-ranks cache, committed so `make reproduce` runs fully offline.
TIKTOKEN_CACHE_DIR = REPO_ROOT / "eval" / "tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

# ── languages (FLORES+ codes) ────────────────────────────────────────────────
# English is the baseline every premium is measured against, so it is flagged
# separately rather than sitting in the "contrast" set.
BASELINE_LANG = "eng_Latn"
LANGUAGES: dict[str, str] = {
    "eng_Latn": "English",
    "vie_Latn": "Vietnamese",   # lead language of the article
    "zho_Hans": "Chinese (Simplified)",
    "rus_Cyrl": "Russian",
    "deu_Latn": "German",
}

# ── corpora (dual-register) ──────────────────────────────────────────────────
# Premium is reported *per corpus* so the article can show the tax holds across
# both a formal register and the conversational one SME chatbots actually serve.
#   flores  — FLORES+ formal/encyclopedic prose (committed under flores200_dataset/)
#   massive — MASSIVE short virtual-assistant utterances (committed under
#             eval/massive/, built once by eval/build_massive.py; CC BY 4.0)
CORPORA: dict[str, str] = {"flores": "FLORES+", "massive": "MASSIVE"}

# ── counter availability ─────────────────────────────────────────────────────
# live      : runs now, fully offline, real measured data.
# needs_key : implemented + plumbed; skipped until an API key / access token is
#             present (Anthropic count_tokens — the headline hook; Llama 4's
#             gated HF repo needs an HF token).
# needs_sdk : implemented + plumbed; offline (no key), but skipped until an
#             optional SDK is installed and its one-time tokenizer asset is
#             downloaded (Gemini google-genai LocalTokenizer). Kept out of the
#             lean `requirements-eval.txt` so `make setup` stays minimal.
STATUS_LIVE = "live"
STATUS_NEEDS_KEY = "needs_key"
STATUS_NEEDS_SDK = "needs_sdk"

# Counter kinds that hit a rate-limited network API. For these, the driver
# measures the aggregate premium (one call per language) and takes at most
# API_PER_SENTENCE_SUBSAMPLE per-sentence calls for the distribution — the full
# per-sentence sweep would be ~2000 free-but-rate-limited requests per language.
# `count_tokens` is not token-billed, so this is a latency/RPM budget, not cost.
API_KINDS = {"anthropic"}
API_PER_SENTENCE_SUBSAMPLE = 200  # per-sentence calls per API counter for the
# premium *distribution* (median/p10..p90). 0 = aggregate-only. Deterministic
# head-slice sentences[:N] (no sampling), so re-runs stay byte-identical *with a
# key*; offline counters always do the full per-sentence sweep. 200 keeps the
# ~3k free-but-rate-limited count_tokens calls well inside the RPM budget while
# giving stable percentiles. Note: the slice is FLORES+ dev-split sentences.


@dataclass(frozen=True)
class Counter:
    """One token counter in the matrix.

    `kind` selects the implementation in measure.py. `generation` distinguishes
    the two Claude tokenizer generations (the within-vendor jump is the hook).
    `stands_in_for` documents when a tokenizer is measured as a proxy for a
    model whose tokenizer it shares (e.g. o200k_base for GPT-5.6).
    """
    id: str
    display: str
    provider: str
    kind: str                 # tiktoken | anthropic | hf | gemini_local
    status: str
    generation: str = ""      # e.g. "claude-old" / "claude-new"; "" if n/a
    spec: str = ""            # tiktoken encoding name, HF id, or Anthropic model id
    stands_in_for: str = ""   # model whose tokenizer this measures, if a proxy
    init_kwargs: dict = field(default_factory=dict)


# The trimmed flagship matrix. ≈7 counters: one flagship per provider + both
# Claude tokenizer generations + an open-weight representative + historical
# OpenAI baselines. Counting is free, so the trim is for table legibility.
MODEL_MATRIX: list[Counter] = [
    # OpenAI — offline via tiktoken, real now.
    Counter("o200k_base", "GPT-5.6 (o200k_base)", "OpenAI", "tiktoken",
            STATUS_LIVE, spec="o200k_base", stands_in_for="gpt-5.6"),
    Counter("cl100k_base", "cl100k_base (GPT-4/3.5 era)", "OpenAI", "tiktoken",
            STATUS_LIVE, spec="cl100k_base",
            stands_in_for="historical OpenAI baseline + validation oracle"),
    # Anthropic — count_tokens API, both tokenizer generations. Plumbed; runs
    # once ANTHROPIC_API_KEY is set. Exposes the ~30–41% within-vendor jump.
    Counter("claude-new", "Claude (newer tokenizer)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-new", spec="claude-opus-4-8",
            stands_in_for="shared Fable 5 / Opus 4.8 / Sonnet 5 tokenizer"),
    Counter("claude-old", "Claude (older tokenizer)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-old", spec="claude-sonnet-4-6",
            stands_in_for="older Claude tokenizer baseline"),
    # Sonnet 5 shares the newer Claude tokenizer with Opus 4.8. Included so the
    # coincidence check *confirms* that in the committed dataset (identical
    # counts across all languages), not just by assertion.
    Counter("claude-sonnet-5", "Claude Sonnet 5 (newer, shared)", "Anthropic",
            "anthropic", STATUS_NEEDS_KEY, generation="claude-new",
            spec="claude-sonnet-5", stands_in_for="shared newer Claude tokenizer (verify)"),
    # Gemini — offline LocalTokenizer (google-genai). `spec` is the SDK's exact
    # supported model string. In the shipped SDK the whole Gemini 2.0/2.5/3 line
    # maps to one `gemma3` sentencepiece tokenizer, so there is no within-vendor
    # generational split on Google's side (contrast the Claude jump). Local, no key.
    Counter("gemini-3-pro", "Gemini 3 Pro (local)", "Google", "gemini_local",
            STATUS_NEEDS_SDK, spec="gemini-3-pro-preview",
            stands_in_for="shared Gemini 2.x/3 'gemma3' tokenizer"),
    # Open-weight representative — HuggingFace AutoTokenizer. Gated repo: needs an
    # HF access token (HF_TOKEN). Content-token count (no BOS/EOS).
    Counter("llama-4", "Llama 4 Scout (open-weight)", "Meta", "hf",
            STATUS_NEEDS_KEY, spec="meta-llama/Llama-4-Scout-17B-16E"),
]

MATRIX_BY_ID = {c.id: c for c in MODEL_MATRIX}


# ── pricing (dated, sourced, confidence-flagged) ─────────────────────────────
# Cost-per-1k-chars = (tokens per 1k NFC chars, measured) × (USD per token).
# The token side is always real measurement; only the $ side lives here.
# UNVERIFIED prices are None → cost is omitted for that counter, never guessed.
PRICING_AS_OF = "2026-07-18"
USD_TO_VND = 26_100.0          # dated FX anchor; confidence: medium (mid-2026).
USD_TO_VND_AS_OF = "2026-07-18"


@dataclass(frozen=True)
class Price:
    input_usd_per_mtok: Optional[float]   # USD per 1M input tokens; None = unverified
    as_of: str
    confidence: str                       # high | medium | low | unknown
    source: str


# Input prices per 1M input tokens (serving/inference list price — NOT the token
# *counting* endpoint, which is free). Dated 2026-07-18; each carries a confidence.
# Unpriced models stay None → cost is omitted (premium-only), never invented.
# Kept here as the single edit point for price ratification.
#
# Two editorial choices flagged for review (both defensible, both easy to change here):
#  * o200k_base / GPT-5.6: the o200k tokenizer is shared across the GPT-5.6 tiers
#    (Luna $1.00 / Terra $2.50 / Sol $5.00 input); we price the *flagship* Sol tier
#    ($5.00) for a flagship-vs-flagship comparison. Confidence medium for the tier
#    pick, not the number.
#  * claude-sonnet-5: standard $3.00 used (the durable price). An introductory
#    $2.00/1M applies through 2026-08-31, then reverts to $3.00 on 2026-09-01.
PRICING: dict[str, Price] = {
    "o200k_base": Price(5.00, PRICING_AS_OF, "medium",
                        "GPT-5.6 flagship (Sol) input list price; o200k shared across "
                        "GPT-5.6 tiers (Luna $1.00 / Terra $2.50 / Sol $5.00)"),
    "cl100k_base": Price(None, PRICING_AS_OF, "unknown",
                         "historical baseline tokenizer; not a current serving SKU"),
    "claude-new": Price(5.00, PRICING_AS_OF, "high",
                        "Claude Opus 4.8 input list price ($5.00/1M in, $25.00/1M out)"),
    "claude-old": Price(3.00, PRICING_AS_OF, "high",
                        "Claude Sonnet 4.6 input list price ($3.00/1M in, $15.00/1M out)"),
    "claude-sonnet-5": Price(3.00, PRICING_AS_OF, "high",
                             "Claude Sonnet 5 standard input list price ($3.00/1M in, "
                             "$15.00/1M out); intro $2.00/1M in effect through 2026-08-31"),
    "gemini-3-pro": Price(2.00, PRICING_AS_OF, "high",
                          "Gemini 3 Pro input list price, <=200K-context tier "
                          "($2.00/1M in; $4.00/1M above 200K — our texts are short)"),
    "llama-4": Price(None, PRICING_AS_OF, "unknown",
                     "self-host / open-weight; no single per-token list price"),
}
