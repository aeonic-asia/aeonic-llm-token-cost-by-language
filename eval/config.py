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
# needs_key : implemented + plumbed; skipped until an API key is present
#             (Anthropic count_tokens — the article's headline hook).
# deferred  : interface slot reserved; not wired to live downloads this session
#             (exact model IDs are medium-confidence; verify at build time).
STATUS_LIVE = "live"
STATUS_NEEDS_KEY = "needs_key"
STATUS_DEFERRED = "deferred"

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
    # Gemini — offline LocalTokenizer (Gemma family). Deferred: exact 2026 IDs
    # and local-vs-hosted parity to verify at build time.
    Counter("gemini-3.1-pro", "Gemini 3.1 Pro (local)", "Google", "gemini_local",
            STATUS_DEFERRED, spec="gemini-3.1-pro"),
    # Open-weight representative — HuggingFace AutoTokenizer, gated. Deferred:
    # needs HF token + verified 2026 model id.
    Counter("llama-4", "Llama 4 (open-weight)", "Meta", "hf",
            STATUS_DEFERRED, spec="meta-llama/Llama-4-Scout-17B-16E"),
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


# Input prices per 1M tokens. Populated only where defensibly dated/sourced;
# forward-dated flagships stay None until ratified in the draft step (5-3),
# so the committed cost table reports premiums-only for them rather than
# inventing a price. Kept here as the single edit point for price ratification.
PRICING: dict[str, Price] = {
    "o200k_base": Price(None, PRICING_AS_OF, "unknown",
                        "GPT-5.6 list price unconfirmed; ratify in draft step"),
    "cl100k_base": Price(None, PRICING_AS_OF, "unknown",
                         "historical baseline tokenizer; not a current serving SKU"),
    "claude-new": Price(None, PRICING_AS_OF, "unknown",
                        "set at ratification alongside the API run"),
    "claude-old": Price(None, PRICING_AS_OF, "unknown",
                        "set at ratification alongside the API run"),
    "gemini-3.1-pro": Price(None, PRICING_AS_OF, "unknown", "deferred"),
    "llama-4": Price(None, PRICING_AS_OF, "unknown", "self-host; no per-token list price"),
}
