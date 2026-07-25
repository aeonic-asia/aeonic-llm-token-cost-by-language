"""Aeonic token-cost eval — configuration (languages, model matrix, pricing).

This is the single binding spec for the eval. Everything downstream (run.py,
analyze.py, figures.py) reads scope from here. Scope is locked for the
*Vietnamese Token Tax* article: five languages, premium measured vs. English,
current-generation model matrix, dual-corpus (FLORES+ and MASSIVE, both live).

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

# Message-envelope calibration. Anthropic's `count_tokens` counts the fully
# rendered chat prompt, so a fixed turn/role frame sits on top of the content
# tokens (measured: 6 tokens for the newer Claude tokenizer, 7 for the older).
# The offline counters (tiktoken / gemma / Llama) count bare text with no frame,
# so a raw comparison would inflate Claude's *per-sentence* counts by that fixed
# floor — negligible on the whole-corpus aggregate (one call over ~10^5 tokens),
# but material on short sentences (a true 2.0x premium reads ~1.6x when +6 lands
# on both sides of the ratio). The driver measures this floor per API counter and
# subtracts it from the per-sentence counts so the distribution is comparable to
# the offline counters — measured, never assumed. A lone ASCII character is
# exactly one token in every tokenizer here (BPE never merges a single char), so
# the floor = count(probe) - 1. Empty content is rejected by the API, hence the
# single-char probe. The aggregate is left uncorrected (fixed frame is <0.01% of
# a ~10^5-token concatenated call) and stays the paper-style raw count.
ENVELOPE_PROBE = "x"
ENVELOPE_PROBE_TOKENS = 1

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
    # HF (`kind="hf"`) only: does the repo require accepting a license / an HF
    # token? Gated repos (Meta Llama) need HF_TOKEN; ungated Apache-2.0 repos
    # (Qwen) download with no credentials. Drives the token gate in measure.py so
    # an ungated counter isn't wrongly skipped for a missing token.
    gated: bool = False
    init_kwargs: dict = field(default_factory=dict)
    # ── figure display (headline charts) ────────────────────────────────────
    # The full matrix carries proxy/duplicate counters (e.g. the three models
    # that share one Claude tokenizer, both Gemini generations) so the eval can
    # *verify* equivalence in-dataset. The headline figures collapse those to one
    # column per distinct tokenizer to stay readable. These fields drive that
    # collapse; nothing here affects measurement — only what the charts show.
    headline: bool = True     # own column in the headline figures?
    headline_display: str = ""  # short chart/legend label (falls back to display)
    flagship_group: str = ""  # id of the headline column a folded counter maps to
    fold_reason: str = ""     # why folded: "shared" (byte-identical tokenizer)
    #                           | "superseded" (older flagship, near-identical)


# The trimmed flagship matrix. ≈7 counters: one flagship per provider + both
# Claude tokenizer generations + an open-weight representative + historical
# OpenAI baselines. Counting is free, so the trim is for table legibility.
MODEL_MATRIX: list[Counter] = [
    # OpenAI — offline via tiktoken, real now.
    Counter("o200k_base", "GPT-5.6 (o200k_base)", "OpenAI", "tiktoken",
            STATUS_LIVE, spec="o200k_base", stands_in_for="gpt-5.6",
            headline_display="GPT-5.6"),
    Counter("cl100k_base", "cl100k_base (GPT-4/3.5 era)", "OpenAI", "tiktoken",
            STATUS_LIVE, spec="cl100k_base",
            stands_in_for="historical OpenAI baseline + validation oracle",
            headline_display="cl100k (2023)"),
    # Anthropic — count_tokens API, both tokenizer generations. Plumbed; runs
    # once ANTHROPIC_API_KEY is set. Exposes the ~30–41% within-vendor jump.
    # The newer tokenizer is shared by Opus 5 / Opus 4.8 / Sonnet 5 / Fable 5, so
    # it is the flagship column those fold into (verified byte-identical below).
    # Headline label names the CURRENT flagship (Opus 5); the committed hook
    # numbers were measured on the claude-opus-4-8 endpoint (spec below), and the
    # claude-opus-5 fold-proxy confirms in-dataset that Opus 5 counts identically —
    # so the generational id (release-stable) and the flagship label stay decoupled.
    Counter("claude-new", "Claude (newer tokenizer)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-new", spec="claude-opus-4-8",
            stands_in_for="shared newer Claude tokenizer — Opus 5 (current flagship) / "
                          "Opus 4.8 / Sonnet 5 / Fable 5; measured on the Opus 4.8 endpoint",
            headline_display="Claude Opus 5"),
    Counter("claude-old", "Claude (older tokenizer)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-old", spec="claude-sonnet-4-6",
            stands_in_for="older Claude tokenizer baseline",
            headline_display="Claude Sonnet 4.6"),
    # Sonnet 5 shares the newer Claude tokenizer with Opus 4.8. Included so the
    # coincidence check *confirms* that in the committed dataset (identical
    # counts across all languages), not just by assertion. Folds into claude-new.
    Counter("claude-sonnet-5", "Claude Sonnet 5 (newer, shared)", "Anthropic",
            "anthropic", STATUS_NEEDS_KEY, generation="claude-new",
            spec="claude-sonnet-5", stands_in_for="shared newer Claude tokenizer (verify)",
            headline=False, headline_display="Sonnet 5",
            flagship_group="claude-new", fold_reason="shared"),
    # Fable 5 was assumed (source research) to share the newer Claude tokenizer
    # with Opus 4.8 / Sonnet 5; now CONFIRMED in-dataset — the coincidence check
    # finds claude-fable-5 ≡ claude-new ≡ claude-sonnet-5 byte-identical across
    # all five languages in BOTH corpora (see summary.json shared_tokenizer_pairs).
    Counter("claude-fable-5", "Claude Fable 5 (newer, shared — confirmed)", "Anthropic",
            "anthropic", STATUS_NEEDS_KEY, generation="claude-new",
            spec="claude-fable-5", stands_in_for="shared newer Claude tokenizer (confirmed)",
            headline=False, headline_display="Fable 5",
            flagship_group="claude-new", fold_reason="shared"),
    # Opus 5 (released 2026-07-24) is the current everyday Claude flagship. Docs
    # place every Claude 4.7+ model on the newer tokenizer, so it is expected to
    # share Opus 4.8's tokenizer — NOT assumed here but MEASURED via count_tokens
    # and confirmed byte-identical to claude-new in the coincidence check. Folds
    # into claude-new; priced $5/1M in (= Opus 4.8), so "same tokens, same price"
    # as the model the headline column was historically measured on.
    Counter("claude-opus-5", "Claude Opus 5 (newer, shared — confirm)", "Anthropic",
            "anthropic", STATUS_NEEDS_KEY, generation="claude-new",
            spec="claude-opus-5", stands_in_for="current newer-Claude flagship; confirms shared tokenizer",
            headline=False, headline_display="Opus 5",
            flagship_group="claude-new", fold_reason="shared"),
    # Haiku 4.5 — the cheapest Claude serving tier ($1/1M in). Which tokenizer
    # generation it uses was NOT assumed: measured via count_tokens, it is
    # byte-identical to claude-old (Sonnet 4.6) across all five languages in BOTH
    # corpora (see summary.json shared_tokenizer_pairs). So the *older* Claude
    # tokenizer spans Sonnet 4.6 + Haiku 4.5 — folds into claude-old, at 1/3 the price.
    Counter("claude-haiku-4-5", "Claude Haiku 4.5", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-old", spec="claude-haiku-4-5",
            stands_in_for="shared older Claude tokenizer (confirmed ≡ Sonnet 4.6); cheapest Claude tier",
            headline=False, headline_display="Haiku 4.5",
            flagship_group="claude-old", fold_reason="shared"),
    # Gemini — offline LocalTokenizer (google-genai 2.12.1). Google DOES have a
    # within-vendor tokenizer split, at the 3.0 -> 3.1 boundary (verified against
    # the SDK's own _local_tokenizer_loader model->tokenizer map):
    #   gemma3  <- Gemini 2.0 / 2.5 / 3.0 (gemini-3-pro-preview, gemini-3-flash-preview)
    #   gemma4  <- Gemini 3.1 / 3.5 / 4    (gemini-3.1-pro-preview, gemini-3.5-flash, ...)
    # Both load offline (gemma3 via a pinned URL; gemma4 via HF google/gemma-4-E4B-it,
    # unauthenticated download OK). We measure BOTH Pro generations: gemma3 = the 3.0
    # Pro tokenizer, gemma4 = the CURRENT Pro flagship (3.1 Pro). No key.
    # gemma3 (3.0-era Pro) is superseded by gemma4 (3.1 Pro, current flagship) and
    # within ~0.01% of it — it folds into the 3.1 Pro column as "superseded", NOT
    # "shared" (the two are distinct tokenizers: gemma3≠gemma4 in Vietnamese/FLORES).
    Counter("gemini-3-pro", "Gemini 3 Pro (gemma3)", "Google", "gemini_local",
            STATUS_NEEDS_SDK, spec="gemini-3-pro-preview",
            generation="gemini-gemma3",
            stands_in_for="Gemini 2.0/2.5/3.0 'gemma3' tokenizer (superseded by gemma4 at 3.1)",
            headline=False, headline_display="Gemini 3 Pro",
            flagship_group="gemini-3-1-pro", fold_reason="superseded"),
    Counter("gemini-3-1-pro", "Gemini 3.1 Pro (gemma4)", "Google", "gemini_local",
            STATUS_NEEDS_SDK, spec="gemini-3.1-pro-preview",
            generation="gemini-gemma4",
            stands_in_for="current Google Pro flagship — Gemini 3.1/3.5/4 'gemma4' tokenizer",
            headline_display="Gemini 3.1 Pro"),
    # Open-weight representatives — HuggingFace AutoTokenizer. Content-token count
    # (no BOS/EOS), matching the tiktoken counters.
    #   Llama 4 Scout — GATED Meta repo: needs an HF access token (HF_TOKEN).
    #   Qwen 3.6      — UNGATED Apache-2.0 repo: downloads with no credentials.
    # Both need `transformers` installed (documented add-on in requirements-eval.txt);
    # a default keyless/transformers-less `make reproduce` carries their committed
    # rows forward. `gated` drives the token gate in measure.py so Qwen isn't
    # wrongly skipped for a missing HF token.
    Counter("llama-4", "Llama 4 Scout (open-weight)", "Meta", "hf",
            STATUS_NEEDS_KEY, spec="meta-llama/Llama-4-Scout-17B-16E",
            gated=True, headline_display="Llama 4"),
    # Qwen 3.6 (Alibaba, released 2026-04; open-weight, Apache-2.0). A NEW, larger
    # tokenizer — ~248k vocab vs. Qwen3/2.5's ~152k — so it counts multilingual
    # text differently and earns its own headline column (the first non-incumbent
    # vendor in the matrix). Self-host: no single per-token serving list price →
    # premium-only, like Llama 4. Cite the dense flagship repo; the MoE sibling
    # shares this tokenizer.
    Counter("qwen-3-6", "Qwen 3.6 (open-weight)", "Alibaba", "hf",
            STATUS_NEEDS_SDK, spec="Qwen/Qwen3.6-27B",
            gated=False, headline_display="Qwen 3.6"),
]

MATRIX_BY_ID = {c.id: c for c in MODEL_MATRIX}


# ── pricing (dated, sourced, confidence-flagged) ─────────────────────────────
# Cost-per-1k-chars = (tokens per 1k NFC chars, measured) × (USD per token).
# The token side is always real measurement; only the $ side lives here.
# UNVERIFIED prices are None → cost is omitted for that counter, never guessed.
PRICING_AS_OF = "2026-07-18"
# Interbank mid-market USD/VND. Source: TradingEconomics / Wise interbank quote
# for the as-of date (prior-week range 26,249–26,305). Confidence: medium — a
# reference mid-rate, not a specific bank's card rate; cost figures scale
# linearly with it, so treat VND as indicative and USD as the primary unit.
USD_TO_VND = 26_295.0
USD_TO_VND_AS_OF = "2026-07-17"

# Measurement date of the committed token dataset — kept distinct from
# PRICING_AS_OF (the $ layer can be re-ratified without re-measuring tokens, and
# vice versa; stamping one with the other would assert a false provenance date).
DATASET_AS_OF = "2026-07-19"


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
                        "Newer-Claude tokenizer; current flagship Claude Opus 5 and "
                        "Opus 4.8 share the same $5.00/1M in ($25.00/1M out) input list price"),
    "claude-old": Price(3.00, PRICING_AS_OF, "high",
                        "Claude Sonnet 4.6 input list price ($3.00/1M in, $15.00/1M out)"),
    "claude-sonnet-5": Price(3.00, PRICING_AS_OF, "high",
                             "Claude Sonnet 5 standard input list price ($3.00/1M in, "
                             "$15.00/1M out); intro $2.00/1M in effect through 2026-08-31"),
    "claude-fable-5": Price(10.00, "2026-07-19", "high",
                            "Claude Fable 5 input list price ($10.00/1M in, $50.00/1M out) — "
                            "2x Opus 4.8; shares Opus 4.8's tokenizer so identical token counts "
                            "at double the price. Verified 2026-07-19 vs. the Anthropic pricing "
                            "page + claude-api reference"),
    "claude-haiku-4-5": Price(1.00, "2026-07-19", "high",
                              "Claude Haiku 4.5 input list price ($1.00/1M in, $5.00/1M out) — "
                              "the cheapest Claude serving tier. Verified 2026-07-19 vs. the "
                              "claude-api reference"),
    # Opus 5 is deliberately UNPRICED here — not because the price is unknown
    # ($5.00/1M in, $25.00/1M out, verified 2026-07-25 vs. the Anthropic pricing
    # page) but because it is *identical* to the claude-new headline column this
    # counter folds into, which already carries that $5.00 and is labelled "Claude
    # Opus 5". The dollar figures deliberately do NOT fold shared-tokenizer models,
    # since their whole point is the price split within one tokenizer (Sonnet 5 $3
    # / Fable 5 $10 / Haiku 4.5 $1 all differ, so all are priced). Opus 5 differs
    # in neither tokens nor price, so pricing it would draw a duplicate bar at an
    # identical height. This counter's job is the in-dataset tokenizer
    # confirmation; the $ story is told once, by claude-new.
    "claude-opus-5": Price(None, "2026-07-25", "high",
                           "premium-only by design: same tokenizer AND same $5.00/1M input price as "
                           "the claude-new headline column it folds into (Claude Opus 5, verified "
                           "2026-07-25) — priced there, not duplicated here"),
    "gemini-3-pro": Price(2.00, PRICING_AS_OF, "medium",
                          "Gemini 3.0-generation Pro (gemma3 tokenizer). Priced at the Google "
                          "Pro <=200K tier ($2.00/1M in); the 3.0 preview shares this tier price "
                          "with 3.1, but 3.0 Pro is superseded — confidence medium on the exact SKU"),
    "gemini-3-1-pro": Price(2.00, "2026-07-19", "high",
                            "Gemini 3.1 Pro (CURRENT Google Pro flagship, gemma4 tokenizer) input "
                            "list price, <=200K-context tier ($2.00/1M in; $4.00/1M above 200K). "
                            "Verified 2026-07-19 vs. the Anthropic/Google 2026 pricing comparisons"),
    "llama-4": Price(None, PRICING_AS_OF, "unknown",
                     "self-host / open-weight; no single per-token list price"),
    "qwen-3-6": Price(None, "2026-07-25", "unknown",
                      "self-host / open-weight (Apache-2.0); no single per-token list price. "
                      "Alibaba Cloud hosts distinct commercial Qwen3.6 SKUs (Plus/Max/Flash, "
                      "~$0.19–1.30/1M in) — different models from this open-weight repo, so NOT "
                      "used as its price. Premium-only, like Llama 4"),
}
