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
from dataclasses import dataclass
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
# Manifest metadata only — nothing branches on it.
# live      : runs offline with the pinned deps.
# needs_key : skipped until its credential is set (vendor count-tokens APIs).
# needs_sdk : offline, but needs an optional SDK and a one-time tokenizer download
#             (Gemini LocalTokenizer, HF transformers).
STATUS_LIVE = "live"
STATUS_NEEDS_KEY = "needs_key"
STATUS_NEEDS_SDK = "needs_sdk"

# Counter kinds that call a rate-limited vendor API. For these the driver measures
# the aggregate (one call per language) and at most API_PER_SENTENCE_SUBSAMPLE
# per-sentence calls for the distribution. Counting endpoints are free, so this is
# a latency/RPM budget, not cost.
API_KINDS = {"anthropic", "openai", "gemini_api"}

# Message-envelope calibration. A count-tokens endpoint counts the fully rendered
# request, so a fixed turn/role frame can sit on top of the content (per counter in
# run_manifest.envelope_tokens_by_counter; Claude 5.5: 8). Offline counters count
# bare text. Left uncorrected the frame inflates per-sentence counts — a true 2.0x
# premium reads ~1.6x when +8 lands on both sides of a short sentence's ratio. So every API counter measures
# its own frame (measure.ApiCounter.envelope_tokens) and the driver subtracts it
# from the per-sentence counts. count(probe) is MEASURED; the subtrahend — a lone
# character is one token — is ASSUMED, and tested across character classes rather
# than trusted. The aggregate stays the paper-style raw total (envelope-inclusive;
# the frame is <0.25% of any corpus total).
ENVELOPE_PROBE = "x"
ENVELOPE_PROBE_TOKENS = 1
# Probes span lowercase / uppercase / digit / punctuation / non-ASCII. A single
# character is never fewer than one token, so a two-token probe can only push its
# apparent frame UP: the floor is the estimator and an outlier is a fact about that
# character. Keep known outliers in the set (on the Claude tokenizer, 'Z' costs
# two) so that path stays exercised.
ENVELOPE_PROBE_ALTS = ("q", "a", "Z", "#", "é", "7", "0")
# How many probes must agree on the floor before it is trusted as the frame.
ENVELOPE_MIN_AGREEING = 4
# Sanity bound: a frame outside 0..N measured something other than a turn frame.
ENVELOPE_MAX_PLAUSIBLE = 64

API_PER_SENTENCE_SUBSAMPLE = 200  # per-sentence calls per API counter for the
# premium distribution. Deterministic HEAD slice sentences[:N] — re-runs are
# byte-identical, but the slice is topically clustered, not a random sample;
# premium_by_language.csv records the basis per row. Offline counters sweep fully.

# Retry budget and per-request timeout for every vendor SDK client. SDK defaults
# absorb a blip, not a sustained rate limit across thousands of sequential calls,
# and a minutes-long default timeout multiplied by retries only describes a hung
# connection. Let the SDKs retry (they honour `retry-after`); size them here.
# If the budget is exhausted the counter's pass rolls back and run.py exits
# non-zero, so a stale counter cannot pass for a clean credentialed run.
API_MAX_RETRIES = 8
API_TIMEOUT_S = 30.0
# Concurrent per-sentence calls per API counter. Calls are latency-bound (~1s on
# OpenAI's endpoint), so a sequential pass took hours. 16 stays far below every
# vendor's published count-tokens RPM; the SDK retry budget absorbs any 429.
API_CONCURRENCY = 16

@dataclass(frozen=True)
class Counter:
    """One token counter in the matrix.

    `kind` selects the implementation in measure.py. `stands_in_for` documents
    what a counter is evidence for when it is not itself a sold model.
    """
    id: str
    display: str
    provider: str
    kind: str                 # tiktoken | anthropic | openai | gemini_api | gemini_local | hf
    status: str
    spec: str = ""            # tiktoken encoding, vendor model id, or HF repo id
    stands_in_for: str = ""   # model whose tokenizer this measures, if a proxy
    # HF (`kind="hf"`) only: does the repo require accepting a license / an HF
    # token? Gated repos (Meta Llama) need HF_TOKEN; ungated Apache-2.0 repos
    # (Qwen) download with no credentials. Drives the token gate in measure.py so
    # an ungated counter isn't wrongly skipped for a missing token.
    gated: bool = False
    # ── figure display ──────────────────────────────────────────────────────
    # Headline figures draw one column per distinct tokenizer; folded counters
    # map onto it. Nothing here affects measurement.
    headline: bool = True     # own column in the headline figures?
    headline_display: str = ""  # short chart/legend label (falls back to display)
    family_display: str = ""    # heatmap column label naming every model the
    #                             column covers (falls back to headline_display)
    flagship_group: str = ""  # id of the headline column a folded counter maps to
    fold_reason: str = ""     # why folded — "shared" (byte-identical tokenizer,
    #                           re-verified against the counts before any caption
    #                           claims it) is the only supported value.


# The model matrix — one HEADLINE column per distinct tokenizer, each measured on
# the model it names, plus fold proxies: the vendor's other price tiers (which
# draw their own bars in the dollar figures) and offline proofs that tie an API
# model to a published tokenizer artifact. Every fold is a hypothesis the
# coincidence check confirms in-dataset before a caption may claim it.
#
# Scope (refreshed 2026-10-19): each vendor's CURRENT lineup, plus cl100k_base as
# the single historical anchor. Superseded models are dropped rather than kept as
# proxies; their history lives in git and the workshop decision log.
MODEL_MATRIX: list[Counter] = [
    # Historical anchor (GPT-4 era) and the paper oracle — offline via tiktoken.
    Counter("cl100k_base", "cl100k_base (GPT-4 era)", "OpenAI", "tiktoken",
            STATUS_LIVE, spec="cl100k_base",
            stands_in_for="historical OpenAI baseline + validation oracle",
            headline_display="GPT-4 (2023)"),
    # OpenAI — GPT-6, measured through the Responses API input-token count:
    # tiktoken maps no GPT-6 model and OpenAI has not named its tokenizer.
    Counter("gpt", "GPT-6.1 Sol", "OpenAI", "openai", STATUS_NEEDS_KEY,
            spec="gpt-6.1-sol", headline_display="GPT-6.1 Sol",
            family_display="GPT-6 Luna / 6.1 Sol / Astra"),
    Counter("gpt-6-astra", "GPT-6 Astra (shared)", "OpenAI", "openai",
            STATUS_NEEDS_KEY, spec="gpt-6-astra", headline=False,
            headline_display="GPT-6 Astra", flagship_group="gpt", fold_reason="shared"),
    Counter("gpt-6-luna", "GPT-6 Luna (shared)", "OpenAI", "openai",
            STATUS_NEEDS_KEY, spec="gpt-6-luna", headline=False,
            headline_display="GPT-6 Luna", flagship_group="gpt", fold_reason="shared"),
    # Offline proof: does GPT-6 tokenize as the published o200k_base (GPT-5 era)?
    Counter("o200k_base", "o200k_base (GPT-5 era)", "OpenAI", "tiktoken",
            STATUS_LIVE, spec="o200k_base", headline=False,
            stands_in_for="hypothesis: GPT-6 shares the GPT-5 tokenizer",
            headline_display="o200k_base", flagship_group="gpt", fold_reason="shared"),
    # Anthropic — count_tokens API. Anthropic documents every current model on the
    # tokenizer introduced with Claude Opus 4.7 and publishes no artifact for it.
    Counter("claude", "Claude Opus 5.5", "Anthropic", "anthropic", STATUS_NEEDS_KEY,
            spec="claude-opus-5-5", headline_display="Claude Opus 5.5",
            family_display="Claude 5.5 / Fable 5.1"),
    Counter("claude-fable-5-1", "Claude Fable 5.1 (shared)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, spec="claude-fable-5-1", headline=False,
            headline_display="Fable 5.1", flagship_group="claude", fold_reason="shared"),
    Counter("claude-sonnet-5-5", "Claude Sonnet 5.5 (shared)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, spec="claude-sonnet-5-5", headline=False,
            headline_display="Sonnet 5.5", flagship_group="claude", fold_reason="shared"),
    Counter("claude-haiku-5-5", "Claude Haiku 5.5 (shared)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, spec="claude-haiku-5-5", headline=False,
            headline_display="Haiku 5.5", flagship_group="claude", fold_reason="shared"),
    # Google — Gemini API countTokens: the SDK's offline table maps no model newer
    # than 3.5 Flash, and assigning one by hand would assert an unpublished fact.
    # TWO columns, measured 2026-10-19: Gemini 4 Argon tokenizes differently from
    # the published gemma4 vocabulary (37% of sentences differ, both directions),
    # while the Flash line matches it on MASSIVE; on FLORES it differs only on the
    # HTML-bearing vie_Latn 429 (summary.json: coincident_in_some_corpora).
    Counter("gemini", "Gemini 4 Argon", "Google", "gemini_api", STATUS_NEEDS_KEY,
            spec="gemini-4-argon", headline_display="Gemini 4 Argon"),
    Counter("gemini-3-8-flash", "Gemini 3.8 Flash", "Google", "gemini_api",
            STATUS_NEEDS_KEY, spec="gemini-3.8-flash",
            headline_display="Gemini 3.8 Flash", family_display="Gemini Flash / Flash-Lite"),
    Counter("gemini-3-5-flash-lite", "Gemini 3.5 Flash-Lite (shared)", "Google",
            "gemini_api", STATUS_NEEDS_KEY, spec="gemini-3.5-flash-lite", headline=False,
            headline_display="Gemini 3.5 Flash-Lite", flagship_group="gemini-3-8-flash",
            fold_reason="shared"),
    # Offline proof that the Flash line is the published gemma4 tokenizer, which
    # Google's Gemma reports state is unchanged from Gemini 2.0. `spec` is a key
    # into the SDK's model->tokenizer table, not an endpoint. Its FLORES Vietnamese
    # total sits 8 tokens off the API's: all of it is the corpus's one line of HTML
    # (vie_Latn 429, `km<sup>2</sup>`: 104 via the API, 112 offline), so the
    # coincidence check lists the pair as coincident on MASSIVE only. Not a
    # language finding.
    Counter("gemma4", "gemma4 tokenizer (offline)", "Google", "gemini_local",
            STATUS_NEEDS_SDK, spec="gemini-3.5-flash", headline=False,
            stands_in_for="the Gemini Flash line's published tokenizer",
            headline_display="gemma4", flagship_group="gemini-3-8-flash",
            fold_reason="shared"),
    # Open-weight — Qwen 3.8 (ungated Apache-2.0; needs `transformers`, no token).
    # Content tokens only (no BOS/EOS), matching tiktoken.
    Counter("qwen", "Qwen 3.8 (open-weight)", "Alibaba", "hf", STATUS_NEEDS_SDK,
            spec="Qwen/Qwen3.8-27B", gated=False, headline_display="Qwen 3.8"),
]

MATRIX_BY_ID = {c.id: c for c in MODEL_MATRIX}


def _check_matrix_integrity() -> None:
    """Fail at import on the mistakes the documented extension path invites.

    Adding a counter means copy-pasting a `Counter(...)`. Forget to change the
    `id` and MATRIX_BY_ID silently collapses the pair while run.py iterates the
    *list* and measures both — surfacing much later as
    "ValueError: Index contains duplicate entries, cannot reshape" from a pivot,
    naming neither the counter nor the duplication. Likewise a renamed id that
    misses PRICING drops the model out of both dollar figures AND makes the
    caption assert it has no serving list price — a false claim in a published
    asset, produced by a rename.
    """
    seen: set[str] = set()
    dupes = sorted({c.id for c in MODEL_MATRIX if c.id in seen or seen.add(c.id)})
    if dupes:
        raise ValueError(f"duplicate counter id(s) in MODEL_MATRIX: {dupes} — "
                         "each Counter needs a unique id")
    orphan_prices = sorted(set(PRICING) - set(MATRIX_BY_ID))
    if orphan_prices:
        raise ValueError(f"PRICING keys with no counter in MODEL_MATRIX: "
                         f"{orphan_prices} — a rename left the price behind")
    unpriced = sorted(set(MATRIX_BY_ID) - set(PRICING))
    if unpriced:
        raise ValueError(f"counters with no PRICING entry: {unpriced} — add an "
                         "explicit Price(None, ..., 'unknown') to state that the "
                         "omission is deliberate rather than an oversight")
    bad_folds = sorted(c.id for c in MODEL_MATRIX
                       if c.flagship_group and c.flagship_group not in MATRIX_BY_ID)
    if bad_folds:
        raise ValueError(f"counters folding into a non-existent flagship_group: {bad_folds}")
    bad_reason = sorted({c.fold_reason for c in MODEL_MATRIX
                         if c.fold_reason and c.fold_reason != "shared"})
    if bad_reason:
        raise ValueError(f"unsupported fold_reason(s): {bad_reason} — only 'shared' "
                         "is supported, and it is re-verified against the counts")
    missing_oracle = sorted(set(PAPER_CL100K_FLORES) - set(LANGUAGES))
    if missing_oracle:
        raise ValueError(f"oracle languages absent from LANGUAGES: {missing_oracle} — "
                         "the cl100k validation gate would silently check fewer "
                         "languages and report a tighter delta")


# ── pricing (dated, sourced, confidence-flagged) ─────────────────────────────
# Cost-per-1k-chars = (tokens per 1k NFC chars, measured) × (USD per token).
# The token side is always real measurement; only the $ side lives here.
# UNVERIFIED prices are None → cost is omitted for that counter, never guessed.
# Layer-level stamp: when the whole pricing layer was last re-verified. Each row
# still carries its own `as_of`; a single-row correction leaves this alone.
PRICING_AS_OF = "2026-10-19"
# Mid-market USD/VND (Wise converter, read 2026-10-19). Confidence: medium — a
# reference rate, not a bank's card rate; treat VND as indicative, USD as primary.
USD_TO_VND = 25_890.0
USD_TO_VND_AS_OF = "2026-10-19"

# Scope stamp of the committed dataset, distinct from PRICING_AS_OF (either layer
# can move without the other).
DATASET_AS_OF = "2026-10-19"   # current-lineup refresh; superseded models dropped
# NOTE this is the dataset's SCOPE stamp, not a measurement date. Per-row
# measurement dates live in raw_counts/aggregate_counts `measured_on` (written by
# run.py from the run date) and are summarised in
# run_manifest.measured_on_by_counter. Cite those, not this.

# ── validation oracle (single source of truth) ───────────────────────────────
# The paper's Table 1 cl100k_base premiums (Petrov et al., arXiv:2305.15425),
# measured on FLORES. Keyed by LANGUAGE CODE, not display name: keying on the
# display string meant renaming a language in LANGUAGES silently dropped that
# language from the oracle, which then reported a *tighter* max delta computed
# over fewer checks and still passed. Both the analyze-time check and
# tests/test_oracle.py read these, so the two implementations of the same gate
# cannot drift apart.
PAPER_CL100K_FLORES: dict[str, float] = {
    "vie_Latn": 2.45,
    "zho_Hans": 1.91,
    "deu_Latn": 1.58,
}
# The paper reports 2 dp, so half a step is the tightest defensible tolerance.
ORACLE_TOL = 0.005


@dataclass(frozen=True)
class Price:
    input_usd_per_mtok: Optional[float]   # USD per 1M input tokens; None = unverified
    as_of: str
    confidence: str                       # high | medium | low | unknown
    source: str


# Input list prices per 1M tokens (serving price — the counting endpoints are
# free). Each row carries its own verification date. Rules:
#  * TODAY's list price, as of the row's date: the dataset is a dated snapshot and
#    is not re-priced later. A vendor's announced dated step-up is noted in the
#    source string, not tracked.
#  * Unpriced (None) means there is no list price to record — a self-hosted model,
#    a tokenizer with no SKU, or an offline proof whose bar its API model draws.
#  * The cheapest context tier is the right one: every request here is one
#    sentence, far below any long-context boundary.
PRICING: dict[str, Price] = {
    "cl100k_base": Price(None, "2026-10-19", "unknown",
                         "historical tokenizer; no current SKU is served on it"),
    "gpt": Price(2.00, "2026-10-19", "high",
                 "GPT-6.1 Sol (`gpt-6.1-sol`) $2.00 in / $10.00 out — "
                 "developers.openai.com/api/docs/pricing"),
    "gpt-6-astra": Price(10.00, "2026-10-19", "high",
                         "GPT-6 Astra (`gpt-6-astra`) $10.00 in / $50.00 out — "
                         "developers.openai.com/api/docs/pricing"),
    "gpt-6-luna": Price(0.10, "2026-10-19", "high",
                        "GPT-6 Luna (`gpt-6-luna`) $0.10 in / $0.50 out — "
                        "developers.openai.com/api/docs/pricing"),
    "o200k_base": Price(None, "2026-10-19", "unknown",
                        "offline tokenizer proof; the GPT-6 rows carry the prices"),
    "claude": Price(4.00, "2026-10-19", "high",
                    "Claude Opus 5.5 (`claude-opus-5-5`) $4.00 in / $20.00 out — "
                    "platform.claude.com models overview"),
    "claude-fable-5-1": Price(10.00, "2026-10-19", "high",
                              "Claude Fable 5.1 (`claude-fable-5-1`) $10.00 in / $50.00 "
                              "out — platform.claude.com models overview"),
    "claude-sonnet-5-5": Price(2.00, "2026-10-19", "high",
                               "Claude Sonnet 5.5 (`claude-sonnet-5-5`) $2.00 in / $10.00 "
                               "out — platform.claude.com models overview"),
    "claude-haiku-5-5": Price(0.10, "2026-10-19", "high",
                              "Claude Haiku 5.5 (`claude-haiku-5-5`) $0.10 in / $0.50 out "
                              "for prompts up to 100K tokens ($0.50 / $2.50 above) — "
                              "platform.claude.com Haiku 5.5 overview"),
    "gemini": Price(2.00, "2026-10-19", "high",
                    "Gemini 4 Argon (`gemini-4-argon`) $2.00 in / $10.00 out until "
                    "2026-11-30; $4.00 / $20.00 from 2026-12-01 — "
                    "ai.google.dev/gemini-api/docs/pricing"),
    "gemini-3-8-flash": Price(0.75, "2026-10-19", "high",
                              "Gemini 3.8 Flash (`gemini-3.8-flash`) $0.75 in / $3.75 "
                              "out until 2026-12-31; $1.50 / $7.50 from 2027-01-01 — "
                              "ai.google.dev/gemini-api/docs/pricing"),
    "gemini-3-5-flash-lite": Price(0.30, "2026-10-19", "high",
                                   "Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) $0.30 "
                                   "in / $2.50 out (text) — "
                                   "ai.google.dev/gemini-api/docs/pricing"),
    "gemma4": Price(None, "2026-10-19", "unknown",
                    "offline tokenizer proof; the Gemini Flash API rows carry the prices"),
    "qwen": Price(None, "2026-10-19", "unknown",
                  "self-hosted open-weight model; no single per-token list price"),
}


# Defined above PRICING but called here, once both are in scope.
_check_matrix_integrity()
