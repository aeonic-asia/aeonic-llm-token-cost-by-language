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
#             downloaded — Gemini (google-genai LocalTokenizer) and the *ungated*
#             HF repos (Qwen 3.6, which needs `transformers` but no credentials).
#             Kept out of the lean `requirements-eval.txt` so `make setup` stays
#             minimal.
# Note `kind="hf"` spans both statuses by design: gated repos (Llama 4) are
# needs_key because a token is genuinely required; ungated ones (Qwen) are
# needs_sdk because only the library is missing. Status is manifest metadata
# only — nothing branches on it; the real gate is `Counter.gated`.
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
# the offline counters. Precisely: count(probe) is MEASURED, and the subtrahend
# below is ASSUMED — a lone ASCII character is one token (byte-level BPE has no
# merge shorter than a character), which cannot be verified directly against a
# tokenizer whose vocabulary Anthropic does not publish. So the assumption is
# tested rather than trusted: measure.py probes several distinct single
# characters and requires them all to yield the same frame, and bounds the
# result. Empty content is rejected by the API, hence a single-char probe.
# The aggregate is left uncorrected (the fixed frame is <0.01% of a ~10^5-token
# concatenated call) and stays the paper-style raw count.
ENVELOPE_PROBE = "x"
ENVELOPE_PROBE_TOKENS = 1
# Cross-checks for the assumption above. Each must be a single ASCII character
# that no BPE can split, drawn from different classes (letter / letter / digit)
# so a class-specific surprise shows up as a disagreement rather than a silent
# constant shift in every per-sentence count.
ENVELOPE_PROBE_ALTS = ("q", "7")
# Sanity bound on the measured frame. Observed: 6 (newer Claude), 7 (older).
# A value outside this range means the probe measured something other than a
# turn/role frame — abort rather than record it as "measured".
ENVELOPE_MAX_PLAUSIBLE = 64

API_PER_SENTENCE_SUBSAMPLE = 200  # per-sentence calls per API counter for the
# premium *distribution* (median/p10..p90). 0 = aggregate-only. Deterministic
# head-slice sentences[:N] (no sampling), so re-runs stay byte-identical *with a
# key*; offline counters always do the full per-sentence sweep.
#
# Call volume, stated accurately: 6 Anthropic counters x 2 corpora x 5 languages
# x (1 aggregate + 200 per-sentence) = ~12,000 calls per full pass, plus envelope
# probes. `count_tokens` is free, so this is an RPM/latency budget, not cost —
# but there is no retry, backoff or throttle in the driver, and a single 429
# discards that counter's whole pass (carry-forward then restores its previous
# rows, so the run still succeeds with one counter silently stale). Treat a full
# credentialed pass as something to watch, not to fire and forget.
#
# Note the slice is a deterministic HEAD slice of a corpus ordered by source
# document, so it is topically clustered — fine for a stable percentile, but not
# a random sample of the corpus. premium_by_language.csv records the sample size
# per row so a reader can tell these percentiles from the full sweeps.


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


# The trimmed flagship matrix — 11 counters: one column per DISTINCT tokenizer
# (o200k, cl100k, newer Claude, older Claude, gemma4, Llama 4, Qwen 3.6) plus the
# fold proxies that prove the sharing in-dataset rather than asserting it. The
# original scope lock said "≈7 counters"; the proxies and the extra Anthropic
# price tiers took it to 11, each addition recorded in the workshop decision log.
# Counting is free, so the trim is for table legibility, not cost.
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
    # The id stays generational and release-stable; the *spec* and the *label* both
    # name the current flagship, so the column is measured on the model it claims.
    # That equality is deliberate: an earlier cut labelled this column "Opus 5"
    # while measuring the claude-opus-4-8 endpoint, which put the measured model
    # nowhere in the published figure. Opus 4.8 is now its own fold proxy below, so
    # the caption names four models and every name is backed by its own measurement.
    Counter("claude-new", "Claude (newer tokenizer)", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-new", spec="claude-opus-5",
            stands_in_for="shared newer Claude tokenizer — Opus 5 (Anthropic's recommended "
                          "default) / Opus 4.8 / Sonnet 5 / Fable 5; measured on the Opus 5 "
                          "endpoint. Note Anthropic calls Fable 5 its most capable widely "
                          "released model, so avoid 'flagship' for Opus 5 in print",
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
    # all five languages in BOTH corpora (see summary.json tokenizer_coincidence_check).
    Counter("claude-fable-5", "Claude Fable 5 (newer, shared — confirmed)", "Anthropic",
            "anthropic", STATUS_NEEDS_KEY, generation="claude-new",
            spec="claude-fable-5", stands_in_for="shared newer Claude tokenizer (confirmed)",
            headline=False, headline_display="Fable 5",
            flagship_group="claude-new", fold_reason="shared"),
    # Opus 4.8 — the prior everyday flagship, and the endpoint this column was
    # historically measured on. Retained as a fold proxy after claude-new moved to
    # the Opus 5 endpoint (2026-07-25): keeping it measured is what lets the
    # caption say "Opus 5 = Opus 4.8 = Sonnet 5 = Fable 5" with every name backed
    # by its own measurement, rather than asserting the older model's equivalence
    # from a comment. Same $5.00/1M input price as Opus 5 — see PRICING for why it
    # is deliberately left unpriced here.
    Counter("claude-opus-4-8", "Claude Opus 4.8 (newer, shared — confirmed)", "Anthropic",
            "anthropic", STATUS_NEEDS_KEY, generation="claude-new",
            spec="claude-opus-4-8",
            stands_in_for="prior newer-Claude flagship; confirms the shared tokenizer spans generations",
            headline=False, headline_display="Opus 4.8",
            flagship_group="claude-new", fold_reason="shared"),
    # Haiku 4.5 — the cheapest Claude serving tier ($1/1M in). Which tokenizer
    # generation it uses was NOT assumed: measured via count_tokens, it is
    # byte-identical to claude-old (Sonnet 4.6) across all five languages in BOTH
    # corpora (see summary.json tokenizer_coincidence_check). So the *older* Claude
    # tokenizer spans Sonnet 4.6 + Haiku 4.5 — folds into claude-old, at 1/3 the price.
    Counter("claude-haiku-4-5", "Claude Haiku 4.5", "Anthropic", "anthropic",
            STATUS_NEEDS_KEY, generation="claude-old", spec="claude-haiku-4-5",
            stands_in_for="shared older Claude tokenizer (confirmed ≡ Sonnet 4.6); cheapest Claude tier",
            headline=False, headline_display="Haiku 4.5",
            flagship_group="claude-old", fold_reason="shared"),
    # Gemini — ONE column, offline LocalTokenizer (google-genai). There is no
    # Google within-vendor tokenizer split to measure, and an earlier cut of this
    # matrix was wrong to carry two Gemini counters. Three independent lines of
    # evidence, all checked 2026-07-25:
    #
    #  1. The vocabularies are the same. The SDK resolves 2.0/2.5/3.0 to "gemma3"
    #     (hash-pinned SentencePiece from google/gemma_pytorch) and 3.1/3.5/4 to
    #     "gemma4" (HF google/gemma-4-E4B-it) — two artifacts, but each holds
    #     262,144 pieces and the ordered list of all 255,892 real text tokens is
    #     byte-identical with unchanged ids. Only 19 pieces differ each way and
    #     ALL of them are angle-bracket control tokens (gemma3's <start_of_turn>/
    #     <end_of_image>/<unusedNNNN> vs gemma4's <|tool>/<|think|>/<|audio>/...).
    #  2. Google says so. Gemma 3 tech report §2.2: "the same tokenizer as Gemini
    #     2.0 ... 262k entries". Gemma 4 tech report (arXiv:2607.02770) §2.4: "the
    #     same tokenizer as Gemini Team [2025]" — i.e. Gemini 2.5. No published
    #     tokenization change anywhere from Gemini 2.0 through Gemini 4.
    #  3. The measured difference was corpus noise. Across 20,210 per-sentence
    #     comparisons the two counters differed on exactly ONE — FLORES vie_Latn
    #     429 — and that sentence is the only line in either corpus containing HTML
    #     (`km<sup>2</sup>`), which SentencePiece keeps as single tokens and the HF
    #     BPE splits. A markup artifact meeting two loader implementations, not a
    #     property of Vietnamese.
    #
    # Independently, `gemini-3-pro-preview` was SHUT DOWN on 2026-03-09 and now
    # aliases to gemini-3.1-pro-preview (Google's deprecations page), so the second
    # column was measuring a retired id against a stale client-side lookup table,
    # and pricing it put a $2.00 row in the committed CSV for a model Google no
    # longer lists. Removed on both counts.
    #
    # Keep it one column. If a future Gemini genuinely changes tokenization, that
    # is a new headline column — establish it with a vocabulary diff, not with an
    # equality test on counts.
    Counter("gemini-3-1-pro", "Gemini 3.1 Pro (gemma4)", "Google", "gemini_local",
            STATUS_NEEDS_SDK, spec="gemini-3.1-pro-preview",
            generation="gemini-gemma4",
            stands_in_for="Google Pro flagship — Gemini 3.1/3.5/4 'gemma4' tokenizer; "
                          "text vocabulary unchanged from the gemma3 line (2.0-3.0)",
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
    # tokenizer — 248,044 vocab entries, counted directly from the downloaded
    # tokenizer.json (reproducible; the only vocab figure here that is measured
    # rather than cited). Earlier Qwen generations are widely reported at ~152k,
    # but that is a secondary-source number and is not restated as fact. The size
    # difference is not the finding anyway — the measured premiums are. Earns its
    # own headline column as the first non-incumbent vendor in the matrix.
    # Self-host: no single per-token serving list price → premium-only,
    # like Llama 4. This is the dense flagship repo; the MoE sibling is *believed*
    # to share this tokenizer but that is NOT measured here — do not state it as
    # fact, and add it as a fold proxy if the claim ever needs to be made in print.
    Counter("qwen-3-6", "Qwen 3.6 (open-weight)", "Alibaba", "hf",
            STATUS_NEEDS_SDK, spec="Qwen/Qwen3.6-27B",
            gated=False, headline_display="Qwen 3.6"),
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
DATASET_AS_OF = "2026-07-25"   # model-currency refresh: Opus 5 endpoint, Opus 4.8 proxy, Qwen 3.6


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
    # Dated 2026-07-25, not PRICING_AS_OF: this entry now makes a claim about Opus 5,
    # which did not exist on the 2026-07-18 as-of date. The price is unchanged ($5.00),
    # but stamping an Opus 5 claim with a pre-release date would assert a false
    # provenance — the same error DATASET_AS_OF exists to prevent.
    "claude-new": Price(5.00, "2026-07-25", "high",
                        "Newer-Claude tokenizer; current flagship Claude Opus 5 and "
                        "Opus 4.8 share the same $5.00/1M in ($25.00/1M out) input list "
                        "price. Verified 2026-07-25 vs. the Anthropic pricing page"),
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
    # Opus 4.8 is deliberately UNPRICED here — not because the price is unknown
    # ($5.00/1M in, $25.00/1M out, verified 2026-07-25 vs. the Anthropic pricing
    # page) but because it is *identical* to the claude-new headline column this
    # counter folds into, which already carries that $5.00 and is labelled "Claude
    # Opus 5". The dollar figures deliberately do NOT fold shared-tokenizer models,
    # since their whole point is the price split within one tokenizer (Sonnet 5 $3
    # / Fable 5 $10 / Haiku 4.5 $1 all differ, so all are priced). Opus 4.8 differs
    # in neither tokens nor price, so pricing it would put a second bar at an
    # identical height beside Opus 5's. This counter's job is the in-dataset
    # tokenizer confirmation; the $ story is told once, by claude-new.
    #
    # ⚠️ If you price it anyway, `figures._series_style` RAISES (Opus 4.8 shares
    # claude-new's colour slot and its solid texture). That is intentional — but the
    # error names a colour slot, not this decision, so read this comment first: the
    # right fix is to leave it unpriced, not to hand it a distinct slot.
    #
    # `confidence` is "unknown" to match every other unpriced row (cl100k, llama-4,
    # qwen-3-6) — the field grades a price, and there is none here. The fact that
    # the price IS known is stated above and in claude-new, not smuggled into a
    # confidence grade that lands in cost_by_language.csv as "high" beside an empty
    # price cell.
    "claude-opus-4-8": Price(None, "2026-07-25", "unknown",
                             "premium-only by design: same tokenizer AND same $5.00/1M input price as "
                             "the claude-new headline column it folds into (Claude Opus 5, verified "
                             "2026-07-25) — priced there, not duplicated here"),
    # The former "gemini-3-pro" entry is gone with its counter: gemini-3-pro-preview
    # was shut down 2026-03-09 and is absent from Google's pricing page, so its
    # $2.00 was a price for a model no longer sold.
    "gemini-3-1-pro": Price(2.00, "2026-07-25", "high",
                            "Gemini 3.1 Pro — Google's current and newest Pro model, still "
                            "'preview' status — input list price, <=200K-context tier "
                            "($2.00/1M in; $4.00/1M above 200K). Verified 2026-07-25 against "
                            "ai.google.dev/gemini-api/docs/pricing. Note it is a preview SKU, "
                            "so the price carries less notice than a stable one"),
    "llama-4": Price(None, PRICING_AS_OF, "unknown",
                     "self-host / open-weight; no single per-token list price"),
    "qwen-3-6": Price(None, "2026-07-25", "unknown",
                      "self-host / open-weight (Apache-2.0); no single per-token list price. "
                      "Alibaba Cloud hosts distinct commercial Qwen3.6 SKUs (Plus/Max/Flash, "
                      "~$0.19–1.30/1M in) — different models from this open-weight repo, so NOT "
                      "used as its price. Premium-only, like Llama 4"),
}


# Defined above PRICING but called here, once both are in scope.
_check_matrix_integrity()
