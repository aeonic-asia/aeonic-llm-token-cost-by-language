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
# floor — small on the whole-corpus aggregate (see the measured shares below), but
# material on short sentences (a true 2.0x premium reads ~1.6x when +6 lands
# on both sides of the ratio). The driver measures this floor per API counter and
# subtracts it from the per-sentence counts so the distribution is comparable to
# the offline counters. Precisely: count(probe) is MEASURED, and the subtrahend
# below is ASSUMED — a lone ASCII character is one token (byte-level BPE has no
# merge shorter than a character), which cannot be verified directly against a
# tokenizer whose vocabulary Anthropic does not publish. So the assumption is
# tested rather than trusted: measure.py probes several distinct single
# characters and requires them all to yield the same frame, and bounds the
# result. Empty content is rejected by the API, hence a single-char probe.
# Note what this does and does not establish: three probes AGREEING rules out a
# class-specific surprise, but cannot distinguish "each probe is one token" from
# "each is two". The subtrahend remains an assumption, tested for consistency.
#
# The aggregate is left uncorrected and stays the paper-style raw count. State the
# frame's share per corpus rather than as one bound — it is not uniformly <0.01%.
# Measured against each counter's English total: FLORES 0.0073% (newer) / 0.0124%
# (older); MASSIVE 0.0286% / 0.0463%, because MASSIVE's totals are ~1.5-2.1x10^4
# tokens, not ~10^5. Worst resulting bias on a published aggregate premium is
# +0.0007 (massive/claude-old/vie) — below the 4th decimal, so the numbers stand,
# but the aggregate is envelope-INCLUSIVE while the per-sentence distribution is
# envelope-stripped: near-identical bases, not identical ones. Do not describe them
# as "consistent bare-text bases".
ENVELOPE_PROBE = "x"
ENVELOPE_PROBE_TOKENS = 1
# Cross-checks for the assumption above, deliberately spanning character classes
# (lowercase / uppercase / digit / punctuation / non-ASCII). A single character
# is never *fewer* than one token in a byte-level BPE, so a probe that is worth
# two tokens can only push its apparent frame UP, never down. That asymmetry is
# what makes the floor (see measure.envelope_tokens) the right estimator and an
# outlier a finding rather than an abort.
#
# Both classes of surprise are represented on purpose, because both are real and
# were measured on 2026-07-26:
#   older tokenizer (claude-sonnet-4-6, claude-haiku-4-5): every DIGIT costs two
#     tokens; 'x' 'q' 'a' 'Z' '#' 'é' cost one. Frame 7.
#   newer tokenizer (claude-opus-5 and the counters folded into it): digits cost
#     one, but uppercase 'Z' costs two. Frame 6.
# An earlier three-probe set of ('x', 'q', '7') required unanimity, so the older
# tokenizer aborted on the digit — while the newer one passed only because the
# set happened to exclude 'Z'. Keeping a known outlier for each generation in the
# set means the outlier path stays exercised instead of latent.
ENVELOPE_PROBE_ALTS = ("q", "a", "Z", "#", "é", "7", "0")
# How many probes must agree on the floor before it is trusted as the frame. The
# floor being supported by a single probe would mean nearly every character is
# multi-token on that endpoint, which is not a tokenizer this method can measure.
ENVELOPE_MIN_AGREEING = 4
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
# probes. `count_tokens` is free, so this is an RPM/latency budget, not cost.
#
# The calls ARE retried. measure.py hands the SDK an explicit retry budget and
# per-request timeout (below); the SDK retries 408/409/429/5xx and connection
# errors with exponential backoff and obeys a `retry-after` header when the
# server sends one. What the driver does NOT do is throttle: it issues the
# ~12,000 calls back to back and relies on backoff to absorb a limit. If the
# budget is exhausted anyway, that counter's whole pass is rolled back and
# carry-forward restores its previous rows — but run.py then refuses to exit 0
# (see the exhausted-counter guard there), so a silently stale counter can no
# longer pass for a clean credentialed pass. Treat a full pass as something to
# watch, not to fire and forget.
#
# Note the slice is a deterministic HEAD slice of a corpus ordered by source
# document, so it is topically clustered — fine for a stable percentile, but not
# a random sample of the corpus. premium_by_language.csv records the sample size
# per row so a reader can tell these percentiles from the full sweeps.

# Retry budget and per-request timeout for the Anthropic client
# (measure.py:_ensure_client). The SDK defaults — max_retries=2 and a 10-minute
# timeout — are the wrong shape for this workload in both directions:
#
#  * Two retries buy ~1.5s of cumulative backoff. The SDK sleeps
#    min(0.5 * 2^n, 8) seconds, jittered, so the first two waits are ~0.5s and
#    ~1s. That absorbs a blip, not a sustained limit across ~12,000 sequential
#    calls. Eight retries give ~40s of cumulative backoff (0.5+1+2+4+8+8+8+8).
#  * A 10-minute timeout on a request whose body is one sentence only ever
#    describes a hung connection, and retries multiply it: worst case per call
#    is timeout x (max_retries + 1). At the defaults that is 30 minutes on a
#    single sentence; at 30s x 9 it is 4.5 minutes.
#
# Do not hand-roll a retry loop around these. The SDK's is correct, and it is
# the part that reads `retry-after`.
ANTHROPIC_MAX_RETRIES = 8
ANTHROPIC_TIMEOUT_S = 30.0


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
    # ── figure display (headline charts) ────────────────────────────────────
    # The full matrix carries proxy counters — the FOUR models that share one
    # newer-Claude tokenizer (Opus 5, Opus 4.8, Sonnet 5, Fable 5) and the two
    # that share the older one (Sonnet 4.6, Haiku 4.5) — so the eval can *verify*
    # equivalence in-dataset rather than asserting it. The headline figures
    # collapse those to one column per distinct tokenizer to stay readable. These
    # fields drive that collapse; nothing here affects measurement — only what the
    # charts show.
    headline: bool = True     # own column in the headline figures?
    headline_display: str = ""  # short chart/legend label (falls back to display)
    flagship_group: str = ""  # id of the headline column a folded counter maps to
    fold_reason: str = ""     # why folded — "shared" (byte-identical tokenizer,
    #                           re-verified against the counts before any caption
    #                           claims it) is the only supported value.


# The trimmed flagship matrix — 14 counters: one column per DISTINCT tokenizer
# (o200k, cl100k, newer Claude, older Claude, gemma4, Llama 4, Qwen) plus the
# fold proxies that prove the sharing in-dataset rather than asserting it. The
# original scope lock said "≈7 counters"; the proxies and the extra Anthropic
# price tiers took it to 11, then the two Gemini Flash price tiers to 13, then the
# retained Qwen 3.6 proxy to 14, each addition recorded in the workshop decision
# log. Note the count grows only with PRICE tiers and fold proofs — the number of
# distinct tokenizers is still seven, and that is the number the headline figures
# draw.
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
    #
    # ── 2026-07-27 model-currency check ──────────────────────────────────────
    # Google shipped four newer GA models since the last pass. None of them earns a
    # column, and the Pro column does not rotate. What changed is the PRICE layer:
    # this tokenizer now serves an 8x price range, so two of the four join as fold
    # proxies (below) to draw it.
    #
    #   Gemini 3.6 Flash       GA  $1.50  — NOT ADDED, see below
    #   Gemini 3.5 Flash       GA  $1.50  — added as a fold proxy
    #   Gemini 3.5 Flash-Lite  GA  $0.30  — NOT ADDED, see below
    #   Gemini 3.1 Flash-Lite  GA  $0.25  — added as a fold proxy
    #
    # Three findings, each load-bearing for a decision above:
    #
    #  1. No new tokenizer, so no new column. Every model the SDK maps from 3.1
    #     onward resolves to `gemma4`, and Google has published no tokenization
    #     change. The bar for a column is a vocabulary diff plus a vendor report
    #     (see the note above); neither exists here.
    #  2. The Pro column stays on 3.1 Pro because Google shipped no newer Pro. The
    #     Flash line advanced 3.1 -> 3.5 -> 3.6 while Pro did not move, so the
    #     "flagship rotated, repoint the spec" rule does not fire. gemini-3.1-pro-
    #     preview remains Google's newest Pro, and is still Preview rather than GA
    #     — note the two proxies below are GA, so this column is the only Preview-
    #     priced row in the matrix.
    #  3. Gemini 3.6 Flash and 3.5 Flash-Lite are UNMEASURABLE here, which is why
    #     the two newest models are absent while older ones are present. The
    #     counter is offline: `spec` is not an endpoint, it is a lookup key into
    #     the SDK's model->tokenizer table, and neither id is in that table — not
    #     in the pinned 2.12.1 and not on python-genai `main` (checked 2026-07-27;
    #     the two tables are identical, so upgrading the SDK would gain nothing).
    #     `get_tokenizer_name('gemini-3.6-flash')` raises ValueError. Assigning
    #     them gemma4 by hand would be asserting a tokenizer Google has not
    #     published and the SDK does not claim — exactly the inference this matrix
    #     refuses elsewhere. Revisit when the SDK maps them; the price rows are
    #     ready-made ($1.50 and $0.30).
    Counter("gemini-3-1-pro", "Gemini 3.1 Pro (gemma4)", "Google", "gemini_local",
            STATUS_NEEDS_SDK, spec="gemini-3.1-pro-preview",
            generation="gemini-gemma4",
            stands_in_for="Google Pro flagship — Gemini 3.1/3.5/4 'gemma4' tokenizer; "
                          "text vocabulary unchanged from the gemma3 line (2.0-3.0)",
            headline_display="Gemini 3.1 Pro"),
    # The two Google fold proxies. Same job as the Claude proxies: confirm the
    # shared tokenizer IN-DATASET via the coincidence check rather than on the
    # SDK's mapping table alone, and give the dollar figures the price span this
    # tokenizer actually serves. Before this pass Google drew a single $2.00 bar,
    # so the "same tokens, different price" argument rested on Claude alone; it now
    # has a second vendor, with a WIDER span (8x, $0.25 -> $2.00) than Claude's 10x
    # split across two tokenizers. Both are Flash-tier, both GA, both mapped to
    # gemma4 by the pinned SDK — measurable offline with no credential.
    Counter("gemini-3-5-flash", "Gemini 3.5 Flash (gemma4, shared)", "Google",
            "gemini_local", STATUS_NEEDS_SDK, spec="gemini-3.5-flash",
            generation="gemini-gemma4",
            stands_in_for="shared gemma4 tokenizer (verify); mid Flash tier",
            headline=False, headline_display="Gemini 3.5 Flash",
            flagship_group="gemini-3-1-pro", fold_reason="shared"),
    Counter("gemini-3-1-flash-lite", "Gemini 3.1 Flash-Lite (gemma4, shared)", "Google",
            "gemini_local", STATUS_NEEDS_SDK, spec="gemini-3.1-flash-lite",
            generation="gemini-gemma4",
            stands_in_for="shared gemma4 tokenizer (verify); cheapest mapped Gemini tier",
            headline=False, headline_display="Gemini 3.1 Flash-Lite",
            flagship_group="gemini-3-1-pro", fold_reason="shared"),
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
    # Qwen (Alibaba; open-weight, Apache-2.0). A large tokenizer — 248,044 vocab
    # entries, counted directly from the downloaded tokenizer.json (reproducible;
    # the only vocab figure here that is measured rather than cited). Earlier Qwen
    # generations are widely reported at ~152k, but that is a secondary-source
    # number and is not restated as fact. The size difference is not the finding
    # anyway — the measured premiums are. Earns its own headline column as the
    # first non-incumbent vendor in the matrix. Self-host: no single per-token
    # serving list price → premium-only, like Llama 4. These are the DENSE flagship
    # repos; the MoE sibling (Qwen3.8-2.4T-A95B) is *believed* to share this
    # tokenizer but that is NOT measured here — do not state it as fact, and add it
    # as a fold proxy if the claim ever needs to be made in print.
    #
    # ── 2026-08-16: repointed 3.6 → 3.8, and the tokenizer did NOT change ────────
    # Alibaba released Qwen3.8 in 2026-08. Per the flagship-rotation rule above the
    # spec moves to the current dense flagship and the outgoing model stays as a
    # fold proxy, so both names are backed by their own measured rows.
    #
    # The rotation is safe because the tokenizer is provably unchanged. A direct
    # diff of both `tokenizer.json` files (2026-08-16) found the BASE TEXT VOCABULARY
    # BYTE-IDENTICAL — 248,044 entries in each, zero pieces added, zero removed, zero
    # ids moved — AND the BPE merge lists identical as Python objects (247,587 rules
    # each). The only difference is seven added/special tokens present in 3.8 and not
    # 3.6, all audio/TTS control tokens (`<tts_pad>`, `<tts_text_bos>`,
    # `<tts_text_bos_single>`, `<tts_text_eod>`, `<|audio_start|>`, `<|audio_end|>`,
    # `<|audio_pad|>`), which is what a multimodal successor would add; every added
    # token the two share keeps its id.
    #
    # Note the strength of that evidence relative to everything else in this repo.
    # Identical vocab AND identical merges means the two tokenizers produce equal
    # counts BY CONSTRUCTION, on any corpus — not "equal on the corpus we happened to
    # run", which the Gemini episode showed proves nothing. The fold proxy below is
    # therefore confirmation, not the evidence; the vocabulary diff is the evidence.
    # This is the same shape as gemma3/gemma4: one text vocabulary, two artifacts,
    # differing only in control tokens. Two vendors independently.
    Counter("qwen-3-8", "Qwen 3.8 (open-weight)", "Alibaba", "hf",
            STATUS_NEEDS_SDK, spec="Qwen/Qwen3.8-27B",
            gated=False, headline_display="Qwen 3.8"),
    # Qwen 3.6 — the prior dense flagship, retained as a fold proxy after the column
    # moved to 3.8, exactly as claude-opus-4-8 was retained behind claude-new. Keeping
    # it measured is what lets a caption say "Qwen 3.8 = Qwen 3.6" with both names
    # backed by their own rows rather than by a comment.
    Counter("qwen-3-6", "Qwen 3.6 (open-weight, shared — confirmed)", "Alibaba", "hf",
            STATUS_NEEDS_SDK, spec="Qwen/Qwen3.6-27B",
            gated=False, headline=False, headline_display="Qwen 3.6",
            flagship_group="qwen-3-8", fold_reason="shared"),
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
    # analyze._within_vendor_inflation keys on these two literal ids to produce
    # within_vendor_inflation.csv — the article's headline hook. A rename used to
    # make it return empty: the CSV was unlinked, the summary key became {}, and
    # the console printed nothing at all. Fail at import instead.
    for required in ("claude-new", "claude-old"):
        if required not in MATRIX_BY_ID:
            raise ValueError(
                f"MODEL_MATRIX has no '{required}' counter — the within-vendor "
                "inflation metric (the article's headline hook) is keyed on "
                "'claude-new'/'claude-old' and would silently produce nothing. "
                "Keep the ids generational, or update analyze._within_vendor_inflation.")
    missing_oracle = sorted(set(PAPER_CL100K_FLORES) - set(LANGUAGES))
    if missing_oracle:
        raise ValueError(f"oracle languages absent from LANGUAGES: {missing_oracle} — "
                         "the cl100k validation gate would silently check fewer "
                         "languages and report a tighter delta")


# ── pricing (dated, sourced, confidence-flagged) ─────────────────────────────
# Cost-per-1k-chars = (tokens per 1k NFC chars, measured) × (USD per token).
# The token side is always real measurement; only the $ side lives here.
# UNVERIFIED prices are None → cost is omitted for that counter, never guessed.
# Layer-level stamp: the date the WHOLE pricing layer was last re-ratified against
# vendor pages. It is not a substitute for a row's own `as_of` and no PRICING entry
# reads it — every entry, priced or not, carries its own literal date, because every
# entry makes a claim about a specific SKU (or a specific reason not to price one).
# The two coincide today only because 2026-07-27 re-verified all eleven at once; a
# single-row correction moves that row's date and leaves this constant alone.
PRICING_AS_OF = "2026-08-16"
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


# Input prices per 1M input tokens (serving/inference list price — NOT the token
# *counting* endpoint, which is free). Every entry carries its own verification date
# and a confidence. Unpriced models stay None → cost is omitted (premium-only), never
# invented. Kept here as the single edit point for price ratification.
#
# ── 2026-07-27 re-ratification ───────────────────────────────────────────────
# All eleven entries re-checked against the vendor's current page. **No price moved**,
# so no cost figure and no published dollar number changes; only `as_of`, one
# `confidence`, and the source strings differ. Sources used, all primary:
#   * Anthropic — platform.claude.com models overview (Fable 5 $10/$50, Opus 5 and
#     Opus 4.8 $5/$25, Sonnet 5 $3/$15 with intro $2/$10 through 2026-08-31,
#     Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5). Cross-checked against the `claude-api`
#     skill's pricing table; both agree on all six.
#   * OpenAI — developers.openai.com/api/docs/pricing (gpt-5.6-luna $1.00 /
#     gpt-5.6-terra $2.50 / gpt-5.6-sol $5.00 input).
#   * Google — ai.google.dev/gemini-api/docs/pricing (Gemini 3.1 Pro Preview
#     $2.00 ≤200K / $4.00 >200K input; still preview, not GA).
#
# ── 2026-08-16 re-verification: the first pass where a price actually MOVED ──
# Prompted by a pre-merge freshness check — the ratification below was three weeks
# old and this branch's whole contribution is a price ladder. Ten priced/vendor rows
# re-checked; the three unpriced self-host/historical rows were NOT re-checked and
# keep their 2026-07-27 dates, because a row's date must mean "someone looked".
#
#  * **claude-sonnet-5 $3.00 -> $2.00.** Anthropic CANCELLED the scheduled increase
#    and made the launch price standard: "the previously scheduled increase to
#    $3/$15 per million input/output tokens on September 1, 2026 will not occur."
#    The elaborate list-vs-effective decision recorded below is therefore MOOT, not
#    overruled — there is no promotion left to decline to price. Every Sonnet 5
#    dollar figure drops by a third and the cost ladder REORDERS; this is the first
#    time this table has moved a published number.
#  * Everything else held: Opus 5 / Opus 4.8 $5, Sonnet 4.6 $3, Haiku 4.5 $1,
#    Fable 5 $10, GPT-5.6 Sol $5, Gemini 3.1 Pro $2.00 <=200K (still Preview),
#    3.5 Flash $1.50, 3.1 Flash-Lite $0.25 text.
#  * Two source strings were wrong without their prices being wrong, and are fixed
#    in place: OpenAI cut Luna to $0.20 and Terra to $2.00 (Sol unmoved, so this row
#    does not move), and Gemini 3.6 Flash is $0.75 through 2026-12-31 rather than the
#    $1.50 this table briefly claimed.
#  * Two new models seen and deliberately NOT added. **Claude Mythos 5** ($10/$50,
#    limited availability via Anthropic's Glasswing programme) is on the newer
#    tokenizer per Anthropic's own note, but it shares both that tokenizer AND
#    Fable 5's exact price, so by the claude-opus-4-8 rule it would draw a duplicate
#    bar; it also needs access this eval does not have. **Gemini 3.7 Flash**
#    (Preview, $0.75 -> $1.50) is unmappable like 3.6 Flash and 3.5 Flash-Lite.
#
# ── 2026-07-27 addendum: two Google rows added ───────────────────────────────
# The re-ratification above covered the eleven entries that existed at the time and
# moved no price. Separately and on the same date, the model-currency check (see the
# Gemini block in MODEL_MATRIX) added two Flash tiers, so this table now holds
# thirteen. Both new prices are first verifications, not re-verifications, read off
# the same ai.google.dev pricing page in the same pass. No existing price moved as a
# result — the two rows are additive, and every previously published dollar figure
# is unchanged.
#
# ── Two editorial choices, both DECIDED here rather than left implicit ────────
#  * ~~claude-sonnet-5 → list $3.00, not the effective $2.00 intro.~~ **SUPERSEDED
#    2026-08-16 — kept only because it explains why the row read $3.00 for three
#    weeks.** The reasoning was: price every row undiscounted, so a promotion does
#    not put a temporarily-cheap bar in a list-price ladder and flatter Sonnet 5
#    against Sonnet 4.6 at the same $3.00 list. Anthropic then cancelled the
#    scheduled reversion and made $2.00 the standard price, which dissolves the
#    choice rather than deciding it the other way: there is no discount to decline.
#    The row is $2.00 because that is list. Do not resurrect the caveat.
#  * o200k_base → **the Sol tier**, now claimed by exact SKU id (`gpt-5.6-sol`), so
#    `confidence` is `high`: the number is primary-sourced for a named SKU. What is
#    editorial is the *tier choice*, not the number, and a confidence grade is the
#    wrong field to record a scope decision in — it travels to
#    `cost_by_language.price_confidence`, where "medium" reads as "we are unsure of
#    $5.00" rather than "we picked the flagship of three". The tier caveat now lives
#    in the source string, which is where a drafter reads it.
#    ⚠️ Comparability caveat worth carrying into the article: one tokenizer serving
#    several price points is visible in the dollar figures for Claude (Haiku /
#    Sonnet / Opus / Fable each draw their own bar) but NOT for OpenAI, whose Luna
#    and Terra tiers share o200k at $1.00 and $2.50 and appear nowhere. OpenAI's bar
#    is therefore the TOP of its tokenizer's price range while Claude's span theirs.
#    Do not read the charts as "OpenAI costs $5.00 to serve"; read them as "the
#    flagship tier does".
PRICING: dict[str, Price] = {
    "o200k_base": Price(5.00, "2026-08-16", "high",
                        "GPT-5.6 Sol (SKU `gpt-5.6-sol`) input list price — the flagship "
                        "of three GPT-5.6 tiers that all share the o200k tokenizer "
                        "(Luna $0.20 / Terra $2.00 / Sol $5.00 in; the cheaper two were "
                        "cut from $1.00 and $2.50 between 2026-07-27 and 2026-08-16, "
                        "which does NOT move this row but widens the unseen range below "
                        "it to 25x). Verified 2026-08-16 "
                        "vs. developers.openai.com/api/docs/pricing. `high` because the "
                        "number is primary-sourced for a NAMED SKU; the tier pick is an "
                        "editorial scope choice, not an uncertainty. Note the cheaper "
                        "tiers tokenize identically and draw no bar, so this is the TOP "
                        "of o200k's price range, not OpenAI's only price"),
    "cl100k_base": Price(None, "2026-07-27", "unknown",
                         "historical baseline tokenizer (2023 GPT-4 / ChatGPT era); OpenAI "
                         "sells no cl100k-served SKU, so there is no list price to record "
                         "rather than one we declined to look up. Re-confirmed 2026-07-27"),
    # Each Claude row is dated by when its own SKU claim was last checked, never by
    # PRICING_AS_OF: an entry naming Opus 5 stamped with a pre-Opus-5 date would assert
    # a false provenance — the same error DATASET_AS_OF exists to prevent.
    "claude-new": Price(5.00, "2026-08-16", "high",
                        "Newer-Claude tokenizer, priced on Claude Opus 5 (SKU "
                        "`claude-opus-5`): $5.00/1M in, $25.00/1M out. Opus 5 is "
                        "Anthropic's recommended default — Fable 5 is its most capable "
                        "widely released model, so avoid 'flagship' here. Opus 4.8 shares "
                        "both this tokenizer and this exact price (see its entry). "
                        "Verified 2026-07-27 vs. the Anthropic models/pricing page"),
    "claude-old": Price(3.00, "2026-08-16", "high",
                        "Older-Claude tokenizer, priced on Claude Sonnet 4.6 (SKU "
                        "`claude-sonnet-4-6`): $3.00/1M in, $15.00/1M out — an "
                        "undiscounted list price with no promotion attached. Verified "
                        "2026-07-27 vs. the Anthropic models/pricing page"),
    "claude-sonnet-5": Price(2.00, "2026-08-16", "high",
                             "Claude Sonnet 5 (SKU `claude-sonnet-5`) standard list price: "
                             "$2.00/1M in, $10.00/1M out. THE ONLY PRICE THAT HAS MOVED IN "
                             "THIS TABLE — and it moved by being made permanent, not by "
                             "changing. Through 2026-07-27 this row carried $3.00 with a "
                             "deliberate list-vs-effective note: $2.00 was introductory "
                             "pricing due to revert to $3.00 on 2026-09-01, and list won on "
                             "comparability. Anthropic has since cancelled that increase — "
                             "its pricing page now states the $2/$10 launch pricing 'is now "
                             "the standard price' and 'the previously scheduled increase to "
                             "$3/$15 per million input/output tokens on September 1, 2026 "
                             "will not occur.' So $2.00 is simply the list price, the old "
                             "reasoning is moot rather than overridden, and NO intro-price "
                             "caveat belongs in print. Consequence worth carrying: Sonnet 5 "
                             "is now CHEAPER than Sonnet 4.6 ($3.00), so the newer Claude "
                             "tokenizer is no longer uniformly dearer at the Sonnet tier — "
                             "more tokens, lower unit price. Verified 2026-08-16 vs. the "
                             "Anthropic models/pricing page"),
    "claude-fable-5": Price(10.00, "2026-08-16", "high",
                            "Claude Fable 5 (SKU `claude-fable-5`) input list price "
                            "($10.00/1M in, $50.00/1M out) — 2x Opus 5 / Opus 4.8; shares "
                            "the newer-Claude tokenizer, so identical token counts at "
                            "double the price. Verified 2026-07-27 vs. the Anthropic "
                            "models/pricing page + claude-api reference"),
    "claude-haiku-4-5": Price(1.00, "2026-08-16", "high",
                              "Claude Haiku 4.5 (SKU `claude-haiku-4-5`) input list price "
                              "($1.00/1M in, $5.00/1M out) — the cheapest Claude serving "
                              "tier, and it runs the OLDER Claude tokenizer. Verified "
                              "2026-07-27 vs. the Anthropic models/pricing page + "
                              "claude-api reference"),
    # Opus 4.8 is deliberately UNPRICED here — not because the price is unknown
    # ($5.00/1M in, $25.00/1M out, re-verified 2026-07-27 vs. the Anthropic
    # models/pricing page) but because it is *identical* to the claude-new column this
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
    "claude-opus-4-8": Price(None, "2026-08-16", "unknown",
                             "premium-only by design: same tokenizer AND same $5.00/1M input price as "
                             "the claude-new headline column it folds into (Claude Opus 5) — priced "
                             "there, not duplicated here. Both re-verified 2026-07-27 vs. the "
                             "Anthropic models/pricing page and still identical, so the rationale "
                             "holds; were they ever to diverge, this row must be priced"),
    # The former "gemini-3-pro" entry is gone with its counter: gemini-3-pro-preview
    # was shut down 2026-03-09 and is absent from Google's pricing page, so its
    # $2.00 was a price for a model no longer sold.
    "gemini-3-1-pro": Price(2.00, "2026-08-16", "high",
                            "Gemini 3.1 Pro (SKU `gemini-3.1-pro-preview`) — Google's "
                            "current and newest Pro model — input list price, "
                            "<=200K-context tier ($2.00/1M in; $4.00/1M above 200K). The "
                            "<=200K tier is the CORRECT one here, not merely the cheaper "
                            "one: the priced unit is 1k characters / one sentence, so a "
                            "request under this eval never approaches the 200K boundary. "
                            "Verified 2026-07-27 against ai.google.dev/gemini-api/docs/"
                            "pricing, which still labels it Preview and not GA — a preview "
                            "price carries less notice before it changes than a stable one. "
                            "It is the DEAREST of three gemma4 tiers priced here (Flash-Lite "
                            "$0.25 / Flash $1.50 / Pro $2.00), so unlike o200k this "
                            "tokenizer's full price range is drawn"),
    # The two Flash tiers. Their reason to exist is the DOLLAR figures: identical
    # tokens to the Pro column at a fraction of the price, which is the article's
    # thesis stated by a second vendor. Priced (unlike claude-opus-4-8, which is
    # left unpriced precisely because its price matches its column's) — here every
    # price differs, so each draws its own bar.
    "gemini-3-5-flash": Price(1.50, "2026-08-16", "high",
                              "Gemini 3.5 Flash (SKU `gemini-3.5-flash`) input list price "
                              "($1.50/1M in), GA — not Preview, unlike the Pro column it "
                              "folds into. Verified 2026-08-16 vs. ai.google.dev/gemini-api/"
                              "docs/pricing; unmoved since 2026-07-27. ⚠️ A CAPTION CLAIM "
                              "WAS RETIRED HERE: this row briefly said the newer Gemini 3.6 "
                              "Flash carried the same $1.50, so the bar was current for "
                              "both. That is no longer true — 3.6 Flash is $0.75 through "
                              "2026-12-31 and $1.50 only from 2027-01-01, so the bar is HALF "
                              "its height for most of the article's life. Name 3.5 Flash and "
                              "nothing else; 3.6 Flash remains unmeasurable anyway (the SDK "
                              "maps no tokenizer for it)"),
    "gemini-3-1-flash-lite": Price(0.25, "2026-08-16", "high",
                                   "Gemini 3.1 Flash-Lite (SKU `gemini-3.1-flash-lite`) "
                                   "input list price for TEXT ($0.25/1M in; audio input is "
                                   "a separate $0.50 tier, not used here — this eval feeds "
                                   "text only, so the text tier is the correct one rather "
                                   "than merely the cheaper one). GA. The cheapest Gemini "
                                   "tier the SDK can tokenize, hence the light end of the "
                                   "green ramp; note 3.5 Flash-Lite at $0.30 is newer and "
                                   "also GA but is unmappable, so this is not Google's "
                                   "cheapest model, only the cheapest measurable one. "
                                   "Verified 2026-07-27 vs. ai.google.dev/gemini-api/docs/"
                                   "pricing"),
    "llama-4": Price(None, "2026-07-27", "unknown",
                     "self-host / open-weight; no single per-token list price. Third-party "
                     "hosts serve Llama 4 at differing rates, none of which is Meta's price "
                     "for this open-weight repo, so none is used. Re-confirmed 2026-07-27"),
    "qwen-3-8": Price(None, "2026-08-16", "unknown",
                      "self-host / open-weight (Apache-2.0); no single per-token list price. "
                      "Alibaba Cloud hosts distinct commercial Qwen SKUs (Plus/Max/Flash) — "
                      "different models from this open-weight repo, so NOT used as its price. "
                      "Premium-only, like Llama 4"),
    "qwen-3-6": Price(None, "2026-07-27", "unknown",
                      "premium-only by design, same as the qwen-3-8 column it folds into: "
                      "self-host / open-weight, no single per-token list price. Kept as the "
                      "measured proxy for the prior dense flagship. Date deliberately NOT "
                      "bumped on 2026-08-16 — nothing about this row was re-checked, and "
                      "there is no price to re-check"),
}


# Defined above PRICING but called here, once both are in scope.
_check_matrix_integrity()
