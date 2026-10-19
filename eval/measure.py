"""Measurement primitives: NFC gate, character counting, and token counters.

Design note — why a new contract instead of `TokenizerInterface`.
The upstream `TokenizerInterface` (compute/tokenizer_interface.py) requires
`encode`/`decode`/`align_tokens_to_text` — it assumes you get token *strings*
back. Closed models expose only a `count_tokens` endpoint: a count, no strings.
So this eval hangs off a deliberately narrower contract — `TokenCounter`, which
promises only `count(text) -> int`. Offline tokenizers satisfy it trivially
(`len(encode(text))`); the vendor count-tokens APIs satisfy it directly. One contract,
both worlds, no fake `encode` on the closed models.
"""
from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod

from . import config


# ── NFC gate + character counting ────────────────────────────────────────────
def nfc(text: str) -> str:
    """Normalize to NFC. Applied before *every* token count and char count.

    This is the Vietnamese trap-closer: `ề` can arrive precomposed (U+1EC1, one
    code point) or decomposed (base + two combining marks, three code points).
    Counting tokens or characters on unnormalized text would over- or
    under-count depending on the source's normalization. NFC fixes the origin
    so counts are comparable across languages and tokenizers.
    """
    return unicodedata.normalize("NFC", text)


def char_count(text: str) -> int:
    """Character count = Unicode code points after NFC normalization.

    Disclosed definition (per the eval scope): a "character" is one NFC code
    point, not a grapheme cluster. For the five languages here the two mostly
    coincide once NFC-composed; grapheme segmentation (which would merge e.g.
    emoji ZWJ sequences) is deliberately not used to keep the dependency
    surface minimal and the definition auditable.
    """
    return len(nfc(text))


# ── counter contract ─────────────────────────────────────────────────────────
class TokenCounter(ABC):
    """Count input tokens for a piece of text. NFC is applied by the counter."""

    #: config.Counter this instance was built from (metadata, status, spec).
    spec: config.Counter

    #: Per-probe frame measured by envelope_tokens(), for counters that probe.
    #: Empty for bare-text counters, which have no frame to measure. Always
    #: REPLACED with a fresh dict, never mutated in place — this class-level
    #: default is shared by every instance that does not probe.
    envelope_probe_frames: dict[str, int] = {}

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of input tokens for `text` (after NFC)."""
        raise NotImplementedError

    def envelope_tokens(self) -> int:
        """Fixed per-message token overhead this counter adds beyond bare text.

        Zero for bare-text (offline) counters, which encode the raw string.
        API counters that count a wrapped chat message (Anthropic `count_tokens`)
        override this to expose the turn/role frame so the driver can subtract it
        from per-sentence counts — making them comparable to the offline
        counters. See config.ENVELOPE_PROBE.
        """
        return 0

    @property
    def display(self) -> str:
        return self.spec.display


class TiktokenCounter(TokenCounter):
    """Offline OpenAI tokenizer via tiktoken (cl100k_base / o200k_base).

    Fully offline once the BPE ranks are cached under TIKTOKEN_CACHE_DIR
    (committed). Byte-level BPE, so there is no UNK to gate on.
    """

    def __init__(self, spec: config.Counter):
        import tiktoken  # local import: keeps deferred counters import-free
        self.spec = spec
        self._enc = tiktoken.get_encoding(spec.spec)

    def count(self, text: str) -> int:
        return len(self._enc.encode(nfc(text)))


class ApiCounter(TokenCounter):
    """Base for vendor count-tokens endpoints: a count, never token strings.

    Subclasses name the credential (`KEY_ENVS`) and implement `_make_client` and
    `_count`. These are the article's headline numbers, so they are measured,
    never estimated: a missing key skips the counter, and a failing call raises.
    """

    #: Environment variables any one of which authorizes the client.
    KEY_ENVS: tuple[str, ...] = ()

    def __init__(self, spec: config.Counter):
        self.spec = spec
        self._client = None  # lazy, so availability checks stay cheap

    @classmethod
    def available(cls) -> bool:
        import os
        return any(os.environ.get(k) for k in cls.KEY_ENVS)

    def _make_client(self):
        raise NotImplementedError

    def _count(self, client, text: str) -> int:
        raise NotImplementedError

    def count(self, text: str) -> int:
        if self._client is None:
            self._client = self._make_client()
        return self._count(self._client, nfc(text))

    def envelope_tokens(self) -> int:
        """Turn/role frame `count_tokens` wraps around the content.

        A count-tokens endpoint counts the fully-rendered request, so every call
        can carry a fixed frame on top of the content tokens. `count(probe)` is measured; the
        frame is then `count(probe) - ENVELOPE_PROBE_TOKENS`.

        Be precise about what is measured and what is assumed. The subtrahend
        rests on one assumption — that a lone character is exactly one token —
        which cannot be verified directly against a closed tokenizer that
        publishes no vocabulary. It is not arbitrary (byte-level BPE has no merge
        shorter than one character), but it is an assumption, so it is *tested*
        rather than trusted, across probes spanning several character classes.

        The test is a floor, not unanimity, and the asymmetry is what justifies
        that. A single character can never be worth *fewer* than one token, so a
        probe worth two tokens reports a frame one too HIGH and can never report
        one too low. The minimum across a diverse probe set is therefore the best
        estimator of the frame, and a probe above it is a multi-token character —
        a fact about that character, not evidence the frame is wrong. Requiring
        unanimity instead treats those two cases identically and aborts on the
        first character class the endpoint happens to tokenize differently, which
        is exactly what a digit does on the older Claude tokenizer and uppercase
        'Z' does on the newer one.

        What still aborts is genuine ambiguity: too few probes agreeing on the
        floor means most characters are multi-token on this endpoint, and there is
        no longer a defensible one-token anchor to subtract from.

        Outlier probes are recorded on the instance (`envelope_probe_frames`) so
        the driver can commit them to the manifest — they are evidence about the
        tokenizer, and discarding them would repeat the mistake of measuring
        something and not saying so.

        Deterministic: fixed probes, so re-runs are byte-identical.
        """
        probes = (config.ENVELOPE_PROBE, *config.ENVELOPE_PROBE_ALTS)
        frames = {p: self.count(p) - config.ENVELOPE_PROBE_TOKENS for p in probes}
        self.envelope_probe_frames = dict(frames)
        floor = min(frames.values())
        agreeing = sorted(p for p, f in frames.items() if f == floor)
        if len(agreeing) < config.ENVELOPE_MIN_AGREEING:
            raise RuntimeError(
                f"{self.spec.id}: only {len(agreeing)} of {len(frames)} envelope "
                f"probes agree on the floor ({frames}), below the required "
                f"{config.ENVELOPE_MIN_AGREEING}. That means most single "
                "characters are multi-token on this endpoint, so there is no "
                "one-token anchor to subtract and per-sentence counts cannot be "
                "corrected safely. Investigate before publishing.")
        envelope = floor
        if not 0 <= envelope <= config.ENVELOPE_MAX_PLAUSIBLE:
            raise RuntimeError(
                f"{self.spec.id}: measured envelope {envelope} is outside the "
                f"plausible range 0..{config.ENVELOPE_MAX_PLAUSIBLE}. A wrong "
                "frame shifts the entire per-sentence premium distribution, so "
                "this aborts rather than recording it as 'measured'.")
        return envelope


class AnthropicCounter(ApiCounter):
    """Claude via the Anthropic `messages.count_tokens` API."""

    KEY_ENVS = ("ANTHROPIC_API_KEY",)

    def _make_client(self):
        import anthropic
        return anthropic.Anthropic(max_retries=config.API_MAX_RETRIES,
                                   timeout=config.API_TIMEOUT_S)

    def _count(self, client, text: str) -> int:
        return client.messages.count_tokens(
            model=self.spec.spec, messages=[{"role": "user", "content": text}],
        ).input_tokens


class OpenAICounter(ApiCounter):
    """OpenAI via the Responses API input-token count (`responses.input_tokens`)."""

    KEY_ENVS = ("OPENAI_API_KEY",)

    def _make_client(self):
        import openai
        return openai.OpenAI(max_retries=config.API_MAX_RETRIES,
                             timeout=config.API_TIMEOUT_S)

    def _count(self, client, text: str) -> int:
        return client.responses.input_tokens.count(
            model=self.spec.spec, input=text).input_tokens


class GeminiApiCounter(ApiCounter):
    """Gemini via the Gemini API `models.count_tokens` endpoint."""

    KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def _make_client(self):
        from google import genai
        from google.genai import types
        return genai.Client(http_options=types.HttpOptions(
            timeout=int(config.API_TIMEOUT_S * 1000),  # milliseconds
            retry_options=types.HttpRetryOptions(attempts=config.API_MAX_RETRIES + 1)))

    def _count(self, client, text: str) -> int:
        return client.models.count_tokens(model=self.spec.spec,
                                          contents=text).total_tokens


class HFCounter(TokenCounter):
    """Open-weight tokenizer via HuggingFace `AutoTokenizer` (e.g. Qwen).

    Offline once the tokenizer files are cached, but the first load downloads
    them. Some repos are *gated* (Meta Llama) and need an HF access token (read
    from the standard `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env vars by
    `from_pretrained`); others are ungated (Apache-2.0 Qwen) and download with no
    credentials. The counter's `config.Counter.gated` flag says which, so an
    ungated repo isn't wrongly skipped for a missing token. Counts *content*
    tokens only (`add_special_tokens=False`), matching the tiktoken counters —
    per-sequence BOS/EOS would inflate short-sentence premiums.
    """

    def __init__(self, spec: config.Counter):
        self.spec = spec
        self._tok = None  # lazy: defer the (network) load until first count

    @staticmethod
    def available(gated: bool) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "transformers not installed (pip install transformers sentencepiece)"
        import os
        if gated and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            return False, "HF token unset (gated repo) — set HF_TOKEN and re-run"
        return True, ""

    def _ensure_tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            try:
                self._tok = AutoTokenizer.from_pretrained(self.spec.spec)
            except Exception as exc:  # concise, truthful reason for the manifest
                msg = str(exc)
                # 401 included: an ungated repo that HuggingFace later gates
                # answers 401, not 403, and would otherwise land a raw traceback
                # in the manifest instead of the actionable instruction.
                if any(k in msg for k in ("gated repo", "restricted", "403", "401")):
                    raise RuntimeError(
                        f"gated repo {self.spec.spec}: token lacks access — "
                        "accept the model's license on its HF page, then re-run"
                    ) from None
                raise
        return self._tok

    def count(self, text: str) -> int:
        return len(self._ensure_tok().encode(nfc(text), add_special_tokens=False))


class GeminiLocalCounter(TokenCounter):
    """Gemini token counts via the google-genai offline `LocalTokenizer`.

    No API key: the SDK downloads the Gemma tokenizer once (~30MB), then counts
    fully offline. `count_tokens(text).total_tokens` is the count; only text is
    measured (the SDK ignores non-text parts anyway).

    Which tokenizer a model resolves to is the SDK's own lookup table, and it
    splits by minor version, not by family: Gemini 2.0/2.5/3.0 map to "gemma3"
    (a hash-pinned SentencePiece download), while 3.1/3.5/4 map to "gemma4" (a
    HuggingFace tokenizer). For counting natural-language text the distinction is
    immaterial — the two ship the same 262,144-entry text vocabulary with
    identical token ids, differing only in chat-control tokens, and Google's
    Gemma 3 and Gemma 4 technical reports both state the tokenizer is unchanged
    from Gemini 2.0. Note also that the map is a client-side table that drifts
    from the live API: it still lists model ids Google has since retired.
    """

    def __init__(self, spec: config.Counter):
        self.spec = spec
        self._tok = None  # lazy: first construction triggers the one-time download

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            from google.genai.local_tokenizer import LocalTokenizer  # noqa: F401
        except ImportError:
            return False, "google-genai[local-tokenizer] not installed"
        return True, ""

    def _ensure_tok(self):
        if self._tok is None:
            from google.genai.local_tokenizer import LocalTokenizer
            self._tok = LocalTokenizer(self.spec.spec)
        return self._tok

    def count(self, text: str) -> int:
        return self._ensure_tok().count_tokens(nfc(text)).total_tokens


def build_counter(spec: config.Counter) -> tuple[TokenCounter | None, str]:
    """Instantiate a counter if runnable, else return (None, reason-skipped).

    This is the single gate that keeps the eval honest: a counter runs only when
    its prerequisites are actually present. Nothing is faked in its place.
    """
    if spec.kind == "tiktoken":
        # Guarded like every other kind. Construction resolves the encoding and
        # can raise (typo'd spec, an encoding absent from the committed cache, a
        # corrupt blob, a cache miss with no network) — and this call sits outside
        # the driver's per-counter try, so an unguarded raise killed the entire
        # pass: no manifest, no CSVs, no carry-forward.
        try:
            return TiktokenCounter(spec), ""
        except Exception as exc:  # noqa: BLE001 — one bad counter must not kill the run
            return None, (f"tiktoken encoding '{spec.spec}' unavailable "
                          f"({type(exc).__name__}) — check the name and that it is "
                          "present in eval/tiktoken_cache/")
    api = {"anthropic": AnthropicCounter, "openai": OpenAICounter,
           "gemini_api": GeminiApiCounter}.get(spec.kind)
    if api:
        if not api.available():
            return None, f"{' / '.join(api.KEY_ENVS)} unset — run later with a key"
        return api(spec), ""
    if spec.kind == "hf":
        ok, reason = HFCounter.available(spec.gated)
        return (HFCounter(spec), "") if ok else (None, reason)
    if spec.kind == "gemini_local":
        ok, reason = GeminiLocalCounter.available()
        return (GeminiLocalCounter(spec), "") if ok else (None, reason)
    return None, f"unknown counter kind: {spec.kind}"
