"""Measurement primitives: NFC gate, character counting, and token counters.

Design note — why a new contract instead of `TokenizerInterface`.
The upstream `TokenizerInterface` (compute/tokenizer_interface.py) requires
`encode`/`decode`/`align_tokens_to_text` — it assumes you get token *strings*
back. Closed models expose only a `count_tokens` endpoint: a count, no strings.
So this eval hangs off a deliberately narrower contract — `TokenCounter`, which
promises only `count(text) -> int`. Offline tokenizers satisfy it trivially
(`len(encode(text))`); the Anthropic API satisfies it directly. One contract,
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

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of input tokens for `text` (after NFC)."""
        raise NotImplementedError

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


class AnthropicCounter(TokenCounter):
    """Claude token counts via the Anthropic `count_tokens` API.

    Built and plumbed now; it only runs when an API key is present. This is the
    article's headline hook (the newer Claude tokenizer counts materially more),
    so it must be measured, never estimated — hence the hard failure below
    instead of a silent fallback.
    """

    def __init__(self, spec: config.Counter):
        self.spec = spec
        self._client = None  # lazily created so import/availability checks are cheap

    @staticmethod
    def available() -> bool:
        import os
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # not installed in the offline slice; installed with the key
            self._client = anthropic.Anthropic()
        return self._client

    def count(self, text: str) -> int:
        client = self._ensure_client()
        resp = client.messages.count_tokens(
            model=self.spec.spec,
            messages=[{"role": "user", "content": nfc(text)}],
        )
        return resp.input_tokens


def build_counter(spec: config.Counter) -> tuple[TokenCounter | None, str]:
    """Instantiate a counter if runnable, else return (None, reason-skipped).

    This is the single gate that keeps the eval honest: a counter runs only when
    its prerequisites are actually present. Nothing is faked in its place.
    """
    if spec.kind == "tiktoken":
        return TiktokenCounter(spec), ""
    if spec.kind == "anthropic":
        if not AnthropicCounter.available():
            return None, "ANTHROPIC_API_KEY unset — run later with a key"
        return AnthropicCounter(spec), ""
    if spec.kind in ("gemini_local", "hf"):
        return None, f"deferred ({spec.kind}); verify model id + install SDK/token"
    return None, f"unknown counter kind: {spec.kind}"
