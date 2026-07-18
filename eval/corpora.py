"""Corpus loading. FLORES+ now; MASSIVE deferred (second-domain robustness).

FLORES+ is the committed parallel corpus: the same ~2009 sentences (dev +
devtest) human-translated into every language, one sentence per line, aligned
by line index across languages. That alignment is what lets us compute a
*per-sentence* premium distribution, not just an aggregate ratio.
"""
from __future__ import annotations

from . import config


def load_flores(lang: str) -> list[str]:
    """Return the FLORES+ sentences for `lang` (dev + devtest), NFC-normalized.

    Order is preserved and identical across languages, so `load_flores(a)[i]`
    and `load_flores(b)[i]` are translations of each other. Blank trailing
    lines from the file split are dropped.
    """
    from .measure import nfc

    sentences: list[str] = []
    for split, ext in (("dev", "dev"), ("devtest", "devtest")):
        path = config.FLORES_DIR / split / f"{lang}.{ext}"
        text = path.read_text(encoding="utf-8")
        sentences.extend(nfc(line) for line in text.split("\n") if line.strip())
    return sentences


def load_parallel(langs: list[str]) -> dict[str, list[str]]:
    """Load several languages and assert they are line-aligned (equal length)."""
    data = {lang: load_flores(lang) for lang in langs}
    lengths = {lang: len(s) for lang, s in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"FLORES languages are not line-aligned: {lengths}")
    return data
