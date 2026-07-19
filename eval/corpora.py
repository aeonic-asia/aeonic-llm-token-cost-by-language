"""Corpus loading — dual register. Both corpora are committed and parallel.

FLORES+ : the same ~2009 sentences (dev + devtest) human-translated into every
          language (formal/encyclopedic prose). Committed under flores200_dataset/.
MASSIVE : 2033 short virtual-assistant utterances (the conversational register
          SME chatbots serve), human-translated across locales. Committed under
          eval/massive/ (built once by build_massive.py; see its PROVENANCE.md).

Both are one line per sentence/utterance, aligned by line index across
languages — `load_x(a)[i]` and `load_x(b)[i]` are translations of each other.
That alignment is what lets us compute a *per-sentence* premium distribution,
not just an aggregate ratio, and to report the premium **per corpus**.
"""
from __future__ import annotations

from typing import Callable

from . import config


def load_flores(lang: str) -> list[str]:
    """Return the FLORES+ sentences for `lang` (dev + devtest), NFC-normalized."""
    from .measure import nfc

    sentences: list[str] = []
    for split, ext in (("dev", "dev"), ("devtest", "devtest")):
        path = config.FLORES_DIR / split / f"{lang}.{ext}"
        text = path.read_text(encoding="utf-8")
        sentences.extend(nfc(line) for line in text.split("\n") if line.strip())
    return sentences


def load_massive(lang: str) -> list[str]:
    """Return the committed MASSIVE utterances for `lang`, NFC-normalized."""
    from .measure import nfc

    text = (config.MASSIVE_DIR / f"{lang}.txt").read_text(encoding="utf-8")
    return [nfc(line) for line in text.split("\n") if line.strip()]


_LOADERS: dict[str, Callable[[str], list[str]]] = {
    "flores": load_flores,
    "massive": load_massive,
}


def load_corpus(corpus: str, langs: list[str]) -> dict[str, list[str]]:
    """Load one corpus for several languages; assert line-alignment (equal length)."""
    loader = _LOADERS[corpus]
    data = {lang: loader(lang) for lang in langs}
    lengths = {lang: len(s) for lang, s in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{corpus} languages are not line-aligned: {lengths}")
    return data


def load_parallel(langs: list[str]) -> dict[str, list[str]]:
    """Backward-compatible alias: the FLORES+ corpus."""
    return load_corpus("flores", langs)


def corpus_size(corpus: str) -> int:
    """Number of aligned sentences/utterances in `corpus`.

    The corpora are parallel (one line per sentence, aligned across languages),
    so the count is language-independent — measured on the baseline language.
    This is corpus metadata and the honest denominator for a *per-sentence* cost,
    distinct from the per-character one: a dense script says the same thing in far
    fewer characters, so the two denominators rank the languages differently.
    """
    return len(_LOADERS[corpus](config.BASELINE_LANG))
