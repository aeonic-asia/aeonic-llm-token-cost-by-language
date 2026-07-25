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


def _lines(text: str, src: str) -> list[str]:
    """Split into sentence lines — CRLF-tolerant, positional, blank-intolerant.

    Alignment across languages is by line index, so a blank line silently
    *filtered* in one language would shift every later index and mis-pair every
    subsequent sentence — and the equal-count guard in load_corpus can't catch a
    same-count shift. So we do NOT filter: strip a single trailing newline's empty
    tail (a real file ends in "\\n"), tolerate CRLF checkouts, and treat any
    remaining blank line as corruption (a hard error naming the offending index).

    Files are read as utf-8-sig, the companion to the CRLF tolerance above: an
    editor re-save on Windows can prepend a BOM, and U+FEFF is not whitespace to
    Python — so under plain utf-8 it survives the blank-line guard and is counted
    and tokenized as the first character of sentence 0.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()                              # the file's terminal newline
    out = [ln.rstrip("\r") for ln in lines]      # tolerate CRLF (core.autocrlf)
    blanks = [i for i, ln in enumerate(out) if not ln.strip()]
    if blanks:
        raise ValueError(f"{src}: blank line(s) at index {blanks} would break "
                         "cross-language line alignment — one non-empty line per "
                         "sentence is required")
    return out


def load_flores(lang: str) -> list[str]:
    """Return the FLORES+ sentences for `lang` (dev + devtest), NFC-normalized."""
    from .measure import nfc

    sentences: list[str] = []
    for split, ext in (("dev", "dev"), ("devtest", "devtest")):
        path = config.FLORES_DIR / split / f"{lang}.{ext}"
        sentences.extend(nfc(ln) for ln in _lines(path.read_text(encoding="utf-8-sig"), str(path)))
    return sentences


def load_massive(lang: str) -> list[str]:
    """Return the committed MASSIVE utterances for `lang`, NFC-normalized."""
    from .measure import nfc

    path = config.MASSIVE_DIR / f"{lang}.txt"
    return [nfc(ln) for ln in _lines(path.read_text(encoding="utf-8-sig"), str(path))]


_LOADERS: dict[str, Callable[[str], list[str]]] = {
    "flores": load_flores,
    "massive": load_massive,
}


def _loader(corpus: str) -> Callable[[str], list[str]]:
    """Resolve a corpus id to its loader, naming the registry on a miss.

    `config.CORPORA` and `_LOADERS` are two independent registries. A corpus in
    one but not the other used to die with a bare KeyError from deep inside the
    load — and a corpus *removed* from config while its rows remain in the
    committed CSVs hits the same path via corpus_size() during analyze.
    """
    try:
        return _LOADERS[corpus]
    except KeyError:
        raise KeyError(
            f"no loader registered for corpus '{corpus}' (known: "
            f"{sorted(_LOADERS)}). If it was retired, its rows may still be in "
            "eval/results/ — re-run `make reproduce` to purge them; if it is new, "
            "register a loader here and a path in config.") from None


def load_corpus(corpus: str, langs: list[str]) -> dict[str, list[str]]:
    """Load one corpus for several languages; assert line-alignment (equal length)."""
    loader = _loader(corpus)
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
    return len(_loader(corpus)(config.BASELINE_LANG))
