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

Known corpus artifact, preserved deliberately: FLORES `vie_Latn` index 429
contains literal HTML (`km<sup>2</sup>`) — the ONLY markup in any of the ten
corpus files (6 tags in ~1.52M NFC characters). Every counter tokenizes it as
content, which is correct (we measure the corpus as published, not a cleaned
variant), but it is worth knowing: it was the sole source of the only
cross-tokenizer divergence ever seen in this dataset, when a SentencePiece and
an HF BPE loader split those tags differently. Separately, `vie_Latn` 745 carries
an undecoded `&amp;` entity. Both are upstream data, faithfully preserved.
"""
from __future__ import annotations

import hashlib
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


def _read(path, corpus: str, lang: str) -> str:
    """Read a corpus file, naming the remedy if it is absent.

    Adding a language to `config.LANGUAGES` without corpus files for it used to
    die with a bare FileNotFoundError from inside the loader — the one guard in
    this package that named neither the cause nor a fix.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise SystemExit(
            f"{corpus}: no corpus file for language '{lang}' at {path}. "
            f"config.LANGUAGES lists it, but the corpus does not carry it — add the "
            f"aligned file (for MASSIVE, extend build_massive.MASSIVE_LOCALE and "
            f"re-run it), or remove the language from config.LANGUAGES.") from None


def load_flores(lang: str) -> list[str]:
    """Return the FLORES+ sentences for `lang` (dev + devtest), NFC-normalized."""
    from .measure import nfc

    sentences: list[str] = []
    for split, ext in (("dev", "dev"), ("devtest", "devtest")):
        path = config.FLORES_DIR / split / f"{lang}.{ext}"
        sentences.extend(nfc(ln) for ln in _lines(_read(path, "flores", lang), str(path)))
    return sentences


def load_massive(lang: str) -> list[str]:
    """Return the committed MASSIVE utterances for `lang`, NFC-normalized."""
    from .measure import nfc

    path = config.MASSIVE_DIR / f"{lang}.txt"
    return [nfc(ln) for ln in _lines(_read(path, "massive", lang), str(path))]


_LOADERS: dict[str, Callable[[str], list[str]]] = {
    "flores": load_flores,
    "massive": load_massive,
}


def _check_corpus_registry() -> None:
    """`config.CORPORA` and `_LOADERS` are two registries describing one thing.

    A corpus declared in config with no loader here (or the reverse) used to
    surface only at load time, deep inside a run. Assert the symmetry at import,
    the same way config asserts MODEL_MATRIX <-> PRICING.
    """
    declared, implemented = set(config.CORPORA), set(_LOADERS)
    if declared != implemented:
        raise ValueError(
            f"corpus registries disagree — declared in config.CORPORA but with no "
            f"loader: {sorted(declared - implemented)}; loader registered but not "
            f"declared: {sorted(implemented - declared)}")


_check_corpus_registry()


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


def corpus_fingerprint(corpus: str, langs: list[str]) -> dict[str, object]:
    """Line count + SHA-256 over the NFC text of `corpus`, for provenance.

    Carry-forward preserves token totals measured against a *particular* corpus
    version, while `corpus_size()` and the per-character denominator are re-read
    from disk at analyze time. Nothing tied the two together: rebuilding the
    MASSIVE slice (its intersection size is data-dependent) or editing a corpus
    file silently re-normalized every carried counter's per-sentence cost, with
    exit 0 and no warning. Recording this in the manifest — and refusing to carry
    rows across a fingerprint change — closes that gap.

    Hashed over the loaded, NFC-normalized sentences, so it is invariant to the
    BOM/CRLF differences the loader already tolerates and sensitive to exactly
    what the counters see.
    """
    data = load_corpus(corpus, langs)
    h = hashlib.sha256()
    for lang in langs:                      # caller-stable order
        h.update(lang.encode("utf-8"))
        h.update(b"\0")
        h.update("\n".join(data[lang]).encode("utf-8"))
        h.update(b"\0")
    return {"n_sentences": len(data[langs[0]]), "sha256": h.hexdigest()}


def corpus_size(corpus: str) -> int:
    """Number of aligned sentences/utterances in `corpus`.

    The corpora are parallel (one line per sentence, aligned across languages),
    so the count is language-independent — measured on the baseline language.
    This is corpus metadata and the honest denominator for a *per-sentence* cost,
    distinct from the per-character one: a dense script says the same thing in far
    fewer characters, so the two denominators rank the languages differently.
    """
    return len(_loader(corpus)(config.BASELINE_LANG))
