"""One-time corpus-prep: build the committed MASSIVE parallel slice.

NOT part of `make reproduce` (which stays fully offline). This derives the
committed `eval/massive/{lang}.txt` files from the MASSIVE 1.1 release once; the
reproduce path only ever reads those committed files. Re-run this only to
regenerate the slice (needs network + a working CA bundle; certifi is used if
present).

MASSIVE — FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural
Language Understanding Dataset with 51 Typologically-Diverse Languages"
(arXiv:2204.08582), Amazon Science. Licensed **CC BY 4.0** (see
eval/massive/PROVENANCE.md). It is fully parallel: the same utterance `id` is
human-translated across every locale, so a per-utterance premium vs. English is
well-defined — MASSIVE is the eval's *second register* (short virtual-assistant
utterances, e.g. "wake me up at five am this week") alongside FLORES+'s formal
encyclopedic prose.

Slice: the standard **dev** partition (~2033 parallel utterances, comparable to
FLORES+'s 2009), intersected across the five locales, NFC-normalized, one
utterance per line aligned by line index (same contract as load_flores).

Usage (from repo root):
    python -m eval.build_massive                     # download + build
    python -m eval.build_massive --data-dir DIR      # use pre-extracted jsonl
"""
from __future__ import annotations

import argparse
import json
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from . import config
from .measure import nfc

TARBALL_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)
PARTITION = "dev"
# eval language code (FLORES+ style) -> MASSIVE locale
MASSIVE_LOCALE: dict[str, str] = {
    "eng_Latn": "en-US",
    "vie_Latn": "vi-VN",
    "zho_Hans": "zh-CN",
    "rus_Cyrl": "ru-RU",
    "deu_Latn": "de-DE",
}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _jsonl_dir(data_dir: str | None) -> Path:
    """Return a dir holding the per-locale jsonl (download + extract if needed)."""
    if data_dir:
        return Path(data_dir)
    tmp = Path(tempfile.mkdtemp(prefix="massive-"))
    tgz = tmp / "massive.tar.gz"
    print(f"  downloading {TARBALL_URL} -> {tgz}")
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_ssl_context()))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(TARBALL_URL, tgz)
    members = [f"1.1/data/{loc}.jsonl" for loc in MASSIVE_LOCALE.values()]
    with tarfile.open(tgz) as t:
        t.extractall(tmp, members=[m for m in t.getnames() if m in members])
    return tmp / "1.1" / "data"


def _load_locale(path: Path) -> dict[str, str]:
    """id -> NFC utterance, for the chosen partition only."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("partition") != PARTITION:
                continue
            utt = nfc(d["utt"].strip())
            # newline-free is required by the line-aligned committed format
            if utt and "\n" not in utt:
                out[d["id"]] = utt
    return out


def build(data_dir: str | None = None) -> None:
    src = _jsonl_dir(data_dir)
    by_lang = {lang: _load_locale(src / f"{loc}.jsonl")
               for lang, loc in MASSIVE_LOCALE.items()}

    # Parallel slice = ids present (newline-free, non-empty) in *every* locale.
    common = set.intersection(*(set(d) for d in by_lang.values()))
    ordered = sorted(common, key=int)  # MASSIVE ids are integer strings
    dropped = {lang: len(d) - len(common) for lang, d in by_lang.items()}
    print(f"  parallel utterances: {len(ordered)} "
          f"(per-locale drop vs. intersection: {dropped})")

    config.MASSIVE_DIR.mkdir(parents=True, exist_ok=True)
    for lang, d in by_lang.items():
        lines = [d[i] for i in ordered]
        (config.MASSIVE_DIR / f"{lang}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {lang}.txt ({len(lines)} lines)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", help="dir with pre-extracted {locale}.jsonl "
                    "(skip download)")
    build(ap.parse_args().data_dir)
