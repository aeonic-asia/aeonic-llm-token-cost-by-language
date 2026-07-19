# MASSIVE slice — provenance & license

The `*.txt` files here are a committed, parallel slice of the **MASSIVE** dataset,
used as the eval's second register (short virtual-assistant utterances) alongside
FLORES+'s formal prose.

- **Source:** MASSIVE 1.1 (`amazon-massive-dataset-1.1.tar.gz`), Amazon Science —
  dataset home <https://huggingface.co/datasets/AmazonScience/massive> (also
  <https://github.com/alexa/massive>).
- **Paper:** FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural
  Language Understanding Dataset with 51 Typologically-Diverse Languages*,
  arXiv:2204.08582.
- **License:** the **dataset** is **CC BY 4.0**
  (https://creativecommons.org/licenses/by/4.0/) — note this is the data license,
  distinct from the Apache-2.0 license on the `alexa/massive` tooling repo.
  Redistribution is permitted with attribution (as above) and an indication of
  changes (see "How derived"). Only a small derived slice is redistributed here.
- **How derived:** `python -m eval.build_massive` — the standard **dev** partition,
  intersected across the five locales (en-US, vi-VN, zh-CN, ru-RU, de-DE),
  NFC-normalized, one utterance per line aligned by line index. 2033 parallel
  utterances (comparable to FLORES+'s 2009). Regenerate with that script.

Each `{lang}.txt` uses the same FLORES+ language codes as the rest of the eval,
so `load_massive(lang)[i]` and `load_massive(other)[i]` are translations of the
same utterance.
