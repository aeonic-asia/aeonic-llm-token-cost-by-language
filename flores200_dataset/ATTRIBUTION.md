# FLORES dataset — attribution & license

This directory holds a committed copy of the **FLORES-200** evaluation corpus
(the multilingual parallel benchmark from Meta AI's *No Language Left Behind*
project). It is redistributed here under its own license — **not** the
repository's top-level MIT LICENSE, which covers the code only.

- **License:** Creative Commons Attribution-ShareAlike 4.0 International
  (**CC BY-SA 4.0**). Redistribution and derivative use are permitted with
  attribution and share-alike.
- **Source / attribution:** NLLB Team et al., *No Language Left Behind: Scaling
  Human-Centered Machine Translation* — FLORES-200. The corpus is now maintained
  and distributed as **FLORES+** by the Open Language Data Initiative (OLDI),
  <https://github.com/openlanguagedata/flores> (also on the Hugging Face Hub as
  `openlanguagedata/flores_plus`), likewise under CC BY-SA 4.0.
- **Provenance in this repo:** inherited unchanged from the upstream fork
  (`AleksandarPetrov/tokenization-fairness`, MIT), which committed it at the repo
  root; the Aeonic eval reuses that copy rather than duplicating it.

### A note on naming — "FLORES+" vs "FLORES-200"

The eval and its docs refer to this corpus as **FLORES+** (the current
distribution name). The files committed here are the **FLORES-200** release that
FLORES+ continues; for these five languages the sentence sets are the same
parallel corpus. Both names denote the same data under the same CC BY-SA 4.0
license — "FLORES+" is used in prose, "flores200_dataset/" is the on-disk path
inherited from upstream.

The MASSIVE corpus (the second register, under `eval/massive/`) has its own
CC BY 4.0 attribution in `eval/massive/PROVENANCE.md`.
