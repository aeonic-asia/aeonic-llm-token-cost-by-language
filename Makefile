# Aeonic token-cost eval — reproducible, offline from committed data.
#
#   make setup       create the eval venv and install pinned deps
#   make reproduce   run counters -> analysis -> figures (offline)
#   make test        correctness gates: the paper's cl100k oracle + the
#                    carry-forward invariant (committed rows are never lost)
#   make clean       remove generated results (keeps the committed dataset in git)
#
# Offline: the tiktoken BPE ranks are committed under eval/tiktoken_cache/, so
# no network is needed once `make setup` has installed the pinned wheels.

VENV := .venv-eval
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export TIKTOKEN_CACHE_DIR := $(CURDIR)/eval/tiktoken_cache

.PHONY: setup reproduce test figures clean check-venv

setup:
	python3 -m venv $(VENV)
	$(PIP) install -q --disable-pip-version-check -r requirements-eval.txt

# Guard: reproduce/test/figures need the venv from `make setup`. Without this
# they fail with a cryptic "no such file: .venv-eval/bin/python".
check-venv:
	@test -x $(PY) || { echo "eval venv missing — run 'make setup' first"; exit 1; }

# CORPORA / COUNTERS (optional): comma-separated subsets to (re)measure, e.g.
#   make reproduce CORPORA=massive
#   make reproduce COUNTERS=gemini-3-1-pro       # add one counter, keep the rest
# Cells outside the selected (corpus × counter) grid are carried forward verbatim
# from the committed dataset. Empty (default) = full rebuild of everything.
CORPORA ?=
COUNTERS ?=
reproduce: check-venv
	EVAL_CORPORA='$(CORPORA)' EVAL_COUNTERS='$(COUNTERS)' $(PY) -m eval.run
	$(PY) -m eval.analyze
	$(PY) -m eval.figures

# Guard the DATA precondition, not just the venv: `make clean` removes exactly
# the two CSVs figures reads, so `make clean && make figures` — both documented
# targets — used to die with a bare FileNotFoundError.
figures: check-venv
	@test -f eval/results/premium_by_language.csv -a -f eval/results/cost_by_language.csv \
	  || { echo "analysis outputs missing (removed by 'make clean'?) — run 'make reproduce' first"; exit 1; }
	$(PY) -m eval.figures

# Two gates: the oracle checks the measurement core against the paper; the
# carry-forward suite checks the plumbing that decides which measurements survive
# a pass. The second is the one that has actually broken.
test: check-venv
	$(PY) -m unittest discover -s eval/tests -t . -v

# Removes every GENERATED analysis artifact (keeps raw_counts/aggregate_counts —
# the committed measured dataset — in git). Lists each generated file so a new
# artifact isn't silently left behind on a partial run.
clean:
	rm -rf eval/results/figures
	rm -f eval/results/summary.json eval/results/premium_by_language.csv \
	      eval/results/cost_by_language.csv eval/results/within_vendor_inflation.csv \
	      eval/results/run_manifest.json
