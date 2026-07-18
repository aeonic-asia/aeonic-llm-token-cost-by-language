# Aeonic token-cost eval — reproducible, offline from committed data.
#
#   make setup       create the eval venv and install pinned deps
#   make reproduce   run counters -> analysis -> figures (offline)
#   make test        reproduce the paper's cl100k oracle (correctness gate)
#   make clean       remove generated results (keeps the committed dataset in git)
#
# Offline: the tiktoken BPE ranks are committed under eval/tiktoken_cache/, so
# no network is needed once `make setup` has installed the pinned wheels.

VENV := .venv-eval
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export TIKTOKEN_CACHE_DIR := $(CURDIR)/eval/tiktoken_cache

.PHONY: setup reproduce test figures clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install -q --disable-pip-version-check -r requirements-eval.txt

reproduce:
	$(PY) -m eval.run
	$(PY) -m eval.analyze
	$(PY) -m eval.figures

figures:
	$(PY) -m eval.figures

test:
	$(PY) -m unittest eval.tests.test_oracle -v

clean:
	rm -rf eval/results/figures
	rm -f eval/results/summary.json eval/results/premium_by_language.csv eval/results/cost_by_language.csv
