# Reproduce everything:  make setup && make all     (15-25 min on an 11-core laptop; see results/environment.txt)
PY ?= .venv/bin/python

.PHONY: setup data test lint results evaluate figure all clean

setup:            ## create .venv with the pinned environment and install the package
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip   # pip >= 21.3 needed for editable pyproject installs
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

data:             ## download the source table (pinned upstream commit, SHA-256 verified)
	bash scripts/get_data.sh

test:             ## unit tests on synthetic data (no download needed)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests gee

results:          ## E1 reproduction, E3 132-pair transfer matrix, E4 overlap measures
	$(PY) -m aeoverlap.pipeline 2>&1 | tee results/pipeline.log

evaluate:         ## E5/E6 pre-registered tests -> results/tests.json, results/summary.md
	$(PY) -m aeoverlap.evaluate

figure:           ## results/figure.png
	$(PY) -m aeoverlap.figures

all: data test results evaluate figure

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
