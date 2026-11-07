# Reproduce everything:  make setup && make all     (about 30-40 min on an 11-core laptop)
# Naming: e1-e6 = pre-registered (PREREGISTRATION.md), f1-f3 = exploratory, fig1/fig2 = figures.
PY ?= .venv/bin/python

.PHONY: setup data test lint preregistered evaluate exploratory figures all clean

setup:            ## create .venv with the pinned environment and install the package
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip   # pip >= 21.3 needed for editable pyproject installs
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

data:             ## download the source tables (pinned versions, SHA-256 verified)
	bash scripts/get_data.sh

test:             ## unit tests on synthetic data (no download needed)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests gee

preregistered:    ## E1 reproduction, E3 132-pair transfer matrix, E4 overlap -> results/preregistered/
	mkdir -p results/preregistered
	$(PY) -m aeoverlap.run_preregistered 2>&1 | tee results/preregistered/run.log

evaluate:         ## E5/E6 pre-registered tests -> results/preregistered/e5_e6_*
	$(PY) -m aeoverlap.evaluate_preregistered

exploratory:      ## F1-F3 exploratory follow-ups -> results/exploratory/
	mkdir -p results/exploratory
	$(PY) -m aeoverlap.run_exploratory 2>&1 | tee results/exploratory/run.log

figures:          ## results/figures/fig1_preregistered.png, fig2_exploratory.png
	$(PY) -m aeoverlap.figures

all: data test preregistered evaluate exploratory figures

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
