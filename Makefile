# Convenience targets. Everything runs inside the project virtual environment.

PYTHON_BOOTSTRAP ?= /usr/local/bin/python3.11
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: help setup test validate experiments figures docs all clean

help:
	@echo "make setup        Create .venv and install requirements"
	@echo "make test         Run the unit test suite"
	@echo "make validate     Sanity-check the environment with non-learning agents"
	@echo "make experiments  Run every experiment and write results/raw/*.csv"
	@echo "make figures      Rebuild every figure from the existing CSVs"
	@echo "make docs         Regenerate the generated documentation"
	@echo "make all          setup + test + experiments + figures"
	@echo "make clean        Remove caches and generated results"

setup:
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

validate:
	$(PY) -m experiments.validate_environment

experiments:
	$(PY) -m experiments.run_all

figures:
	$(PY) -m experiments.run_all --figures-only

docs:
	$(PY) -m src.environment.observation > docs/STATE_SPEC.md

all: setup test experiments figures

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -f results/raw/*.csv results/processed/*.csv results/figures/*.png
