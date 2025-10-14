# ACORN-Edu Research Pipeline Makefile
# Gold-standard automation for research simulation

.PHONY: setup run figures lint test clean

setup:
	python -m pip install -U pip
	pip install -r requirements.txt

run:
	python simulation_harness.py --runs 30 --separate-figs

figures:
	python scripts/plot_checks.py

lint:
	python -m compileall -q .

test:
	pytest -q

clean:
	rm -rf data/*.csv data/*.json figures/*.png