# ACORN-Edu Research Pipeline Makefile
# Gold-standard automation for research simulation

.PHONY: setup lint test run clean

setup:
	python -m pip install -U pip
	pip install -r requirements.txt

lint:
	flake8 simulation_harness.py tests

test:
	pytest -q

test-units:
	pytest tests/test_units.py -q

run:
	python simulation_harness.py

clean:
	rm -rf data figures