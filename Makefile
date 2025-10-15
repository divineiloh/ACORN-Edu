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

verify-artifacts:
	@test -f data/bap_network_scenario_results.csv
	@test -f data/bap_ablation_study_results.csv
	@test -f figures/network_bytes.png
	@test -f figures/network_hitrate.png
	@test -f figures/ablation_hitrate.png
	@test -f figures/ablation_bytes.png