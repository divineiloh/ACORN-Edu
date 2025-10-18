setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt

run:
	. .venv/bin/activate && python acorn.py

run_bench:
	. .venv/bin/activate && python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario nightly_wifi
	. .venv/bin/activate && python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario spotty_cellular
	. .venv/bin/activate && python acorn.py --policy LRU_whole --trials 30 --seed-base 1337 --scenario nightly_wifi
	. .venv/bin/activate && python acorn.py --policy LRU_whole --trials 30 --seed-base 1337 --scenario spotty_cellular

ablation:
	. .venv/bin/activate && for f in alpha0 beta0 gamma0 delta0; do \
		python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario nightly_wifi --$$f; \
		python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario spotty_cellular --$$f; \
	done

figures:
	. .venv/bin/activate && python acorn.py

clean:
	rm -f data/*.csv data/run_metadata.json figures/*.png
