setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt

run:
	. .venv/bin/activate && python acorn.py

clean:
	rm -f data/*.csv data/run_metadata.json figures/*.png
