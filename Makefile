PY := .venv/bin/python -I

.PHONY: setup data demo-data run run-demo test

setup:            ## create the virtual environment and install the app
	uv venv --python 3.12
	uv pip install -e ".[dev]"

data:             ## download the public cohort (about 80 MB)
	$(PY) scripts/fetch_data.py

demo-data:        ## regenerate the synthetic serial study
	$(PY) scripts/make_synthetic_demo.py

run:              ## public cohort at http://127.0.0.1:8050
	$(PY) -m variant_browser.app

run-demo:         ## synthetic study, with recorded ID decisions, at http://127.0.0.1:8051
	$(PY) -m variant_browser.app --port 8051 --study data/synthetic_serial \
		--overrides data/synthetic_serial/overrides.example.tsv

test:
	$(PY) -m pytest -q
