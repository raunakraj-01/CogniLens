.PHONY: setup test run run-replay collect synth dataset train eval report clean

PYTHON ?= ./venv/bin/python
PIP ?= ./venv/bin/pip
PYTEST ?= $(PYTHON) -m pytest

setup:
	python3 -m venv venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	mkdir -p models
	[ -f models/face_landmarker.task ] || curl -sSL -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

test:
	$(PYTEST) tests/

run:
	$(PYTHON) -m cogload.app.server

run-replay:
	$(PYTHON) -m cogload.app.server --video $(VIDEO)

collect:
	$(PYTHON) -m cogload.app.server --mode collect

synth:
	$(PYTHON) scripts/make_synthetic.py

dataset:
	$(PYTHON) -m cogload.data.build_dataset

train:
	$(PYTHON) scripts/run_eval.py --train-only

eval:
	$(PYTHON) scripts/run_eval.py

report:
	$(PYTHON) scripts/make_report.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
