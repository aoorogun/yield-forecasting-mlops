.PHONY: install data train evaluate scenario monitor test lint all clean

install:
	pip install -e ".[dev]"

data:
	yield-forecast generate-data

train:
	yield-forecast train

evaluate:
	yield-forecast evaluate

scenario:
	yield-forecast reforecast --scenario configs/scenario_example.json

monitor:
	yield-forecast monitor

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src tests

all: data train evaluate scenario monitor test

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

