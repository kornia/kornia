BENCHMARK_SOURCE 	= benchmarks/
BENCHMARK_BACKENDS 	= inductor,eager
BENCHMARK_OPTS 		=

.PHONY: test test-cpu test-cuda lint mypy build-docs install uninstall FORCE

test: mypy lint build-docs test-all

test-all: FORCE
	pytest -v --device all --dtype all --cov=kornia tests/

test-cpu: FORCE
	pytest -v --device cpu --dtype all --cov=kornia tests/

test-cuda: FORCE
	pytest -v --device cuda --dtype all --cov=kornia tests/

test-mps: FORCE
	pytest -v --device mps --dtype float32 -k "not (grad or exception or jit or dynamo)"  tests/

test-module: FORCE
	pytest -v --device all --dtype all  tests/$(module)

test-jit: FORCE
	pytest -v --device all --dtype all -m jit

test-gradcheck: FORCE
	pytest -v --device all --dtype all -m grad

test-nn: FORCE
	pytest -v --device all --dtype all -m nn

test-quick: FORCE
	pytest -v --device all --dtype all -m "not (jit or grad or nn)"

test-slow: FORCE
	pytest -v --device all --dtype all -m "(jit or grad or nn)"

test-coverage: FORCE
	coverage erase && coverage run --source kornia/ -m pytest --device=all --dtype float32,float64 tests/ && coverage report

lint: FORCE
	pre-commit run ruff --all-files

mypy: FORCE
	mypy

doctest:
	pytest -v --doctest-modules kornia/

docstyle: FORCE
	pydocstyle kornia/

build-docs: FORCE
	cd docs; make clean html

install: FORCE
	python setup.py install

install-dev: FORCE
	python setup.py develop

benchmark: FORCE
	pytest $(BENCHMARK_SOURCE) --optimizer=$(BENCHMARK_BACKENDS) $(BENCHMARK_OPTS) $(0)

uninstall: FORCE
	pip uninstall kornia
