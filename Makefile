.PHONY: test test-cpu test-cuda lint mypy build-docs install uninstall FORCE

test: mypy lint build-docs test-all

test-all: FORCE
	pytest -n auto -v --device all --dtype all --cov=kornia test/

test-cpu: FORCE
	pytest -n auto -v --device cpu --dtype all --cov=kornia test/

test-cuda: FORCE
	pytest -n auto -v --device cuda --dtype all --cov=kornia test/

test-mps: FORCE
	pytest -n auto -v --device mps --dtype float32 -k "not (grad or exception or jit or dynamo)"  test/

test-module: FORCE
	pytest -n auto -v --device all --dtype all  test/$(module)

test-jit: FORCE
	pytest -n auto -v --device all --dtype all -m jit

test-gradcheck: FORCE
	pytest -n auto -v --device all --dtype all -m grad

test-nn: FORCE
	pytest -n auto -v --device all --dtype all -m nn

test-quick: FORCE
	pytest -n auto -v --device all --dtype all -m "not (jit or grad or nn)"

test-slow: FORCE
	pytest -n auto -v --device all --dtype all -m "(jit or grad or nn)"

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
	for f in test/performance/*.py  ; do python -utt $${f}; done

uninstall: FORCE
	pip uninstall kornia
