.PHONY: test test-cpu test-cuda lint mypy build-docs install uninstall FORCE

test: mypy lint build-docs test-all

test-all: FORCE
	pytest -v --device all --dtype all --cov=kornia test/

test-cpu: FORCE
	pytest -v --device cpu --dtype all --cov=kornia test/

test-cuda: FORCE
	pytest -v --device cuda --dtype all --cov=kornia test/

test-mps: FORCE
	pytest -v --device mps --dtype float32 -k "not (grad or exception or jit or dynamo)"  test/

test-module: FORCE
	pytest -v --device all --dtype all  test/$(module)

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

lint: FORCE
	pre-commit run flake8 --all-files

mypy: FORCE
	mypy

yapf: FORCE
	yapf --in-place --parallel --recursive kornia/ test/ examples/

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
