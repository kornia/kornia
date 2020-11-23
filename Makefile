.PHONY: test test-cpu test-cuda lint mypy build-docs install uninstall FORCE

test: mypy lint build-docs test-all
# TODO: Add cuda-float16 when #649 is solved
test-all: FORCE
	pytest -v --device all --dtype float32,float64 --cov=kornia test/ --flake8 --mypy

test-cpu: FORCE
	pytest -v --device cpu --dtype all --cov=kornia test/ --flake8 --mypy

test-cuda: FORCE
	pytest -v --device cuda --dtype all --cov=kornia test/ --flake8 --mypy

test-module: FORCE
	pytest -v --device all --dtype all  test/$(module) --flake8 --mypy

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
	pytest -v --flake8 -m flake8

mypy: FORCE
	pytest -v --mypy -m mypy test/

autopep8: FORCE
	autopep8 --in-place --aggressive --recursive kornia/ test/ examples/

doctest:
	pytest -v --doctest-modules kornia/color kornia/augmentation #... etc

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
