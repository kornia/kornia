.PHONY: test test-cpu test-gpu lint mypy build-docs install uninstall FORCE

test: test_cpu test_gpu

test-cpu: FORCE
	pytest --typetest cpu -vx

test-cpu-cov: FORCE
	pytest --typetest cpu -vx --cov=torchgeometry test

test-gpu: FORCE
	pytest --typetest cuda -vx

lint: FORCE
	python verify.py --check lint

autopep8: FORCE
	autopep8 --in-place --aggressive --recursive torchgeometry/ test/

mypy: FORCE
	python verify.py --check mypy

build-docs: FORCE
	python verify.py --check build-docs

install: FORCE
	python setup.py install

uninstall: FORCE
	pip uninstall torchgeometry

FORCE:
