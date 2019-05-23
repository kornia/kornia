.PHONY: test test-cpu test-gpu lint mypy build-docs install uninstall FORCE

test: mypy lint test-cpu

test-cpu: FORCE
	pytest --typetest cpu -vx

test-cpu-cov: FORCE
	pytest --typetest cpu -vx --cov=kornia test

test-gpu: FORCE
	pytest --typetest cuda -vx

lint: FORCE
	python verify.py --check lint

autopep8: FORCE
	autopep8 --in-place --aggressive --recursive kornia/ test/

mypy: FORCE
	python verify.py --check mypy

build-docs: FORCE
	python verify.py --check build-docs

install: FORCE
	python setup.py install

uninstall: FORCE
	pip uninstall kornia

FORCE:
