.PHONY: test test-cpu test-gpu lint mypy build-docs install uninstall FORCE

test: mypy lint test-cpu

test-cpu: FORCE
	pytest --typetest cpu -v

test-cpu-cov: FORCE
	pytest --typetest cpu -v --cov=kornia test

test-gpu: FORCE
	pytest --typetest cuda -v

lint: FORCE
	python verify.py --check lint

autopep8: FORCE
	autopep8 --in-place --aggressive --recursive kornia/ test/ examples/

mypy: FORCE
	python verify.py --check mypy

build-docs: FORCE
	python verify.py --check build-docs

install: FORCE
	python setup.py install
	
benchmark: 
	for f in test/performance/*.py  ; do python -utt $${f}; done

uninstall: FORCE
	pip uninstall kornia

FORCE:
