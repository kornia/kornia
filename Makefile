.PHONY: test test-cpu test-cuda lint mypy build-docs install uninstall FORCE

test: mypy lint build-docs test-all

test-all: FORCE
	pytest -v --device cpu,cuda --dtype float16,float32,float64 --cov=kornia test/

test-cpu: FORCE
	pytest -v --device cpu --dtype float32,float64 --cov=kornia test/

test-cuda: FORCE
	pytest -v --device cuda --dtype float16,float32,float64 --cov=kornia test/

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
	
install-dev: FORCE
	python setup.py develop

benchmark: FORCE
	for f in test/performance/*.py  ; do python -utt $${f}; done

uninstall: FORCE
	pip uninstall kornia

FORCE:
