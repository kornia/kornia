.PHONY: test test_cpu test_gpu

test: test_cpu test_gpu

test_cpu: FORCE
	python setup.py install && pytest --typetest cpu

test_gpu: FORCE
	python setup.py install && pytest --typetest cuda

lint: FORCE
	python verify.py --check lint

mypy: FORCE
	python verify.py --check mypy

docs: FORCE
	$(MAKE) -C docs html

install: FORCE
	python setup.py install

uninstall: FORCE
	pip uninstall torchgeometry

FORCE:
