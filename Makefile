.PHONY: test test_cpu test_gpu

test: test_cpu test_gpu

test_cpu: FORCE
	python setup.py install && pytest --typetest cpu

test_gpu: FORCE
	python setup.py install && pytest --typetest cuda

FORCE:
