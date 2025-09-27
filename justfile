# Run `just` to see all available commands

# Default recipe - show help when running just without arguments
default:
    @just --list

# Variables
benchmark_source := "benchmarks/"
benchmark_backends := "inductor,eager"
benchmark_opts := ""

# Ensure virtual environment is available and set up
# Note: This recipe doesn't actually activate the venv for subsequent commands
# because each line in a just recipe runs in a separate shell.
# Each command must explicitly use the venv paths.
_ensure-venv:
    #!/usr/bin/env bash
    if [[ ! -f "venv/bin/activate" ]]; then
        echo "🔧 Setting up virtual environment..."
        ./setup_dev_env.sh
    else
        echo "Virtual environment found!"
    fi

# Run all tests with coverage
test-all: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all --cov=kornia tests/

# Run CPU-only tests
test-cpu: _ensure-venv
    ./venv/bin/pytest -v --device cpu --dtype all --cov=kornia tests/

test-cpu-f32: _ensure-venv
    ./venv/bin/pytest -v --device cpu --dtype float32 --cov=kornia tests/

# Run CUDA tests
test-cuda: _ensure-venv
    ./venv/bin/pytest -v --device cuda --dtype all --cov=kornia tests/

test-cuda-f32: _ensure-venv
    ./venv/bin/pytest -v --device cuda --dtype float32 --cov=kornia tests/

# Run MPS tests (Apple Silicon)
test-mps: _ensure-venv
    ./venv/bin/pytest -v --device mps --dtype float32 -k "not (grad or exception or jit or dynamo)" tests/

# Run tests for a specific module (usage: just test-module tests/test_color.py)
test-module module: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all tests/{{module}}

# Run JIT compilation tests
test-jit: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all -m jit

# Run gradient check tests
test-gradcheck: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all -m grad

# Run neural network tests
test-nn: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all -m nn

# Run quick tests (excludes slow jit, grad, nn tests)
test-quick: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all -m "not (jit or grad or nn)"

# Run slow tests (jit, grad, nn)
test-slow: _ensure-venv
    ./venv/bin/pytest -v --device all --dtype all -m "(jit or grad or nn)"

# Run tests with coverage report
test-coverage: _ensure-venv
    coverage erase
    coverage run --source kornia/ -m pytest --device=all --dtype float32,float64 tests/
    coverage report

# Run linting with ruff
lint: _ensure-venv
    ./venv/bin/pre-commit run ruff --all-files

# Run type checking with mypy
mypy: _ensure-venv
    ./venv/bin/mypy

# Run doctests (local dev)
doctest: _ensure-venv
    ./venv/bin/pytest -v --doctest-modules kornia/

# CI doctests (no venv, uses system/uv environment)
ci-doctest:
    pytest -v --doctest-modules kornia/

# Build documentation (local dev)
build-docs: _ensure-venv
    cd docs && make clean html

# CI docs build (no venv)
ci-build-docs:
    sphinx-build -W -b html docs/source docs/build/html

# Install package in development mode
install-dev: _ensure-venv
    ./venv/bin/uv pip install -e .

# Run benchmarks
benchmark *args: _ensure-venv
    ./venv/bin/pytest {{benchmark_source}} --benchmark-warmup=on --benchmark-warmup-iterations=100 --benchmark-calibration-precision=10 --benchmark-group-by=func --optimizer={{benchmark_backends}} {{benchmark_opts}} {{args}}

# Build and run benchmark Docker container
benchmark-docker:
    docker image rm kornia-benchmark:latest --force || true
    docker build -t kornia-benchmark:latest -f docker/Dockerfile.benchmark .
    docker run -e "TERM=xterm-256color" \
               -e "BACKENDS={{benchmark_backends}}" \
               -e "OPTS={{benchmark_opts}}" \
               --gpus all \
               -it kornia-benchmark:latest

# Setup development environment
setup:
    ./setup_dev_env.sh

# Activate virtual environment (reminder command)
activate:
    @echo "Run: source venv/bin/activate"

# Show virtual environment status
venv-status:
    #!/usr/bin/env bash
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "Virtual environment active: $VIRTUAL_ENV"
        echo "Python: $(which python)"
        echo "Pip: $(which pip)"
    else
        echo "No virtual environment active"
        echo "Run: source venv/bin/activate"
    fi

# Clean up build artifacts and cache
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -rf build/ dist/ .coverage htmlcov/

# Show help
help:
    @just --list
