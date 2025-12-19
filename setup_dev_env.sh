#!/bin/bash -ex
script_link="$( readlink "$BASH_SOURCE" )" || script_link="$BASH_SOURCE"
apparent_sdk_dir="${script_link%/*}"
if [ "$apparent_sdk_dir" == "$script_link" ]; then
  apparent_sdk_dir=.
fi
sdk_dir="$( command cd -P "$apparent_sdk_dir" > /dev/null && pwd -P )"

# create root directory for the virtual environment
mkdir -p $sdk_dir

# check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# define a python version to initialise the virtual environment.
# by default we assume python 3.11
python_version=${PYTHON_VERSION:-"3.11"}
pytorch_version=${PYTORCH_VERSION:-"2.4.0"}
pytorch_mode=${PYTORCH_MODE:-""}  # use `cpuonly` for CPU or leave it in blank for GPU
cuda_version=${CUDA_VERSION:-"12.1"}

# check if virtual environment already exists
if [ -d "$sdk_dir/venv" ]; then
    echo "Virtual environment found, activating..."
    source $sdk_dir/venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"
    echo "Python: $(which python)"
    echo "To install/update dependencies, delete the venv directory and run this script again."
    exit 0
fi

# create virtual environment with uv
echo "Creating virtual environment with Python $python_version..."
uv venv $sdk_dir/venv --python $python_version

# activate virtual environment
source $sdk_dir/venv/bin/activate

# configure pytorch installation based on mode
if [ "$pytorch_mode" == "cpuonly" ]; then
    pytorch_index_url="https://download.pytorch.org/whl/cpu"
else
    # For CUDA, use the appropriate index URL
    pytorch_index_url="https://download.pytorch.org/whl/cu${cuda_version//./}"
fi

# install pytorch and torchvision
echo "Installing PyTorch $pytorch_version..."
if [ "$pytorch_version" == "nightly" ]; then
    # Install nightly builds
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu${cuda_version//./}
elif [ "$pytorch_version" != "" ]; then
    # Install specific version
    uv pip install torch==$pytorch_version torchvision --index-url $pytorch_index_url
else
    # Install latest stable
    uv pip install torch torchvision --index-url $pytorch_index_url
fi

# install project dependencies with dev and x extras
echo "Installing development dependencies..."
if [ -f "uv.lock" ]; then
    echo "Using uv.lock for reproducible dependency installation..."
    uv sync --frozen --group dev --group x --group docs
else
    echo "No uv.lock found, installing from pyproject.toml..."
    uv sync --group dev --group x --group docs
fi

echo "Development environment setup complete!"
echo "To activate the environment, run: source $sdk_dir/venv/bin/activate"
echo "To deactivate, run: deactivate"
