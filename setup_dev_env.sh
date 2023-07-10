#!/bin/bash -ex
script_link="$( readlink "$BASH_SOURCE" )" || script_link="$BASH_SOURCE"
apparent_sdk_dir="${script_link%/*}"
if [ "$apparent_sdk_dir" == "$script_link" ]; then
  apparent_sdk_dir=.
fi
sdk_dir="$( command cd -P "$apparent_sdk_dir" > /dev/null && pwd -P )"

# create root directory to install miniconda
dev_env_dir=$sdk_dir/.dev_env
mkdir -p $dev_env_dir

# define miniconda paths
conda_bin_dir=$dev_env_dir/bin
conda_bin=$conda_bin_dir/conda

# download and install miniconda
# check the operating system: Mac or Linux
platform=$(uname)
if [[ "$platform" == "Darwin" ]];
then
 download_link=https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
 download_link=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

if [ ! -e $dev_env_dir/miniconda.sh ]; then
    curl -o $dev_env_dir/miniconda.sh \
       	 -O  "$download_link"
    chmod +x $dev_env_dir/miniconda.sh
fi
if [ ! -e $conda_bin ]; then
    $dev_env_dir/miniconda.sh -b -u -p $dev_env_dir
fi

# define a python version to initialise the conda environment
python_version=${PYTHON_VERSION:-"3.10"}
pytorch_version=${PYTORCH_VERSION:-"2.0.1"}
pytorch_mode=${PYTORCH_MODE:-""}  # use `cpuonly` for CPU or leave it in blank for GPU
cuda_version=${CUDA_VERSION:-"11.8"}

# configure for nightly builds
pytorch_channel="pytorch"
if [ "$pytorch_version" == "nightly" ]; then
    pytorch_version=""
    pytorch_channel="pytorch-nightly"
fi

# configure pytorch cuda version
if [ "$pytorch_mode" == "cpuonly" ]; then
    pytorch_cuda_version="cpuonly"
else
    pytorch_cuda_version="pytorch-cuda=$cuda_version -c nvidia"
fi

# create an environment with the specific python version
$conda_bin config --append channels conda-forge
$conda_bin update -n base -c defaults conda
$conda_bin create --name venv python=$python_version
$conda_bin clean -ya

# activate local virtual environment
source $conda_bin_dir/activate $dev_env_dir/envs/venv

# install pytorch
conda install pytorch=$pytorch_version $pytorch_cuda_version -c $pytorch_channel

# install testing dependencies
pip install -e .[dev,x]

# install docs dependencies
pip install -e .[docs]

conda deactivate  # close the `venv` environment
