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

# define a python version to initialise the conda environment.
# by default we assume python 3.7.
python_version=${PYTHON_VERSION:-"3.7"}
pytorch_version=${PYTORCH_VERSION:-"1.10.2"}
pytorch_mode=${PYTORCH_MODE:-""}  # use `cpuonly` for CPU or leave it in blank for GPU
cuda_version=${CUDA_VERSION:-"10.2"}

# configure for nightly builds
pytorch_channel="pytorch"
if [ "$pytorch_version" == "nightly" ]; then
    pytorch_version=""
    pytorch_channel="pytorch-nightly"
fi

# configure cudatoolkit
cuda_toolkit="cudatoolkit=$cuda_version"
if [ "$pytorch_mode" == "cpuonly" ]
then
    cuda_toolkit=""
fi

# create an environment with the specific python version
$conda_bin config --append channels conda-forge
$conda_bin update -n base -c defaults conda
$conda_bin create --name venv python=$python_version
$conda_bin clean -ya

# activate local virtual environment
source $conda_bin_dir/activate $dev_env_dir/envs/venv

# install pytorch and torchvision
conda install pytorch=$pytorch_version torchvision $cuda_toolkit $pytorch_mode -c $pytorch_channel

# install all (testing and documentation) dependencies
pip install -e .[all]

conda deactivate  # close the `venv` environment
