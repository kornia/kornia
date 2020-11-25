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
platform=`uname`
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
<<<<<<< refs/remotes/kornia/master
pytorch_version=${PYTORCH_VERSION:-"1.7.0"}
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
=======
pytorch_version=${PYTORCH_VERSION:-"1.6.0"}
pytorch_mode=${PYTORCH_MODE:-""}  # use `cpuonly` for CPU or leave it in blank for GPU
<<<<<<< refs/remotes/kornia/master
>>>>>>> refactor setup_dev_env script (#756)
=======
cuda_version=${CUDA_VERSION:-""}

cuda_toolkit=""
if [ ! -z "$cuda_version" ]
then
    cuda_toolkit="cudatoolkit=$cuda_version"
fi
>>>>>>> [Feat] update to pytorch 1.7 (#768)

# create an environment with the specific python version
$conda_bin config --append channels conda-forge
$conda_bin update -n base -c defaults conda
$conda_bin create --name venv python=$python_version
$conda_bin clean -ya

# activate local virtual environment
source $conda_bin_dir/activate $dev_env_dir/envs/venv

# install pytorch and torchvision
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
conda install pytorch=$pytorch_version torchvision $cuda_toolkit $pytorch_mode -c $pytorch_channel
=======
conda install pytorch=$pytorch_version torchvision $pytorch_mode -c pytorch
>>>>>>> refactor setup_dev_env script (#756)
=======
conda install pytorch=$pytorch_version torchvision $cuda_toolkit $pytorch_mode -c pytorch
>>>>>>> [Feat] update to pytorch 1.7 (#768)

# install testing dependencies
pip install -r requirements-dev.txt

# install documentation dependencies
pip install -r docs/requirements.txt

conda deactivate  # close the `venv` environment
