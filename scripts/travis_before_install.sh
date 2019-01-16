#!/usr/bin/env bash

PYTORCH_VERSION="1.0.0"

sudo apt-get update

if [ -f $HOME/miniconda/bin ]; then
    echo "[INFO] minicona already installed."
else
    echo "[INFO] Installing miniconda."
    rm -rf $HOME/miniconda
    mkdir -p $HOME/download
    if [[ -d $HOME/download/miniconda.sh ]] ; then rm -rf $HOME/download/miniconda.sh ; fi
    #wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    wget https://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a

echo "\n[INFO] Creating conda test-env.\n"
conda create -q -n test-env python=$TRAVIS_PYTHON_VERSION
source activate test-env

# Install PyTorch
echo "\n[INFO] Installing PyTorch ${PYTORCH_VERSION}.\n"
conda install -c pytorch pytorch-cpu==${PYTORCH_VERSION}

# Install current torchgeometry
echo "\n[INFO] Installing PyTorch Geometry.\n"
python setup.py install
