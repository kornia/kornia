#!/bin/bash -ex

# install torchgeometry from source
cd ../.. && python setup.py install && cd examples/homography_regression

# we need last opencv
conda install opencv --yes
