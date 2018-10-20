#!/bin/bash -ex

# install torchgeometry from source
cd ../.. && python setup.py install && cd examples/depth_warper

# we need last opencv
conda install opencv --yes
