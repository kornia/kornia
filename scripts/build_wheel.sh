#!/bin/bash -ex

## NOTE: change this each release
export KORNIA_BUILD_VERSION=0.1.3
export KORNIA_BUILD_NUMBER=2  # use 1 for final release, otherwise will be post2, post3, postN

# move to project root and create the wheel

cd ..
python3 setup.py sdist bdist_wheel

# upload

python3 -m twine upload dist/*

