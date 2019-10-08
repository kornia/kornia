#!/bin/bash
# This script is useful to download the example data

DATA_DIR=./data

mkdir -p $DATA_DIR
wget http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/img1.ppm -P $DATA_DIR
wget http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/img2.ppm -P $DATA_DIR
wget http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/H1to2p -P $DATA_DIR

echo "## Succeded to download files to $DATA_DIR"
