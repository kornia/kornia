#!/bin/bash
# This script is useful to download the example data

DATA_DIR=./data
IMAGES_FILE_NAME="MPI-Sintel-training_images.zip"
DEPTH_FILE_NAME="MPI-Sintel-depth-training-20150305.zip"

# clean previous and download data
rm -rf ${DATA_DIR} && mkdir -p ${DATA_DIR}
wget http://files.is.tue.mpg.de/sintel/${IMAGES_FILE_NAME} -P ${DATA_DIR}
wget http://files.is.tue.mpg.de/jwulff/sintel/${DEPTH_FILE_NAME} -P ${DATA_DIR}

# unzip to dir
unzip ${DATA_DIR}/${IMAGES_FILE_NAME} -d ${DATA_DIR}
unzip ${DATA_DIR}/${DEPTH_FILE_NAME} -d ${DATA_DIR}
echo "## Succeded to download files to $DATA_DIR"
