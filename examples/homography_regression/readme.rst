Homography Regression by Gradient Descent
=========================================

This examples show how to use the `HomographyWarper` in order to do a regression where the parameter to optimize in this case is the homography driven by the gradient from a photometric loss.

Downloading the data
====================

You can download the data by running:  ``./download_data.sh``

Usage
=====

.. code:: bash

usage: main.py [-h] --input-dir INPUT_DIR --output-dir OUTPUT_DIR
               [--num-iterations N] [--lr LR] [--momentum M] [--cuda]
               [--seed S] [--log-interval N]

Homography Regression with perception loss.

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        the path to the directory with the input data.
  --output-dir OUTPUT_DIR
                        the path to output the results.
  --num-iterations N    number of training iterations (default: 1000)
  --lr LR               learning rate (default: 1e-3)
  --momentum M          SGD momentum (default: 0.9)
  --cuda                enables CUDA training
  --seed S              random seed (default: 666)
  --log-interval N      how many batches to wait before logging training
                        status

