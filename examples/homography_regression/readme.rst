Homography Regression by Gradient Descent
=========================================

This examples show how to use the `HomographyWarper` in order to do a regression where the parameter to optimize in this case is the homography driven by the gradient from a photometric loss.

Downloading the data
====================

You can download the data by running:  ``./download_data.sh``

Usage
=====

1. From root, run the docker developement or build first if needed: ``//torchgeometry/dev_en.sh``
2. Browse to ``cd /code/torchgeometry/examples/homography_regression``
3. Install the dependencies by running: ``./install_dependencies.sh``
4. Now you can run the example followingthe instructions below:

.. code:: bash

 python main.py --input-dir ./data --output-dir ./out --num-iterations 1000 --log-interval-vis 200 --cuda --lr 1e-3


.. code:: bash

main.py [-h] --input-dir INPUT_DIR --output-dir OUTPUT_DIR
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
  --cuda                enables CUDA training
  --seed S              random seed (default: 666)
  --log-interval N      how many batches to wait before logging training
                        status
  --log-interval-vis N  how many batches to wait before visual logging
                        training status
