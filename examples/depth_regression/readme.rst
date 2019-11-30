Gradient Descent Depth
======================

This example shows how to use the `DepthWarper` in order to regress the depth of the reference camera by  warping an image from a non reference camera to the reference
by the depth using the epipolar geometry constraints assumig a pinhole camera model.

Downloading the data
====================

You can download the data by running:  ``./download_data.sh``

Usage
=====

1. From root, run the docker developement or build first if needed: ``//kornia/dev_en.sh``
2. Browse to ``cd /code/kornia/examples/depth_warper``
3. Install the dependencies by running: ``./install_dependencies.sh``
4. Now you can run the example followingthe instructions below:

.. code:: bash

  python main.py --input-dir ./data --output-dir ./out --frame-ref-id 2 --frame-i-id 1

.. code:: bash

usage: main.py [-h] --input-dir INPUT_DIR --output-dir OUTPUT_DIR
               [--num-iterations N] [--sequence-name SEQUENCE_NAME]
               [--frame-ref-id FRAME_REF_ID] [--frame-i-id FRAME_I_ID]
               [--lr LR] [--cuda] [--seed S] [--log-interval N]
               [--log-interval-vis N]

