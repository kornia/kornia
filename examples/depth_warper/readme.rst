Warp Image by Depth
===================

This example shows how to use the `DepthWarper` in order to warp an image from a reference camera to a destination 
by the depth using the epipolar geometry constraints assumig a pinhole camera model.

Downloading the data
====================

You can download the data by running:  ``./download_data.sh``

Usage
=====

1. From root, run the docker developement or build first if needed: ``//torchgeometry/dev_en.sh``
2. Browse to ``cd /code/torchgeometry/examples/depth_warper``
3. Install the dependencies by running: ``./install_dependencies.sh``
4. Now you can run the example followingthe instructions below:

.. code:: bash

  python main.py --input-dir ./data --output-dir ./out --frame-ref-id 2 --frame-i-id 1

.. code:: bash

usage: main.py [-h] --input-dir INPUT_DIR --output-dir OUTPUT_DIR
               [--sequence-name SEQUENCE_NAME]
               [--frame-ref-id FRAME_REF_ID]
               [--frame-i-id FRAME_I_ID] [--cuda]
               [--seed S]

Warp images by depth application.

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        the path to the directory with the input data.
  --output-dir OUTPUT_DIR
                        the path to output the results.
  --sequence-name SEQUENCE_NAME
                        the name of the sequence.
  --frame-ref-id FRAME_REF_ID
                        the id for the reference image in the sequence.
  --frame-i-id FRAME_I_ID
                        the id for the image i in the sequence.
  --cuda                enables CUDA training
  --seed S              random seed (default: 666)
