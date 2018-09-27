Homography Regression by Gradient Descent
=========================================

This examples show how to use the `HomographyWarper` in order to do a regression where the parameter to optimize in this case is the homography driven by the gradient from a photometric loss.

Quick overview
--------------

.. code:: python

 # load the data
 img_src = load_image(os.path.join(args.input_dir, 'img1.ppm'))
 img_dst = load_image(os.path.join(args.input_dir, 'img2.ppm'))
    
 # instantiate the homography warper from `torchgeometry`
 height, width = img_src.shape[-2:]
 warper = dgm.HomographyWarper(height, width)

 # create the homography as the parameter to be optimized
 # NOTE: MyHomography is an inherited nn.Module class
 dst_homo_src = MyHomography().to(device)

 # create optimizer
 optimizer = optim.Adam(dst_homo_src.parameters(), lr=args.lr)

 # main training loop

 for iter_idx in range(args.num_iterations):
     # send data to device
     img_src, img_dst = img_src.to(device), img_dst.to(device)

     # warp the reference image to the destiny with current homography
     img_src_to_dst = warper(img_src, dst_homo_src())

     # compute the photometric loss
     loss = F.l1_loss(img_src_to_dst, img_dst)

     # compute gradient and update optimizer parameters
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()

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
