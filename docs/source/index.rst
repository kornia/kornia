
.. image:: https://github.com/kornia/data/raw/main/kornia_banner_pixie.png
   :align: center

State-of-the-art and curated Computer Vision algorithms for AI.

Kornia AI is on the mission to leverage and democratize the next generation of Computer Vision tools and Deep Learning libraries
within the context of an Open Source community.

.. code:: python

   >>> import kornia.geometry as K
   >>> registrator = K.ImageRegistrator('similarity')
   >>> model = registrator.register(img1, img2)

Ready to use with state-of-the art Deep Learning models:

DexiNed edge detection model.

.. code-block:: python

      image = kornia.utils.sample.get_sample_images()[0][None]
      model = DexiNedBuilder.build()
      model.save(image)

RTDETRDetector for object detection.

.. code-block:: python

      image = kornia.utils.sample.get_sample_images()[0][None]
      model = RTDETRDetectorBuilder.build()
      model.save(image)

BoxMotTracker for object tracking.

.. code-block:: python

      import kornia
      image = kornia.utils.sample.get_sample_images()[0][None]
      model = BoxMotTracker()
      for i in range(4):
         model.update(image)
      model.save(image)

Vision Transformer for image classification.

.. code:: python

   >>> import torch.nn as nn
   >>> from kornia.models import VisionTransformer
   >>> classifier = nn.Sequential(
   ...   VisionTransformer(image_size=224, patch_size=16),
   ...   nn.Linear(768, 1000),  # Example: 768 is the default hidden_dim, 1000 is num_classes
   ... )
   >>> logits = classifier(img)    # BxN
   >>> scores = logits.argmax(-1)  # B

Multi-framework support
-----------------------

You can now use Kornia with `NumPy <https://numpy.org/>`_, `TensorFlow <https://www.tensorflow.org/>`_, and `JAX <https://jax.readthedocs.io/en/latest/index.html>`_.

.. code:: python

  >>> import kornia
  >>> tf_kornia = kornia.to_tensorflow()

.. raw:: html

   <p align="center">
        Powered by
        <a href="https://github.com/ivy-llc/ivy" target="_blank">
            <div class="dark-light" style="display: block;" align="center">
                <img class="dark-light" width="15%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg"/>
            </div>
        </a>
    </p>

Join the community
------------------

- Join our social network communities with 1.8k+ members:
   - `Twitter <https://twitter.com/kornia_foss>`_: we share the recent research and news for out mainstream community.
   - `Slack <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA>`_: come to us and chat with our engineers and mentors to get support and resolve your questions.
- Subscribe to our `YouTube channel <https://www.youtube.com/channel/UCI1SE1Ij2Fast5BSKxoa7Ag>`_ to get the latest video demos.

----

.. toctree::
   :caption: GET STARTED
   :hidden:

   get-started/introduction
   get-started/highlights
   get-started/installation
   get-started/about
   Tutorials <https://kornia.github.io/tutorials/>
   get-started/multi-framework-support
   OpenCV AI Kit <https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/#kornia>
   get-started/governance

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   :hidden:

   augmentation
   color
   contrib
   core
   enhance
   feature
   filters
   geometry
   sensors
   io
   image
   losses
   models
   metrics
   morphology
   onnx
   tracking
   utils

.. toctree::
   :caption: KORNIA APPLICATIONS
   :hidden:

   applications/intro
   applications/visual_prompting
   applications/face_detection
   applications/image_augmentations
   applications/image_matching
   applications/image_stitching
   applications/image_registration
   applications/image_denoising

.. toctree::
   :caption: KORNIA MODELS
   :hidden:

   models/efficient_vit
   models/rt_detr
   models/segment_anything
   models/mobile_sam
   models/yunet
   models/vit
   models/vit_mobile
   models/tiny_vit
   models/loftr
   models/defmo
   models/hardnet
   models/affnet
   models/sold2
   models/dexined

.. toctree::
   :caption: SUPPORT
   :hidden:

   Issue tracker <https://github.com/kornia/kornia/issues>
   Slack community <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA>
   LibreCV community <https://librecv.org>
   Twitter @kornia_foss <https://twitter.com/kornia_foss>
   community/chinese
   Kornia Youtube <https://www.youtube.com/channel/UCI1SE1Ij2Fast5BSKxoa7Ag>
   Kornia LinkedIn <https://www.linkedin.com/company/kornia/>
   Kornia AI <https://kornia.org>

.. toctree::
   :caption: COMMUNITY
   :hidden:

   community/contribute
   community/faqs
   community/bibliography
