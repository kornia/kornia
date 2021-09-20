
.. image:: https://github.com/kornia/data/raw/main/kornia_banner_pixie.png
   :align: center

State-of-the-art and curated Computer Vision algorithms for AI.

Kornia AI is on the mission to leverage and democratize the next generation of Computer Vision tools and Deep Learning libraries
within the context of an Open Source community.

.. code:: python

   >>> import kornia.geometry as K
   >>> registrator = K.ImageRegistrator('similarity')
   >>> model = registrator(img1, img2)

Ready to use with state-of-the art Deep Learning models:

.. code:: python

   >>> import kornia.contrib as K
   >>> classifier = nn.Sequential(
   ...   K.VisionTransformer(image_size=224, patch_size=16),
   ...   K.ClassficationHead(num_classes=1000),
   ... )
   >>> logits = classifier(img)    # BxN
   >>> scores = logits.argmax(-1)  # B

Join the community
------------------

- Join our social network communities with 1.8k+ members:
   - `Twitter <https://twitter.com/kornia_foss>`_: we share the recent research and news for out mainstream community.
   - `Slack <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ>`_: come to us and chat with our engineers and mentors to get support and resolve your questions.
   - `LibreCV <https://librecv.org>`_: its our Open Source and Machine Learning community forum. Come and have fun !
- Subscribe to our `YouTube channel <https://www.youtube.com/channel/UCI1SE1Ij2Fast5BSKxoa7Ag>`_ to get the latest video demos.

----

.. toctree::
   :caption: GET STARTED
   :hidden:

   get-started/introduction
   get-started/highlights
   get-started/installation
   get-started/about
   Tutorials <https://kornia-tutorials.readthedocs.io/en/latest/>

.. toctree::
   :caption: KORNIA APPLICATIONS
   :hidden:

   applications/intro
   applications/image_registration
   applications/image_augmentations
   applications/image_matching
   applications/video_deblur

.. toctree::
   :caption: KORNIA MODELS
   :hidden:

   models/vit
   models/loftr
   models/defmo
   models/hardnet
   models/affnet

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   :hidden:

   augmentation
   color
   contrib
   enhance
   feature
   filters
   geometry
   losses
   morphology
   utils

.. toctree::
   :caption: SUPPORT
   :hidden:

   Issue tracker <https://github.com/kornia/kornia/issues>
   Slack community <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ>
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
   community/governance
   community/bibliography
