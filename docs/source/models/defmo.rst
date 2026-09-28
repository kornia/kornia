DeFMO
=====

.. meta::
   :description: Deblur fast-moving objects into high-speed RGBA subframes with Kornia's pretrained DeFMO model.

.. rst-class:: kornia-badges

:bdg-primary:`Enhance` :bdg-primary:`Deblurring` :bdg-secondary:`Apache-2.0`

DeFMO takes a single frame in which a fast-moving object is motion-blurred, plus an estimate of the static
background, and renders the object sharp at 24 moments of the exposure, as a high-speed camera would have seen it.
:class:`~kornia.feature.DeFMO` takes the two images stacked along the channel axis and returns RGBA sub-frames.

Run it
------

.. code-block:: python

    import torch
    from kornia.feature import DeFMO
    from kornia.io import load_image

    blurred = load_image("blurred.png")[None]  # (1, 3, 240, 320) frame with a motion-blurred object
    background = load_image("background.png")[None]  # (1, 3, 240, 320) the same view without the object

    defmo = DeFMO(pretrained=True).eval()
    with torch.no_grad():
        subframes = defmo(torch.cat([blurred, background], dim=1))  # (1, 24, 4, 240, 320) RGBA sub-frames

    rgba = subframes[0, 0]  # first sub-frame: rgb = rgba[:3], alpha = rgba[3:]
    composed = rgba[3:] * rgba[:3] + (1 - rgba[3:]) * background[0]  # paste it back onto the background

.. figure:: /_static/img/models/defmo.jpg
   :align: center
   :alt: Top row, the blurred input, the background and two ground-truth sub-frames of a striped ball; bottom row, four DeFMO sub-frames showing the sharp ball at successive positions.

   A striped ball rendered at 24 positions over a street photo and averaged into one blurred frame (top left).
   DeFMO recovers its shape and trajectory from that frame and the background (bottom, four of the 24 sub-frames,
   composited onto the background). The direction of motion is ambiguous from a single blur, so the sub-frame
   order may be reversed.

The network is fully convolutional and the output keeps the input resolution; the pretrained weights were trained
on ``240×320`` crops around the object, so results are best near that scale. Each sub-frame's alpha channel is the
object mask, so the trajectory is the sequence of alpha centroids.

Paper
-----

.. card::
    :link: https://paperswithcode.com/paper/defmo-deblurring-and-shape-recovery-of-fast

    **DeFMO: Deblurring and Shape Recovery of Fast Moving Objects**
    ^^^
    **Abstract:** Objects moving at high speed appear significantly blurred when captured with cameras. The blurry appearance is especially ambiguous when the object has complex shape or texture. In such cases, classical methods, or even humans, are unable to recover the object's appearance and motion. We propose a method that, given a single image with its estimated background, outputs the object's appearance and position in a series of sub-frames as if captured by a high-speed camera (i.e. temporal super-resolution). The proposed generative model embeds an image of the blurred object into a latent space representation, disentangles the background, and renders the sharp appearance. Inspired by the image formation model, we design novel self-supervised loss function terms that boost performance and show good generalization capabilities. The proposed DeFMO method is trained on a complex synthetic dataset, yet it performs well on real-world data from several datasets. DeFMO outperforms the state of the art and generates high-quality temporal super-resolution frames.

    **Tasks:** Deblurring, Object Tracking, Super-Resolution, Video Super-Resolution.

    **Datasets:** Falling Objects.

    **Conference:** CVPR 2021

    **Licence:** Apache-2.0

    +++
    **Authors:** Denys Rozumnyi, Martin R. Oswald, Vittorio Ferrari, Jiri Matas, Marc Pollefeys

..  youtube:: pmAynZvaaQ4
