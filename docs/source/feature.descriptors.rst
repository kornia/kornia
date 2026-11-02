Descriptors
===========

.. currentmodule:: kornia.feature

Patch descriptors and LAFs
--------------------------

:class:`SIFTDescriptor`, :class:`MKDDescriptor` and the learned :class:`HardNet`, :class:`HardNet8`,
:class:`HyNet`, :class:`SOSNet` and :class:`TFeat` describe grayscale patches ``(B, 1, PS, PS)``, one
descriptor per patch. ``PS`` is 32 for the learned networks, which raise for another size, and the
``patch_size`` argument of SIFT (41 by default) and MKD (32 by default). All but TFeat return L2-normalised
descriptors. HardNet, HardNet8, MKD and SIFT ignore a positive gain and an offset of the patch intensities, as do
SOSNet and TFeat up to the variance floor of their first ``InstanceNorm2d``; HyNet ignores only a positive gain (see
each Convention block). The learned networks load their pretrained weights by default.

To describe the LAFs of an image, :class:`LAFDescriptor`, which holds a pretrained :class:`HardNet` by default, or
:func:`get_laf_descriptors`, which takes the descriptor in each call, samples one patch per LAF with
:func:`extract_patches_from_pyramid` and runs the patch descriptor on it; pass the patch size of any descriptor
other than the default. The patches follow the orientation of their LAFs, so orient the LAFs first, e.g. with
:class:`LAFOrienter`. For SIFT descriptors of LAFs without sampling patches, use :class:`SIFTDescriptorFromPyramid`
(see below). The pipelines of :doc:`feature.local_features` combine a detector with these descriptors.
:class:`SOLD2` instead detects, describes and matches line segments of whole images.

.. _sift-descriptor-interfaces:

Choosing a SIFT descriptor interface
------------------------------------

These interfaces serve different inputs and outputs; none detects keypoints.
For detection and description together, use the :ref:`SIFT pipeline guide <sift-pipelines>`.

.. list-table:: SIFT descriptor contracts
   :header-rows: 1
   :widths: 25 35 40

   * - Interface
     - Input
     - Output and purpose
   * - :class:`SIFTDescriptor`
     - Canonically sampled grayscale patches ``(N, 1, patch_size, patch_size)``.
     - One descriptor per patch, ``(N, 128)`` with default bins. Use when patch
       extraction, scale normalization and orientation have already been handled.
   * - :class:`DenseSIFTDescriptor`
     - A grayscale image ``(B, 1, H, W)``.
     - A dense descriptor map ``(B, 128, H_out, W_out)`` with default bins.
       Spatial size depends on bin size, stride and padding.
   * - :class:`SIFTDescriptorFromPyramid`
     - A grayscale image and supplied, already oriented LAFs ``(B, N, 2, 3)``.
     - Sparse descriptors ``(B, N, 128)`` with default bins, sharing image-gradient
       histograms between frames. ``forward(image, lafs)`` preserves their orientation.

All three support RootSIFT normalization. They return kornia's angle-major layout;
:func:`convert_sift_descriptor_layout` reorders bins to OpenCV's layout. Layout
conversion does not make differently sampled descriptors numerically equivalent.
The converter expects the descriptor dimension last. For a dense map, move its
channels last before conversion and restore them afterwards::

    opencv_dense = convert_sift_descriptor_layout(
        dense.movedim(1, -1), "kornia", "opencv"
    ).movedim(-1, 1)

:class:`SIFTDescriptorFromPyramid` builds its own pyramid rather than reusing a
detector's Gaussian layers. It constructs histogram maps only for octaves used by
valid supplied frames, which can reduce work and memory for frames concentrated
in a few octaves. See the recorded
`octave-selection measurements <https://github.com/kornia/kornia/blob/main/benchmarks/feature/sift_summary.md#review-follow-up-descriptor-octave-selection>`_.
General affine frames use an approximate angular remapping and spatial pooling;
this is not numerically identical to patch extraction.

Its ``orient_and_describe(image, lafs)`` method additionally estimates orientation,
integrating smoothed gradients before angular binning; this is the orientation
used by ``SIFTFeature(descriptor_backend="pyramid")``. Its performance and accuracy
acceptance criteria are tracked in `#5748 <https://github.com/kornia/kornia/issues/5748>`_;
they do not concern the descriptor-only ``forward`` use on supplied oriented LAFs.
The :ref:`validation and possible deprecation plan <sift-pipelines>` retains all
three standalone descriptor interfaces.

API reference
-------------

.. autoclass:: DenseSIFTDescriptor
   :members: get_gradient_histograms

.. autoclass:: SIFTDescriptorFromPyramid
   :members: forward, orient_and_describe

.. autoclass:: SIFTDescriptor
.. autofunction:: convert_sift_descriptor_layout
.. autoclass:: MKDDescriptor
.. autoclass:: HardNet
.. autoclass:: HardNet8
.. autoclass:: HyNet
.. autoclass:: TFeat
.. autoclass:: SOSNet
.. autoclass:: LAFDescriptor
   :members: forward

.. autoclass:: SOLD2
   :members: forward

.. autofunction:: get_laf_descriptors
