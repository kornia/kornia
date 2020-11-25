Differentiable Data Augmentation
================================

Kornia DDA module leverages differentiable computer vision solutions from Kornia, with an aim of integrating data augmentation (DA) pipelines and strategies to existing PyTorch components (e.g. autograd for differentiability, optim for optimization).


Features
========


Augmentation as a Layer
-----------------------

Any augmentations can be used as a normal layer in PyTorch like Conv2D, AvgPool2D, etc.

1) Use it in nn.Sequential.

    .. code-block:: python

        import kornia.augmentation as K
        import torch.nn as nn

        transform = nn.Sequential(
            K.RandomAffine(360, p=0.5),
            K.ColorJitter(0.2, 0.3, 0.2, 0.3, p=0.5)
        )

        transform(torch.randn(16, 3, 224, 224))

2) Use it in nn.Module.

    .. code-block:: python

        import kornia.augmentation as K
        import torch.nn as nn

        class MyAugmentationPipeline(nn.Module):

            def __init__(self):
                super(MyAugmentationPipeline, self).__init__()
                self.aff = K.RandomAffine(360, p=0.5)
                self.jitter = K.ColorJitter(0.2, 0.3, 0.2, 0.3, p=0.5)

            def forward(self, input):
                input = self.jitter(self.aff(input))
                return input

        aug = MyAugmentationPipeline()
        aug(torch.randn(16, 3, 224, 224))


Reproducibilities
-----------------

1) Reproducible computations accross devices.

    .. code-block:: python

        # will happen in device cpu
        augmented_1 = aug(images.to('cpu'))

        # will happen in device cuda:0
        augmented_2 = aug(images.to('cuda:0'))

        assert_allclose(augmented_1, augmented_2)

2) Save and load your augmentations.

    .. code-block:: python

        # Save your augmentations
        torch.save(aug, "./saved_da.pt")

        # Load your augmentations
        aug_restored = torch.load("./saved_da.pt")

3) Export to ONNX.

    .. code-block:: python

        import onnx
        import torch.onnx

        # Input to the augmentation pipeline
        input = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
        output = torch_model(x)

        # Export the augmentation pipeline
        torch.onnx.export(
            aug,                             # model being run
            input,                           # model input (or a tuple for multiple inputs)
            "MyAugmentationPipeline.onnx",   # where to save the model (can be a file or file-like object)
            export_params=True,              # store the trained parameter weights inside the model file
            opset_version=1,                 # the ONNX version to export the model to
            do_constant_folding=True,        # whether to execute constant folding for optimization
            input_names = ['input'],         # the model's input names
            output_names = ['output'],       # the model's output names
            dynamic_axes={                   # variable lenght axes
                'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}})
        
        # Load the augmentation pipeline
        onnx_model = onnx.load("MyAugmentationPipeline.onnx")
        onnx.checker.check_model(onnx_model)


PyTorch-Backended Optimization
------------------------------

Our framework provides an easy and intuitive solution to backpropagate the gradients through augmentation layers using the native PyTorch workflow. In any augmentations, kornia.augmentation takes nn.Parameter as differentiable parameters while torch.tensor as static parameters. The following example shows how to optimize the differentiable parameters (including brightness, contrast, saturation) of kornia.augmentation.ColorJitter and backpropagate the gradients based on the computed error from a loss function.

.. code-block:: python

    import kornia.augmentation as K
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Make the input image optimizable
    images = torch.tensor(img, requires_grad=True)

    # Define learnable ColorJitter, which having brightness, contrast, saturation learnable and a static hue.
    jitter = K.ColorJitter(
        nn.Parameter(torch.tensor([0.8, 0.8])),
        nn.Parameter(torch.tensor([0.7, 0.7])),
        nn.Parameter(torch.tensor([0.6, 0.6])),
        torch.tensor([0.1, 0.1])
    )

    # Define optimizers
    optimizer_img = torch.optim.SGD([images], lr=1e+5) # Large lr for demo
    optimizer_param = torch.optim.SGD(jitter.parameters(), lr=0.1)

    # Forward
    out = jitter(images)

    # Loss computation
    loss = nn.MSELoss()(out, images)

    # Update
    loss.backward()
    optimizer_img.step()
    optimizer_param.step()

The Updated results as follows.
.. code-block:: bash

    brightness -> [0.8048, 0.8363]     contrast -> [0.7030, 0.7323]
    saturation -> [0.5999, 0.5976]     hue -> [0.1000, 0.1000]

.. image:: https://github.com/kornia/kornia/raw/master/docs/source/_static/img/dda_example.png

From left to right: the original input, augmented image and gradient-updated image.


Customization
-------------

Kornia provides useful 2D and 3D augmentation base classes for an easier customization of your new augmenatation ideas. In general, all augmentations shall inherit from either ``AugmentationBase2D``, ``AugmentationBase3D`` or ``AugmentationBaseMix``. Those base classes would handle 1) forward/backward operations, 2) which images to apply the augmentation in a batch, 3) the device and dtype for random numbers, 4) if to compute the transformation matrices. You shall only need to implement 4 intuitive functions:

    a. **__init__**: To define the learnable or static parameters.
    b. **generate_parameters**: The function to generate the augmentation parameters, that returns a dict with {key: tensor} paradigm.  Note that the random states are **NOT** reproducible across devices.
    c. **compute_transformation**: Compute the corresponding transformation according to the provided parameters. For geometric transformations, it shall return the transformation matrix. Otherwise, it shall return an identity matrix.
    d. **apply_transform**: Compute the augmentation output.

- The following code is a short example of a customized augmentation:

.. code-block:: python

   import torch
   import kornia as K

   from kornia.augmentation import AugmentationBase2D

   class MyRandomTransform(AugmentationBase2D):
        r"""Perform MyRandomTransform to image.

        Args:
            p (float): Probability to equalize an image. Default value is 0.5.
            ...

        Shape:
            - Input: :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
            - Output: :math:`(B, C, H, W)`

        Examples:
            >>> rng = torch.manual_seed(0)
            >>> input = torch.rand(1, 1, 5, 5)
            >>> trans = MyRandomTransform(p=1.)
            >>> trans(input)
            ...
        """

        def __init__(self, same_on_batch: bool = False, return_transform: bool = False, p: float = 0.5) -> None:
            super(MyRandomTransform, self).__init__(
                p=p, return_transform=return_transform, same_on_batch=same_on_batch)

        def generate_parameters(self, input_shape: torch.Size):
            # generate the random parameters for your use case.
            pi = torch.as_tensor(K.pi, device=self.device, dtype=self.dtype)
            angles_rad: torch.Tensor = torch.rand(input_shape[0], device=self.device, dtype=self.dtype) * pi
            angles_deg = kornia.rad2deg(angles_rad) 
            return dict(angles=angles_deg)
      
        def compute_transformation(self, input, params):

            B, _, H, W = input.shape

            # compute transformation
            angles: torch.Tensor = params['angles'].type_as(input)
            center = torch.tensor([[W / 2, H / 2]] * B).type_as(input)
            transform = K.get_rotation_matrix2d(center, angles, torch.ones_like(angles))
            return transform

        def apply_transform(self, input, params):
            _, _, H, W = input.shape
            # compute transformation
            transform = self.compute_transformation(input, params)

            # apply transformation and return
            output = K.warp_affine(input, transform, (H, W))
            return output


Supported Operations
====================

+--------------------------------------------+------------------------------------------+
|  Geometric Augmentations                   |   Color-space Augmentations              |
+==========================+========+========+=========================+========+=======+
|                          | ``2D`` | ``3D`` |                         | ``2D`` | ``3D``|
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomHorizontalFlip     | ✔      | ✔      |ColorJitter              | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomVerticalFlip       | ✔      | ✔      |RandomGrayscale          | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomDepthicalFlip      | ✔      | ✔      |RandomSolarize           | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomRotation           | ✔      | ✔      |RandomPosterize          | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomAffine             | ✔      | ✔      |RandomSharpness          | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomPerspective        | ✔      | ✔      |RandomEqualize           | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomErasing            | ✔      | ✘      |                                          |
+--------------------------+--------+--------+------------------------------------------+
| CenterCrop               | ✔      | ✔      |      **Mix Augmentations**               |
+--------------------------+--------+--------+------------------------------------------+
| RandomCrop               | ✔      | ✔      |                                          |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomResizedCrop        | ✔      | ✔      |RandomMixUp              | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+
| RandomMotionBlur         | ✔      | ✔      |RandomCutMix             | ✔      | ✘     |
+--------------------------+--------+--------+-------------------------+--------+-------+


Cite
====

If you find this library useful for your research, please consider citing `Differentiable Data Augmentation with Kornia <https://arxiv.org/pdf/2011.09832.pdf>`_

.. code:: bash

    @misc{2011.09832,
        Author = {Jian Shi and Edgar Riba and Dmytro Mishkin and Francesc Moreno and Anguelos Nicolaou},
        Title = {Differentiable Data Augmentation with Kornia},
        howpublished = {Workshop on Differentiable Vision, Graphics, and Physics in Machine Learning at NeurIPS 2020},
        Year = {2020},
        url = {https://arxiv.org/pdf/2011.09832.pdf}
    }
