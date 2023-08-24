Image Registration
==================

Image registration is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, data from different sensors, times, depths, or viewpoints. It is used in computer vision, medical imaging, and compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from these different measurements.

Learn more: `https://paperswithcode.com/task/image-registration <https://paperswithcode.com/task/image-registration>`_

..  youtube:: Re1q6vRfZac

We provide the :py:class:`~kornia.geometry.transform.image_registrator.ImageRegistrator` API from which you can leverage to align automatically two images by a process of direct optimisation using the PyTorch Autograd differentiability.

.. code:: python

    from kornia.geometry import ImageRegistrator
    img_src = torch.rand(1, 1, 32, 32)
    img_dst = torch.rand(1, 1, 32, 32)
    registrator = ImageRegistrator('similarity')
    homo = registrator.register(img_src, img_dst)

Then, if you want to perform a more sophisticated process:

.. literalinclude:: ../_static/image_registration.py

To reproduce the same results as in the showed video you can go through or full tutorial using Colab found `here <https://kornia.github.io/tutorials/nbs/image_registration.html>`_ .

Interactive Demo
----------------
.. raw:: html

   <gradio-app space="kornia/image-registration-with-kornia"></gradio-app>


Visit the `image registration demo on the Hugging Face Spaces <https://huggingface.co/spaces/kornia/image-registration-with-kornia>`_.
