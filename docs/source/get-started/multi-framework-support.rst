.. raw:: html

   <a href="https://github.com/ivy-llc/ivy" target="_blank">
       <div style="display: block;" align="center">
           <img class="dark-light" width="30%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg"/>
       </div>
   </a>
   <br>

Multi-Framework Support
=======================

Kornia can now be used with `TensorFlow <https://www.tensorflow.org/>`_, `JAX <https://jax.readthedocs.io/en/latest/index.html>`_,
and `Numpy <https://numpy.org/>`_ thanks to an integration with `Ivy <https://github.com/ivy-llc/ivy>`_. 

This can be accomplished using the following functions, which are now part of the Kornia api:

* :code:`kornia.to_tensorflow()`

* :code:`kornia.to_jax()`

* :code:`kornia.to_numpy()`

Here's an example of using kornia with TensorFlow:

.. code:: python

    import kornia
    import tensorflow as tf

    tf_kornia = kornia.to_tensorflow()

    rgb_image = tf.random.normal((1, 3, 224, 224))
    gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)

So what's happening here? Let's break it down.

#. Transpiling kornia to TensorFlow

    This line lazily transpiles everything in the kornia api to TensorFlow, and creates a new module for this transpiled version of kornia.
    Because the transpilation happens lazily, no function or class will be transpiled until it's actually called.

    .. code-block:: python

        tf_kornia = kornia.to_tensorflow()

#. Calling a TF kornia function

    We can now call any kornia function (or class) with TF arguments. However, this function will be very slow relative to
    the original function - as the function is being transpiled during this step.

    .. code-block:: python

        rgb_image = tf.random.normal((1, 3, 224, 224))
        gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)  # slow

#. Subsequent function calls

    The good news is any calls of the function after the initial call will be much faster, as it has already been transpiled, 
    and should approximately match the speed of the original kornia function.

    .. code-block:: python

        gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)  # fast

#. Transpilations in different Python sessions

    You may be wondering if you'll have to wait for these long initial transpilations to take place each time you start a
    new Python session? The good news is that when a transpilation occurs, Ivy will save the generated source code in the
    local directory, so if the same transpilation is ever attempted again from within the same directory, it will be
    immediately retrieved and used.


Kornia can be used with JAX and NumPy in the same way:

.. code:: python

    import kornia
    import numpy as np

    np_kornia = kornia.to_numpy()

    rgb_image = np.random.normal(size=(1, 3, 224, 224))
    gray_image = np_kornia.color.rgb_to_grayscale(rgb_image)


.. code:: python

    import kornia
    import jax

    jax_kornia = kornia.to_jax()

    rgb_image = jax.random.normal(jax.random.key(42), shape=(1, 3, 224, 224))
    gray_image = jax_kornia.color.rgb_to_grayscale(rgb_image)


Limitations
-----------

* Converting Kornia to TensorFlow or JAX works for functions, classes and trainable modules; converting to NumPy supports functions and classes, but not trainable modules.

* Transpilation does not currently work with custom kernels, such as flash attention.

* Certain stateful classes cannot currently be transpiled, such as optimizers (torch.optim.Adam, etc.), trainers, and data loaders.

* Compatibility with native compilers (*jax.jit* and *tf.function*) is somewhat limited with transpiled versions of Kornia,
  particularly compared with *torch.compile* on standard Kornia. Improving compatibility with these is one of the key areas of
  focus for the current development of Ivy.


From the Ivy Team
-----------------

We hope you find using Kornia with TensorFlow, JAX and NumPy useful! Ivy is still very much under development, 
so if you find any issues/bugs, feel free to raise an issue on the `ivy <https://github.com/ivy-llc/ivy>`_ repository.
We'd also really appreciate a star, if you'd like to show your support!

To learn more about Ivy, we recommend taking a look through our `documentation <https://ivy.dev/docs/>`_.
