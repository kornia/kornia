Image Denoising
===============

Image denoising removes noise from an image while preserving its structure. Because every Kornia operator is
differentiable, a classic variational approach such as total-variation denoising can be written as a few lines of
optimization with :func:`kornia.losses.total_variation`:

.. code:: python

    import torch
    import kornia

    noisy = torch.rand(1, 3, 64, 64)
    denoised = noisy.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([denoised], lr=0.01)

    for _ in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(denoised, noisy) + 1e-4 * kornia.losses.total_variation(denoised).mean()
        loss.backward()
        optimizer.step()

Follow the full walk-through in the `total variation denoising tutorial <https://kornia.github.io/tutorials/nbs/total_variation_denoising.html>`_.
