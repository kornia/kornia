![Kornia Banner](https://github.com/kornia/data/raw/main/kornia_banner_pixie.png)

State-of-the-art and curated Computer Vision algorithms for AI.

Kornia AI is on the mission to leverage and democratize the next generation of Computer Vision tools and Deep Learning libraries
within the context of an Open Source community.

```python
>>> import kornia.geometry as K
>>> registrator = K.ImageRegistrator('similarity')
>>> model = registrator.register(img1, img2)
```

Ready to use with state-of-the art Deep Learning models:

**DexiNed edge detection model.**

```python
image = kornia.utils.sample.get_sample_images()[0][None]
model = DexiNedBuilder.build()
model.save(image)
```

**RTDETRDetector for object detection.**

```python
image = kornia.utils.sample.get_sample_images()[0][None]
model = RTDETRDetectorBuilder.build()
model.save(image)
```

**BoxMotTracker for object tracking.**

```python
import kornia
image = kornia.utils.sample.get_sample_images()[0][None]
model = BoxMotTracker()
for i in range(4):
   model.update(image)
model.save(image)
```

**Vision Transformer for image classification.**

```python
>>> import torch.nn as nn
>>> from kornia.models.vit import VisionTransformer
>>> classifier = nn.Sequential(
...   VisionTransformer(image_size=224, patch_size=16),
...   nn.Linear(768, 1000),  # Example: 768 is the default hidden_dim, 1000 is num_classes
... )
>>> logits = classifier(img)    # BxN
>>> scores = logits.argmax(-1)  # B
```

## Multi-framework support

You can now use Kornia with [NumPy](https://numpy.org/), [TensorFlow](https://www.tensorflow.org/), and [JAX](https://jax.readthedocs.io/en/latest/index.html).

```python
>>> import kornia
>>> tf_kornia = kornia.to_tensorflow()
```

<div align="center">
    <p>Powered by</p>
    <a href="https://github.com/ivy-llc/ivy" target="_blank">
        <img width="15%" src="https://raw.githubusercontent.com/ivy-llc/assets/refs/heads/main/assets/logos/ivy-long.svg" alt="Ivy"/>
    </a>
</div>

## Join the community

- Join our social network communities with 1.8k+ members:
  - [Twitter](https://twitter.com/kornia_foss): we share the recent research and news for out mainstream community.
  - [Slack](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA): come to us and chat with our engineers and mentors to get support and resolve your questions.
- Subscribe to our [YouTube channel](https://www.youtube.com/channel/UCI1SE1Ij2Fast5BSKxoa7Ag) to get the latest video demos.
