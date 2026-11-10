# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Build image, video and volume augmentation pipelines with recorded random parameters.

Transforms take float tensors, ``(B, C, H, W)`` images or ``(B, C, D, H, W)`` volumes; color transforms expect
values in ``[0, 1]``. ``AugmentationSequential`` applies one draw to images, masks, boxes and keypoints together,
and ``ImageSequential`` chains image-only transforms. In eager mode, ``aug._params`` holds the last draw, and
``aug(x, params=aug._params)`` replays it. Inverting a geometric warp cannot restore pixels lost to cropping or
resampling.

Which class to use:

Pipelines and policies:
    - ``AugmentationSequential``: coordinate image, mask, box and keypoint handlers.
    - ``ImageSequential``: chain image transforms, including ordinary PyTorch modules.
    - ``VideoSequential``: apply 2D transforms to video frames, optionally sharing a draw within each clip.
    - ``PatchSequential``: split an image into a grid and augment its patches.
    - ``ManyToManyAugmentationDispather``: send each input bundle to its corresponding pipeline.
    - ``ManyToOneAugmentationDispather``: produce several augmented views of one input bundle.
    - ``auto``: AutoAugment, RandAugment, TrivialAugment and their policy operations.
    - ``container``: pipeline classes and their parameter records.

Image size and geometry:
    - ``Resize``: resize to a target size; ``LongestMaxSize`` and ``SmallestMaxSize`` select the controlling side.
    - ``PadTo``: pad to a target size at the right and bottom, cropping oversized inputs by default.
    - ``CenterCrop``: take a centered crop; ``RandomCrop`` samples its position.
    - ``RandomResizedCrop``: sample a crop's area and aspect ratio, then resize it.
    - ``RandomHorizontalFlip`` and ``RandomVerticalFlip``: reverse columns or rows.
    - ``RandomAffine``: combine rotation, translation, scale and shear in one warp.
    - ``RandomRotation``: rotate by a sampled angle; ``RandomRotation90`` samples multiples of 90 degrees.
    - ``RandomShear`` and ``RandomTranslate``: sample only shear or translation.
    - ``RandomPerspective``: perturb the four image corners for a projective warp.
    - ``RandomElasticTransform``: warp with a smoothed random displacement field.
    - ``RandomThinPlateSpline``: warp with displaced control points.
    - ``RandomFisheye``: apply a radial distortion around a sampled center.

Color and contrast:
    - ``ColorJiggle``: combine additive brightness, multiplicative contrast and HSV saturation/hue adjustments.
    - ``ColorJitter``: use torchvision-compatible brightness, contrast and saturation formulas.
    - ``RandomBrightness`` and ``RandomContrast``: sample one brightness or contrast factor.
    - ``RandomHue`` and ``RandomSaturation``: change one HSV color component.
    - ``RandomGamma``: apply power-law intensity correction.
    - ``RandomGrayscale``: average channels with RGB luminance weights or equal weights, keeping the channel count.
    - ``RandomRGBShift``: shift the three RGB channels independently.
    - ``RandomPlanckianJitter``: vary RGB illumination along a color-temperature curve.
    - ``RandomChannelShuffle``: permute channels; ``RandomChannelDropout`` replaces selected channels.
    - ``RandomAutoContrast``: stretch each channel's intensity range.
    - ``RandomEqualize``: equalize a global histogram; ``RandomClahe`` uses local, contrast-limited histograms.
    - ``RandomInvert``: subtract values from a maximum; ``RandomSolarize`` inverts values above a threshold.
    - ``RandomPosterize``: reduce the number of retained intensity bits.
    - ``Normalize`` and ``Denormalize``: apply or undo channel-wise mean and standard-deviation normalization.

Blur, noise and occlusion:
    - ``RandomBoxBlur``: average a rectangular window; ``RandomGaussianBlur`` uses Gaussian weights.
    - ``RandomMedianBlur``: replace pixels with local medians.
    - ``RandomMotionBlur``: convolve with a sampled directional motion kernel.
    - ``RandomSharpness``: blend an image with its smoothed version.
    - ``RandomGaussianNoise``: add Gaussian noise; ``RandomSaltAndPepperNoise`` inserts dark and bright pixels.
    - ``RandomJPEG``: simulate JPEG compression at a sampled quality.
    - ``RandomErasing``: fill a sampled rectangle with a constant.
    - ``RandomDissolving``: apply a pretrained Stable Diffusion dissolving model.
    - ``RandomRain`` and ``RandomSnow``: synthesize rain streaks or a snow effect.

Spatially varying illumination:
    - ``RandomGaussianIllumination``: add a Gaussian illumination pattern.
    - ``RandomLinearIllumination``: add an edge-to-edge gradient.
    - ``RandomLinearCornerIllumination``: add a gradient starting from a corner.
    - ``RandomPlasmaBrightness``: shift brightness with a fractal map.
    - ``RandomPlasmaContrast``: modulate contrast with a fractal map.
    - ``RandomPlasmaShadow``: darken regions selected by a fractal map.

Mixing and rearranging images:
    - ``RandomMixUpV2``: blend paired images and return label weights when requested.
    - ``RandomCutMixV2``: copy sampled rectangles from paired images.
    - ``PatchMix``: copy a fixed-size square from a paired image.
    - ``RandomMosaic``: assemble tiles from several batch images.
    - ``RandomJigsaw``: permute an image's grid patches.
    - ``RandomTransplantation``: copy regions selected by a donor segmentation mask.

Volumes:
    - ``CenterCrop3D`` and ``RandomCrop3D``: take centered or randomly positioned volume crops.
    - ``RandomHorizontalFlip3D``, ``RandomVerticalFlip3D`` and ``RandomDepthicalFlip3D``: reverse width, height or
      depth.
    - ``RandomAffine3D``: sample a 3D affine warp; ``RandomRotation3D`` samples rotation alone.
    - ``RandomPerspective3D``: perturb the volume corners for a projective warp.
    - ``RandomEqualize3D``: equalize volume intensities; ``RandomMotionBlur3D`` applies a 3D motion kernel.
    - ``RandomTransplantation3D``: copy regions selected by a donor volume's segmentation mask.

Extending the module:
    - ``AugmentationBase2D`` and ``AugmentationBase3D``: share shape handling, sampling and application gates.
    - ``RigidAffineAugmentationBase2D`` and ``RigidAffineAugmentationBase3D``: add transformation-matrix tracking.
    - ``GeometricAugmentationBase2D`` and ``GeometricAugmentationBase3D``: implement spatial transforms.
    - ``IntensityAugmentationBase2D`` and ``IntensityAugmentationBase3D``: implement pixel-value transforms.
    - ``MixAugmentationBaseV2``: implement transforms that combine batch samples and dispatch data keys.
"""

# Lazy loading auto module
from kornia.augmentation import auto, container
from kornia.augmentation._2d import (
    CenterCrop,
    ColorJiggle,
    ColorJitter,
    Denormalize,
    LongestMaxSize,
    Normalize,
    PadTo,
    PatchMix,
    RandomAffine,
    RandomAutoContrast,
    RandomBoxBlur,
    RandomBrightness,
    RandomChannelDropout,
    RandomChannelShuffle,
    RandomClahe,
    RandomContrast,
    RandomCrop,
    RandomCutMixV2,
    RandomDissolving,
    RandomElasticTransform,
    RandomEqualize,
    RandomErasing,
    RandomFisheye,
    RandomGamma,
    RandomGaussianBlur,
    RandomGaussianIllumination,
    RandomGaussianNoise,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomHue,
    RandomInvert,
    RandomJigsaw,
    RandomJPEG,
    RandomLinearCornerIllumination,
    RandomLinearIllumination,
    RandomMedianBlur,
    RandomMixUpV2,
    RandomMosaic,
    RandomMotionBlur,
    RandomPerspective,
    RandomPlanckianJitter,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomPosterize,
    RandomRain,
    RandomResizedCrop,
    RandomRGBShift,
    RandomRotation,
    RandomRotation90,
    RandomSaltAndPepperNoise,
    RandomSaturation,
    RandomSharpness,
    RandomShear,
    RandomSnow,
    RandomSolarize,
    RandomThinPlateSpline,
    RandomTranslate,
    RandomTransplantation,
    RandomVerticalFlip,
    Resize,
    SmallestMaxSize,
)
from kornia.augmentation._2d.base import AugmentationBase2D, RigidAffineAugmentationBase2D
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.augmentation._3d import (
    CenterCrop3D,
    RandomAffine3D,
    RandomCrop3D,
    RandomDepthicalFlip3D,
    RandomEqualize3D,
    RandomHorizontalFlip3D,
    RandomMotionBlur3D,
    RandomPerspective3D,
    RandomRotation3D,
    RandomTransplantation3D,
    RandomVerticalFlip3D,
)
from kornia.augmentation._3d.base import AugmentationBase3D, RigidAffineAugmentationBase3D
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.augmentation._3d.intensity.base import IntensityAugmentationBase3D
from kornia.augmentation.container import (
    AugmentationSequential,
    ImageSequential,
    ManyToManyAugmentationDispather,
    ManyToOneAugmentationDispather,
    PatchSequential,
    VideoSequential,
)

__all__ = [
    "AugmentationBase2D",
    "AugmentationBase3D",
    "AugmentationSequential",
    "CenterCrop",
    "CenterCrop3D",
    "ColorJiggle",
    "ColorJitter",
    "Denormalize",
    "GeometricAugmentationBase2D",
    "GeometricAugmentationBase3D",
    "ImageSequential",
    "IntensityAugmentationBase2D",
    "IntensityAugmentationBase3D",
    "LongestMaxSize",
    "ManyToManyAugmentationDispather",
    "ManyToOneAugmentationDispather",
    "MixAugmentationBaseV2",
    "Normalize",
    "PadTo",
    "PatchMix",
    "PatchSequential",
    "RandomAffine",
    "RandomAffine3D",
    "RandomAutoContrast",
    "RandomBoxBlur",
    "RandomBrightness",
    "RandomChannelDropout",
    "RandomChannelShuffle",
    "RandomClahe",
    "RandomContrast",
    "RandomCrop",
    "RandomCrop3D",
    "RandomCutMixV2",
    "RandomDepthicalFlip3D",
    "RandomDissolving",
    "RandomElasticTransform",
    "RandomEqualize",
    "RandomEqualize3D",
    "RandomErasing",
    "RandomFisheye",
    "RandomGamma",
    "RandomGaussianBlur",
    "RandomGaussianIllumination",
    "RandomGaussianNoise",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomHorizontalFlip3D",
    "RandomHue",
    "RandomInvert",
    "RandomJPEG",
    "RandomJigsaw",
    "RandomLinearCornerIllumination",
    "RandomLinearIllumination",
    "RandomMedianBlur",
    "RandomMixUpV2",
    "RandomMosaic",
    "RandomMotionBlur",
    "RandomMotionBlur3D",
    "RandomPerspective",
    "RandomPerspective3D",
    "RandomPlanckianJitter",
    "RandomPlasmaBrightness",
    "RandomPlasmaContrast",
    "RandomPlasmaShadow",
    "RandomPosterize",
    "RandomRGBShift",
    "RandomRain",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomRotation3D",
    "RandomRotation90",
    "RandomSaltAndPepperNoise",
    "RandomSaturation",
    "RandomSharpness",
    "RandomShear",
    "RandomSnow",
    "RandomSolarize",
    "RandomThinPlateSpline",
    "RandomTranslate",
    "RandomTransplantation",
    "RandomTransplantation3D",
    "RandomVerticalFlip",
    "RandomVerticalFlip3D",
    "Resize",
    "RigidAffineAugmentationBase2D",
    "RigidAffineAugmentationBase3D",
    "SmallestMaxSize",
    "VideoSequential",
    "auto",
    "container",
]
