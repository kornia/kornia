# Kornia Differentiable Data Augmentation

## Supported Operations

<table>
<tr>
    <th>Geometric Augmentations</th>
    <th>Color-space Augmentations</th>
</tr>
<tr>
<td>

|  |     2D      |      3D      |
| ------------ | ----------- | ------------ |
| RandomHorizontalFlip | ✅ | ✅|
| RandomVerticalFlip | ✅ | ✅ |
| RandomDepthicalFlip | - | ✅ |
| RandomRotation | ✅ | ✅ |
| RandomAffine | ✅ | ✅ |
| RandomPerspective | ✅ | ✅ |
| RandomErasing | ✅ | ❌ |
| CenterCrop | ✅ | ✅ |
| RandomCrop | ✅ | ✅ |
| RandomResizedCrop | ✅ | - |
| RandomMotionBlur | ✅ | ✅ |

</td>
<td style="vertical-align:top;">

|  |     2D      |      3D      |
| ------------ | ----------- | ------------ |
| ColorJiggle | ✅ | ❌ |
| RandomGrayscale | ✅ | ❌ |
| RandomSolarize | ✅ | ❌ |
| RandomPosterize | ✅ | ❌ |
| RandomSharpness | ✅ | ❌ |
| RandomEqualize | ✅ | ✅ |

<div style="text-align:center; padding-top:3.3em;">
    <b>Mix Augmentations</b>
</div>

|  |     2D      |      3D      |
| ------------ | ----------- | ------------ |
| RandomMixUp | ✅ | ❌ |
| RandomCutMix &nbsp; &nbsp;  &nbsp;  | ✅ | ❌ |
</td>
</tr>
</table>
