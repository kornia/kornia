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

import torch


def analyze_model_behavior(model, image, augmentation, layers=("layer1",)):
    """Analyze how model internal representations change under augmentations.

    This utility applies an augmentation to an input image and compares
    intermediate feature maps across specified layers.

    Args:
        model: PyTorch model
        image: Input tensor of shape (B, C, H, W)
        augmentation: Kornia augmentation module
        layers: Tuple of layer names to analyze

    Returns:
        dict: Mapping layer name → mean absolute difference
    """
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output

        return hook

    handles = []

    # Register hooks
    for layer_name in layers:
        layer = dict([*model.named_modules()])[layer_name]
        handles.append(layer.register_forward_hook(hook_fn(layer_name)))

    with torch.no_grad():
        # Original pass
        features.clear()
        _ = model(image)
        orig_feats = {k: v.clone() for k, v in features.items()}

        # Augmented pass
        features.clear()
        aug_image = augmentation(image)
        _ = model(aug_image)
        aug_feats = {k: v.clone() for k, v in features.items()}

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute differences
    diffs = {}
    for k, v in orig_feats.items():
        diffs[k] = (v - aug_feats[k]).abs().mean().item()

    return diffs
