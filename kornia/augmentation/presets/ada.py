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

import kornia.augmentation as K


class AdaptiveDiscriminatorAugmentation(K.AugmentationSequential):
    r"""Implementation of Adaptive Discriminator Augmentation (ADA) for GANs training.

    adjust a global probability p over all augmentations list to select a subset of images to augment
    based on an exponential moving average of the Discriminator's accuracy labeling real samples.

    Args:
        *args: a list of kornia augmentation modules, set to a default list if not specified.

        initial_p: initial global probability `p` for applying the augmentations on

        adjustment_speed: step size for updating the global probability `p`

        max_p: the maximum value to clamp `p` at

        target_real_acc: target Discriminator accuracy to prevent overfitting

        ema_lambda: EMA smoothing factor to compute the $
        \mathrm{ema_real_accuracy} = \lambda_\text{EMA} * mathrm{real_accuracy} +
        (1 - \lambda_\text{EMA}) * mathrm{real_accuracy}
        $

        update_every: `p` update frequency

        crop_size: the used in the `RandomCrop` default augmentation

        same_on_batch: apply the same transformation across the batch.
        If None, it will not overwrite the function-wise settings.

        data_keys: the input type sequential for applying augmentations. Accepts "input", "image", "mask",
            "bbox", "bbox_xyxy", "bbox_xywh", "keypoints", "class", "label".

        **kwargs: the rest of the `kwargs` passed to the `AugmentationSequential` attribute containing augmentation

    Examples:
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> input = torch.randn(2, 3, 5, 6)
        >>> ada = AdaptiveDiscriminatorAugmentation()
        ... ...
        >>> Discriminator = ...
        >>> dataloader = ...
        >>> real_acc = None
        >>> for real_samples in dataloader:
        ...     real_samples = ada(real_samples, real_acc=real_acc)
        ...     ....
        ...     real_logits = Discriminator(real_samples)
        ...     real_acc = ...
        ...     ...

    This example demonstrates using default augmentations with AdaptiveDiscriminatorAugmentation in a GAN training loop.


        >>> import kornia.augmentation as K
        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = [
        ...     K.RandomRotation90(times=[0, 3], p=1),
        ...     K.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1), p=1),
        ...     K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=1),
        ... ]

        >>> ada = AdaptiveDiscriminatorAugmentation(*aug_list)
        >>> out = aug_list(input)

    This example demonstrates using custom augmentations with AdaptiveDiscriminatorAugmentation.
    """

    def __init__(
        self,
        *args,
        initial_p=1e-5,
        adjustment_speed=1e-2,
        max_p=0.8,
        target_real_acc=0.85,
        ema_lambda=0.99,
        update_every=5,
        crop_size=(64, 64),
        data_keys=("input",),
        same_on_batch=False,
        **kwargs,
    ):
        if not args:
            args = [
                K.RandomHorizontalFlip(p=1),
                K.RandomRotation90(times=[0, 3], p=1.0),
                K.RandomCrop(size=crop_size, padding=0.1, p=1.0),
                K.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1.0),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                K.RandomGaussianNoise(std=0.1, p=1.0),
            ]
        super().__init__(*args, data_keys=data_keys, same_on_batch=same_on_batch, **kwargs)

        self.p = initial_p
        self.adjustment_speed = adjustment_speed
        self.max_p = max_p
        self.target_real_acc = target_real_acc
        self.real_acc_ema = 0.5
        self.ema_lambda = ema_lambda
        self.update_every = update_every
        self.num_calls = -update_every  # to avoid updating in the first `update_every` steps

    def update(self, real_acc):
        r"""Updates internal params `p` once every `update_every` calls based on the `real_acc` arg by
        adding / subtracting the `adjustment_speed` from it and clamp it at [0, `max_p`]
        increment the internal counter `num_calls` by 1 on each call.

        Args:
            real_acc: the Discriminator's accuracy labeling real samples

        """
        self.num_calls += 1

        if self.num_calls < self.update_every:
            return
        self.num_calls = 0

        self.real_acc_ema = self.ema_lambda * self.real_acc_ema + (1 - self.ema_lambda) * real_acc

        if self.real_acc_ema < self.target_acc:
            self.p = max(0, self.p - self.adjustment_speed)
        else:
            self.p = min(self.p + self.adjustment_speed, self.max_p)

    def forward(self, *inputs, real_acc=None):
        r"""Apply augmentations to a subset of input tensors with global probability `p`.

        This method applies the augmentation pipeline to a subset of input samples, selected stochastically
        using a Bernoulli distribution with probability $p :math:`P_i \\sim \\text{Bernoulli}(p)`,
        where :math:`P_i` is a boolean mask for the :math:`i`-th
        if `real_acc` is provided, the internal probability `p` is updated via the `update` method.
        Non-augmented samples retain their original values, and the output matches the input structure.
        """
        if real_acc is not None:
            self.update(real_acc)

        if self.p == 0:
            return inputs

        P = torch.bernoulli(torch.full((inputs[0].size(0),), self.p, device=inputs[0].device)).bool()

        if not P.any():
            return inputs if len(inputs) > 1 else inputs[0]

        selected_inputs = tuple(inputs[P] for input_ in inputs) if len(inputs) > 1 else inputs[0][P]
        augmented_inputs = super().forward(selected_inputs)

        if len(inputs) > 1:
            outputs = []
            for input_ in inputs:
                output_ = input_.clone()
                output_[P] = augmented_inputs[inputs.index(input_)]
                outputs.append(output_)
            return tuple(outputs)

        outputs = inputs[0].clone()
        outputs[P] = augmented_inputs
        return outputs
