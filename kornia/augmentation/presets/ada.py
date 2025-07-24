from kornia.augmentation.container import AugmentationSequential

class AdaptiveDiscriminatorAugmentation(K.AugmentationSequential):
    r"""
    Implementation of Adaptive Discriminator Augmentation (ADA) for GANs training

    adjust a global probability p over all augmentations list to select a subset of images to augment
    based on an exponential moving average of the Discriminator's accuracy labeling real samples

    Args:
        *args: a list of kornia augmentation modules, set to a default list if not specified.

        initial_p: initial global probability `p` for applying the augmentations on 

        adjustment_speed: step size for updating the global probability `p`

        max_p: the maximum value to clamp `p` at

        target_real_acc: target `discriminator` accuracy to prevent overfitting

        ema_lambda: EMA smoothing factor to compute the $ \mathrm{ema_real_accuracy} = \lambda_\text{EMA} * mathrm{real_accuracy} + (1 - \lambda_\text{EMA}) * mathrm{real_accuracy}
        
        update_every: `p` update frequency

        same_on_batch: apply the same transformation across the batch. If None, it will not overwrite the function-wise settings.

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

    This example demonstrates using default augmentations with AdaptiveDiscriminatorAugmentation in a GAN training loop


        >>> from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = [
        ...     K.RandomRotation90(times=[0, 3], p=1),
        ...     K.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1), p=1),
        ...     K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=1),
        ... ]

        >>> ada = AdaptiveDiscriminatorAugmentation(*aug_list)
        >>> out = aug_list(input)

    This example demonstrates using custom augmentations with AdaptiveDiscriminatorAugmentation

    """
    def __init__(
        self, 
        *args, 
        initial_p=1e-5,
        adjustment_speed=1e-2, 
        max_p=.8,
        target_real_acc=0.85,
        ema_lambda=.99,
        update_every=5,
        data_keys=["input"],
        same_on_batch=False,
        **kwargs
    ):
        
        if not args:
            args = [
                K.RandomHorizontalFlip(p=1),
                K.RandomRotation90(times=[0, 3], p=1),
                K.RandomCrop(padding=0.1, p=1.0),
                K.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1), p=1),
                K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=1),
                K.RandomGaussianNoise(std=0.1, p=1.0),
            ]
        super().__init__(
            *args,
            data_keys=data_keys,
            same_on_batch=same_on_batch,
            **kwargs
        )
        
        self.p = initial_p
        self.adjustment_speed = adjustment_speed
        self.max_p = max_p
        self.target_real_acc = target_real_acc
        self.real_acc_ema = 0.5
        self.ema_lambda = ema_lambda
        self.update_every = update_every
        self.num_calls = - update_every # to avoid updating in the first `update_every` steps
    

    def update(self, real_acc):
        self.num_calls += 1
        
        if self.num_calls < self.update_every:
            return
        self.num_calls = 0        

        self.real_acc_ema = self.ema_lambda * self.real_acc_ema +\
                            (1 - self.ema_lambda) * real_acc
        
        if self.real_acc_ema < self.target_acc:
            self.p = max(0, self.p - self.adjustment_speed)
        else:
            self.p = min(self.p + self.adjustment_speed, self.max_p)


    def forward(self, *inputs, real_acc=None):
        if real_acc is not None:
            self.update(real_acc)

        if self.p == 0:
            return inputs

        P = torch.bernoulli(
            torch.full(
                (inputs[0].size(0),),
                self.p,
                device=inputs[0].device
            )
        ).bool()

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