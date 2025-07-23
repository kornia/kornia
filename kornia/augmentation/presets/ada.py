class AdaptiveAugmentationSequential(K.AugmentationSequential):
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
                K.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1), p=1),
                K.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1, p=1),
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


    def forward(self, *inputs, real_acc):
        assert isinstance(real_acc, (int, float)), "missing real_acc"
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