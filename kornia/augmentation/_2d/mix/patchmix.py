import torch
from .base import MixBase

class PatchMix(MixBase):
    def __init__(self, alpha=1.0, patch_size=16, p=1.0, same_on_batch=False, keepdim=False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.alpha = alpha
        self.patch_size = patch_size

    def generate_parameters(self, batch_shape):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_shape[0],))
        return {"lam": lam}

    def apply_transform(self, input, params, extra_args):
        B, C, H, W = input.shape
        lam = params["lam"].to(input.device)
        idx = torch.randperm(B)
        out = input.clone()
        for i in range(B):
            # Random patch coordinates
            y = torch.randint(0, H - self.patch_size + 1, (1,))
            x = torch.randint(0, W - self.patch_size + 1, (1,))
            out[i, :, y:y+self.patch_size, x:x+self.patch_size] = \
                input[idx[i], :, y:y+self.patch_size, x:x+self.patch_size]
        return out, idx, lam