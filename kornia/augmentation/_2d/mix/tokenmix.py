import torch
from .base import MixBase

class TokenMix(MixBase):
    def __init__(self, alpha=1.0, num_tokens=8, p=1.0, same_on_batch=False, keepdim=False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.alpha = alpha
        self.num_tokens = num_tokens

    def generate_parameters(self, batch_shape):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_shape[0],))
        return {"lam": lam}

    def apply_transform(self, input, params, extra_args):
        B, C, H, W = input.shape
        lam = params["lam"].to(input.device)
        idx = torch.randperm(B)
        out = input.clone()
        token_h = H // self.num_tokens
        token_w = W // self.num_tokens
        for i in range(B):
            for t in range(self.num_tokens):
                y = t * token_h
                x = t * token_w
                out[i, :, y:y+token_h, x:x+token_w] = \
                    input[idx[i], :, y:y+token_h, x:x+token_w]
        return out, idx, lam