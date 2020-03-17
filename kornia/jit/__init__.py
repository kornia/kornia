import torch
import kornia as K

# expose functions to torch.jit
# TODO: find an automatic way to do this
rgb_to_grayscale = torch.jit.script(K.color.rgb_to_grayscale)
