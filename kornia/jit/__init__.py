import torch
import kornia as K

# expose functions to torch.jit
# TODO: find an automatic way to do this
rgb_to_grayscale = torch.jit.script(K.color.rgb_to_grayscale)

spatial_soft_argmax2d = torch.jit.script(K.geometry.spatial_soft_argmax2d)
spatial_softmax_2d = torch.jit.script(K.geometry.dsnt.spatial_softmax_2d)
spatial_expectation_2d = torch.jit.script(K.geometry.dsnt.spatial_expectation_2d)
