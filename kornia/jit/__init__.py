import torch

import kornia as K

# expose functions to torch.jit
# TODO: find an automatic way to do this
rgb_to_grayscale = torch.jit.script(K.color.rgb_to_grayscale)
bgr_to_grayscale = torch.jit.script(K.color.bgr_to_grayscale)

spatial_soft_argmax2d = torch.jit.script(K.geometry.spatial_soft_argmax2d)
spatial_softmax2d = torch.jit.script(K.geometry.dsnt.spatial_softmax2d)
spatial_expectation2d = torch.jit.script(K.geometry.dsnt.spatial_expectation2d)
render_gaussian2d = torch.jit.script(K.geometry.dsnt.render_gaussian2d)
warp_perspective = torch.jit.script(K.geometry.warp_perspective)
