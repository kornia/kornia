A TorchScript-scripted `SOSNet`, `TFeat` or `HyNet` now passes every convolution input through the device's patch
convolution layout as the eager model does, whatever else was scripted earlier in the process. TorchScript resolves
`isinstance(layer, nn.Conv2d)` against the first `Conv2d` class type it compiles, so a scripted `SOSNet` or `TFeat`
used to set the layout before the first convolution only, or before none once another `Conv2d` had been scripted; it
then convolved in a different layout from eager, ran without the CUDA half and MPS channels-last speed-up, and differed
from eager by up to 0.004 in float16 on CUDA.
