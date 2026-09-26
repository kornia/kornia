`RandomClahe` and the per-sample CLAHE path now compute their clip thresholds correctly when the clip tensor is on MPS
(#4970). The limits were moved to the CPU in float64 with a single `.to("cpu", torch.float64)`, which returns zeros
on torch 2.14, so every tile histogram was clipped at one count (a mean error of 0.16 on a 256x256 image), and raises
`TypeError` on torch 2.5.1. Parameters drawn by the default generator stay on the CPU and were not affected.
