`kornia.enhance.threshold` and `kornia.enhance.Threshold` compile with `torch.compile(..., fullgraph=True)` on
torch 2.9 and earlier, where they failed: the supported-type check no longer iterates the `ThresholdType` enum
while tracing, which Dynamo there cannot do.
