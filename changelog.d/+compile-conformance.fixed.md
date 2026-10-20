`kornia.enhance.threshold` and `kornia.enhance.Threshold` compile with `torch.compile(..., fullgraph=True)` on
torch 2.9: the supported-type check no longer iterates the `ThresholdType` enum while tracing, which torch 2.9's
Dynamo cannot do.
