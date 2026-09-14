An **ONNX, torch.compile and torch.export support** page under Resources (`get-started/export-support`)
lists every surveyed public operator, grouped by package and section, with its dynamo ONNX export,
`torch.export` and `torch.compile` outcome, the cause of each failure and a search box with result
filters. It is rendered at build time from the committed snapshot
`docs/source/_data/export_support.json`; `docs/export_support/run.py` reruns the survey to refresh
it. (#4182)
