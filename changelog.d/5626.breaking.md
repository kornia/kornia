`is_exporting()` is now also true inside a `torch.compile` trace. Before this, a compiled call took
the export-safe paths (closed-form inverses, `sort`-based medians, skipped data-dependent checks and
in-`forward` side effects) on some torch versions and the eager paths on others: torch 2.14 reports
`torch.compiler.is_exporting()` as `False` under a Dynamo trace, while e.g. 2.5.1 folds it to
`True`. A compiled call now takes the same paths on every supported version — which on 2.14 means it
skips validation that eager still runs, the behaviour 2.5.1 and 2.9.1 already had.
