The `Import Surface` CI check accepts deliberate public-name removals recorded in the same change.
Edits to `tests/api_surface.json` must correspond to actual export removals and acknowledge only the exact
recorded module/name. Submodule APIs recorded only under an ancestor, and APIs outside the inventory, use
exact module/name entries in `tests/api_surface_removals.json`; only new entries matching a current
`__all__` removal count. Malformed records, deleted inventory module keys, unrelated Python edits, and
acknowledgements staged in earlier changes cannot authorize removals. Both record files trigger the workflow.
The export resolver includes implicit submodule bindings and rejects unsupported dynamic binding expressions
as evidence of removal. Deprecation windows and release notes still apply. (#4190, #4230)
