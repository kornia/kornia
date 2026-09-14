The pinned-torch CI legs no longer run against the lockfile's `triton`. `uv sync` installs the locked torch's
dependency set, and each matrix leg then swaps torch alone for a `+cpu` build -- whose wheels declare no `triton` at
all -- so the lockfile's triton survived as an orphan of a torch that was no longer installed. torch guards its
inductor triton path on import availability rather than on version, so it dispatched onto that orphan and the
`2.6.0` leg died in `from triton.backends.compiler import AttrsDescriptor` on 32 dynamo and export tests, having
never been green. The install step now drops sidecar packages the installed torch does not declare, and the
verification step asserts the whole environment against that torch (`.github/scripts/check_torch_env.py`) so a
mis-provisioned venv fails at setup instead of as a wall of unrelated-looking test failures. The five
`tests/core/test_small_linalg.py` ONNX cases the import error was masking now export with `external_data=False`,
which torch 2.6's dynamo exporter requires when the destination is a `BytesIO`. (#4199, #4200)
