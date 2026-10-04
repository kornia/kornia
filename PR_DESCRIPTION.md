## What does this pull request do?

On MPS with PyTorch versions before 2.7, `torch.maximum` and `torch.minimum` can ignore a NaN when the other operand is finite. Since morphology's `auto` engine selects `shift` on MPS, dilation and erosion could silently return finite values for windows containing NaNs. The shift reduction now tracks NaNs on affected MPS versions and restores them in the result, while leaving other devices and newer PyTorch versions on the existing fast path.

Adds regression coverage for explicit `shift` and `auto` engines. The MPS-specific path is limited to PyTorch versions before 2.7.

## Related issue or discussion

Fixes #4997.

## What did you check?

```text
$ PYTHONPATH="$PWD" /Users/architbagad/Desktop/Kornia/kornia/.venv/bin/python -m pytest tests/morphology/test_dilation.py tests/morphology/test_erosion.py --device=cpu --dtype=float32,float64 -q
618 passed

$ PYTHONPATH="$PWD" /tmp/kornia-otsu-torch251/bin/python -m pytest tests/morphology/test_dilation.py --device=mps --dtype=float32 -k 'non_finite_inputs and (shift or auto)' -q
2 skipped: MPS is unavailable in this environment

$ pixi run pre-commit-all
Passed

$ pixi run typecheck
Passed

$ UV_NO_SYNC=1 pixi run -e default uv build --out-dir /tmp/kornia-4997-dist
Built sdist and wheel
```

The focused dilation and erosion test files also passed on CPU with float32 and float64 (618 passed). The MPS regression test is present but could not execute on this host because MPS is unavailable.

## How did AI help?

AI helped trace the MPS/PyTorch version-specific NaN behavior, implement the narrow compatibility path, and prepare regression coverage. I reviewed the change and ran the checks listed above.
