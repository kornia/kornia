## Summary

Closes #5315. Validate kernel-density bandwidths before division and prevent `image_histogram2d` from generating centers over an empty or non-finite range. Honor the documented `bandwidth=-1` automatic-bandwidth sentinel and treat empty `centers` as automatic centers.

Explicit centers with a valid explicit bandwidth remain independent of `min` and `max`.

## Validation

- `tests/enhance/test_histogram.py`: 87 passed (PyTorch 2.14.0, CPU).
- `pre-commit run --all-files`: passed.
- `pixi run -e default uv build`: wheel and source distribution built successfully.

## Follow-up with #5319

The open conventions PR adds two strict expected-failure pins for #5315 in `tests/enhance/test_conventions_enhance.py`. Once that PR is merged into `main`, remove those pins in this fix so they do not become strict XPASS failures.
