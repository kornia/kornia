## What does this pull request do?

Fixes `RandomCrop(cropping_mode="slice")` annotation transforms when the requested crop is oversized along only one axis. Width and height are now scaled independently to match the image crop; resample-mode matrix behavior remains unchanged.

## Related issue

Fixes #5463

## What did you check?

- Regression coverage for width-only, height-only, both-axis, and no-axis scaling in normal and export paths.
- End-to-end box alignment against the issue's image-coordinate reproduction.
- Focused crop and convention suites: 295 passed, 5 skipped, 1 xfailed.
- `uv build --wheel`: succeeded.

## How did AI help?

AI helped trace the shared matrix scaling logic, add regression coverage, and review the patch.
