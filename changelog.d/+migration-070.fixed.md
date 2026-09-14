`packaging` is no longer a runtime dependency; it was never imported. The `dev` extra no longer lists
`pytest-cov` (CI runs `coverage run -m pytest`), `ruff` (the pre-commit hook installs the pinned copy) or a
`numpy<3` cap, and `uv.lock` is regenerated to match (#4296).
