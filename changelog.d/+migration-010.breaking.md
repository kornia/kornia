Kornia's minimum supported PyTorch version rises to 2.5.1. Previously the declared floor was
2.0.0; PR-time CI has only ever exercised 2.5.1 and newer, and this same change retires the
scheduled-CI legs that tested anything older.
Installing kornia now requires `torch>=2.5.1`; a `torch` install below that version no longer
satisfies the dependency, and `pip`/`uv` will refuse to resolve it. (#4197)
