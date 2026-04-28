"""kornia.augmentations.skills — agent skills shipped with kornia.

The kornia-augmentations skill activates in Claude Code / Cursor / Copilot CLI
sessions when the user works on augmentation pipelines (USER mode) or authors
new transforms (DEVELOPER mode). Install it with:

    python -m kornia.augmentations.skills.install
"""
from kornia.augmentations.skills.install import install, main

__all__ = ["install", "main"]
