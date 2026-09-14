# Changelog fragments

Each PR with a user-visible change adds its own Markdown file here. This avoids
merge conflicts caused by everyone editing the same section of `CHANGELOG.md`.
[Towncrier](https://towncrier.readthedocs.io/en/25.8.0/) combines the files at release time.

## Adding an entry

Name the file `<PR>.<type>.md`, for example `1234.fixed.md`:

```markdown
`CameraModel.project` now accepts empty batches instead of raising an error.
```

Write a short description without a heading or an outer bullet. Towncrier adds
the heading, bullet, and PR link. Use these types:

| Type | Changelog heading | Use for |
| --- | --- | --- |
| `added` | Added | New features and user-facing documentation |
| `fixed` | Bug fixes | Corrections to existing behavior |
| `breaking` | Breaking changes | Incompatible changes; describe both old and new behavior |

A PR can add files in more than one category. Put related changes in one file per
category. If there is no PR number yet, use a unique name such as
`+fix-camera-shape.fixed.md`, then rename it to the PR number before merging.
Internal refactors, tests, and CI-only changes may omit a fragment; explain why
in the PR description. Reviewers check whether an entry is needed.

Preview all pending notes from the repository root:

```bash
pixi run changelog-preview
```

This prints the notes without modifying the changelog or deleting fragments.
The pinned tool runs in an isolated uv tool environment; its first run needs
network access. It does not install Towncrier into Kornia's runtime environment.
Do not commit generated previews in ordinary PRs.

## Preparing a release

On the release branch, with a clean working tree:

1. Run `pixi run changelog-preview` and review the pending entries.
2. Run `pixi run changelog-build --version X.Y.Z` with the actual release version.
   Towncrier inserts a dated release section into `CHANGELOG.md` and asks before
   removing the consumed fragments. Confirm removal so they are not repeated in
   the next release.
3. Review the diff, including the fragment deletions, and commit it in the release PR.
   Keep this README and the insertion marker in `CHANGELOG.md`.

Only release PRs update the assembled changelog. If a pending PR modifies a
fragment already consumed by a release, resolve that conflict by deciding whether
the correction belongs in the released notes or in a new fragment.

## Existing unreleased notes

The `+migration-*` files preserve the entries that were already under `Unreleased`
when this workflow was introduced, including their existing PR references. They
use Towncrier's orphan prefix (`+`) to avoid adding duplicate PR links. Keep those
names until the next release consumes the files; new PRs use the naming rule above.
