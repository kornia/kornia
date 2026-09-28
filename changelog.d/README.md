# Changelog fragments

Each PR with a user-visible change adds its own Markdown file here. This avoids
merge conflicts caused by everyone editing the same section of `CHANGELOG.md`.
[Towncrier](https://towncrier.readthedocs.io/en/25.8.0/) combines the files at release time.

## Adding an entry

Name the file `<PR>.<type>.md`, for example `1234.fixed.md`. Use a PR number,
not an issue number: generated links point to `/pull/<PR>`.

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
category. To credit several PRs for one change, put identical text in a file for
each PR in the same category (for example `1234.fixed.md` and `1235.fixed.md`).
Towncrier combines them into one bullet with both PR links.

If there is no PR number yet, use a unique name such as
`+fix-camera-shape.fixed.md`, then rename it to the PR number before merging.
Internal refactors, tests, and CI-only changes may omit a fragment; explain why
in the PR description and ask a maintainer to apply the `no-changelog` label.
CI requires a fragment and rejects edits to `CHANGELOG.md` on ordinary PRs.
The label permits internal and release PRs to omit a fragment and, when needed,
edit the assembled changelog. Filename and rendering checks still run. Rename temporary `+name` files before merging: CI rejects them.
The reserved `+migration-*` files are the one-time legacy exception.

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
   The version argument is required by this workflow; do not run the build without it.
   The task uses `--yes`: Towncrier stages the dated section in `CHANGELOG.md` and
   removes and stages the consumed fragments without prompting, including in scripts.
3. Review `git diff --cached`, including the fragment deletions, and commit it in
   the release PR. Use the `no-changelog` label for this PR, since it consumes notes
   rather than adding one. Keep this README and the insertion marker in `CHANGELOG.md`.

Only release PRs update the assembled changelog. If a pending PR modifies a
fragment already consumed by a release, resolve that conflict by deciding whether
the correction belongs in the released notes or in a new fragment.

## Existing unreleased notes

The `+migration-*` files preserve the entries that were already under `Unreleased`
when this workflow was introduced, including their existing PR references. They
use Towncrier's orphan prefix (`+`) to avoid adding duplicate PR links. Keep those
names until the next release consumes the files; new PRs use the naming rule above.

## Updating an open PR during the switch

If your PR already edits `CHANGELOG.md`, move only its new entries into
`changelog.d/<PR>.<type>.md`, preserving their text. Then merge or rebase onto the
new `main` and resolve `CHANGELOG.md` to the version on `main`, removing your old
changelog hunk. Check the PR diff: the new fragments should contain your notes,
and `CHANGELOG.md` should have no changes. Run the preview before pushing.

Remove any local `CHANGELOG.md merge=union` rule from `.git/info/attributes` or
your global attributes file before doing this. Use `git check-attr merge -- CHANGELOG.md`
to check the effective rule. A union merge can silently restore the old unreleased
block alongside the fragments, causing every old entry to appear twice at release.
Resolve the transition explicitly instead of keeping both sides of that block.

Towncrier sorts entries within each category, so do not use “above” or “below” to
refer to another entry. Name its category or PR instead. The migrated notes retain
their original plain PR references; new fragments generate links. This mixed style
is limited to the first release after migration.
