# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent.parent / ".github" / "scripts" / "check_changelog.py"
_SPEC = importlib.util.spec_from_file_location("check_changelog", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
check_changelog = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_changelog
_SPEC.loader.exec_module(check_changelog)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: S603, S607


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "changelog.d").mkdir(parents=True)
    (repo / "changelog.d" / "README.md").write_text("guide\n")
    (repo / "CHANGELOG.md").write_text("# Changelog\n")
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    _git("branch", "base", cwd=repo)
    return repo


def _run_in(repo: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.chdir(repo)
    return check_changelog.main(["--base-ref", "base", "--pr-number", "4513", *args])


def test_accepts_current_numeric_fragment(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "changelog.d" / "4513.fixed.md").write_text("Fix a public bug.\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "fragment", cwd=repo)

    assert _run_in(repo, monkeypatch) == 0


def test_requires_a_fragment_for_the_current_pr(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "changelog.d" / "1234.fixed.md").write_text("Credit another pull request.\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "other fragment", cwd=repo)

    assert _run_in(repo, monkeypatch) == 1


def test_no_changelog_skip_does_not_skip_validation(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "changelog.d" / "+unreviewed.fixed.md").write_text("An orphan.\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "invalid fragment", cwd=repo)

    assert _run_in(repo, monkeypatch, "--skip-presence") == 1


def test_no_changelog_skip_allows_an_internal_pr(tmp_path, monkeypatch):
    repo = _repo(tmp_path)

    assert _run_in(repo, monkeypatch, "--skip-presence") == 0


def test_no_changelog_skip_allows_a_release_changelog_edit(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [1.0.0]\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "release notes", cwd=repo)

    assert _run_in(repo, monkeypatch, "--skip-presence") == 0


def test_rejects_changelog_edit_in_an_ordinary_pr(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "changelog.d" / "4513.fixed.md").write_text("Fix a public bug.\n")
    (repo / "CHANGELOG.md").write_text("# Changelog\n\nA premature note.\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "fragment and changelog", cwd=repo)

    assert _run_in(repo, monkeypatch) == 1


def test_migration_fragments_are_the_only_allowed_orphans(tmp_path):
    repo = _repo(tmp_path)
    migration = repo / "changelog.d" / "+migration-001.added.md"
    migration.write_text("Existing unreleased note.\n")
    (repo / "changelog.d" / ".DS_Store").write_bytes(b"finder metadata")

    assert check_changelog.validate_fragments(repo) == []
    migration.rename(repo / "changelog.d" / "+migration-note.added.md")
    assert check_changelog.validate_fragments(repo)


@pytest.mark.parametrize(
    ("filename", "content"),
    [("4513.changed.md", "Unknown section.\n"), ("4513.fixed.md", "   \n")],
)
def test_rejects_invalid_fragment_name_or_content(tmp_path, filename, content):
    repo = _repo(tmp_path)
    (repo / "changelog.d" / filename).write_text(content)

    assert check_changelog.validate_fragments(repo)


def test_presence_uses_a_renamed_fragment_destination(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    old = repo / "changelog.d" / "temporary.fixed.md"
    old.write_text("Temporary name.\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "temporary fragment", cwd=repo)
    _git("branch", "-f", "base", cwd=repo)
    old.rename(repo / "changelog.d" / "4513.fixed.md")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "rename fragment", cwd=repo)

    monkeypatch.chdir(repo)
    changed = check_changelog.changed_paths("base")
    assert changed == {Path("changelog.d/4513.fixed.md")}
    assert check_changelog.has_current_pr_fragment(changed, 4513)


def test_deleting_a_fragment_does_not_satisfy_presence(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    fragment = repo / "changelog.d" / "4513.fixed.md"
    fragment.write_text("A former release note.\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "fragment", cwd=repo)
    _git("branch", "-f", "base", cwd=repo)
    fragment.unlink()
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "remove fragment", cwd=repo)

    monkeypatch.chdir(repo)
    assert check_changelog.changed_paths("base") == {Path("changelog.d/4513.fixed.md")}
    assert check_changelog.main(["--base-ref", "base", "--pr-number", "4513"]) == 1
