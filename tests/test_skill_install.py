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

"""Tests for the kornia-augmentations skill installer."""

from pathlib import Path

from kornia.augmentations.skills.install import _default_target, _src_path, install


def test_src_skill_exists():
    src = _src_path()
    assert src.is_file(), f"skill source not found at {src}"
    assert src.stat().st_size > 1000, "skill source suspiciously small"


def test_default_target_path_shape():
    t = _default_target()
    assert ".claude" in t.parts
    assert "skills" in t.parts
    assert "kornia-augmentations" in t.parts
    assert t.name == "SKILL.md"


def test_install_to_tmp_path(tmp_path: Path):
    target = tmp_path / "skills" / "kornia-augmentations" / "SKILL.md"
    out = install(target=target, quiet=True)
    assert out == target
    assert target.is_file()
    content = target.read_text()
    assert content.startswith("---") or "kornia-augmentations" in content


def test_install_idempotent_no_force(tmp_path: Path):
    target = tmp_path / "SKILL.md"
    install(target=target, quiet=True)
    mtime1 = target.stat().st_mtime
    # Install again without force — should not overwrite
    install(target=target, quiet=True)
    assert target.stat().st_mtime == mtime1


def test_install_force_overwrites(tmp_path: Path):
    target = tmp_path / "SKILL.md"
    install(target=target, quiet=True)
    target.write_text("MODIFIED")
    install(target=target, force=True, quiet=True)
    content = target.read_text()
    assert content != "MODIFIED"
    assert "kornia-augmentations" in content or content.startswith("---")


def test_install_creates_parent_dirs(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "SKILL.md"
    install(target=deep, quiet=True)
    assert deep.is_file()
