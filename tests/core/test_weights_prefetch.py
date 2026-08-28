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

"""Guard the CI weights cache against drifting away from the library.

``.github/download-models-weights.py`` fills the ``weights/`` cache that CI
restores before the test and doctest jobs. A checkpoint missing from it is
downloaded live by every matrix cell at once, which trips the anonymous rate
limits of huggingface.co and github.com and surfaces as a ``Failed to load
weights from all N source(s)`` error naming a URL that is in fact serving fine.
That list had not been touched since the HF mirrors landed in #3655, so OriNet
and sixteen other checkpoints were never cached.

Adding a checkpoint to kornia therefore fails this test until it is either
prefetched or listed in :data:`NOT_PREFETCHED` with a reason.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from kornia.feature import affine_shape, defmo, hardnet, hynet, keynet, mkd, orientation, sosnet, tfeat, xfeat
from kornia.feature import lightglue as lightglue_mod
from kornia.feature.dedode import dedode
from kornia.feature.disk import disk
from kornia.feature.lightglue import LightGlue
from kornia.feature.loftr import loftr
from kornia.feature.sold2 import sold2, sold2_detector
from kornia.filters import dexined
from kornia.models.rt_detr import model as rt_detr
from kornia.models.yunet import model as yunet

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".github" / "download-models-weights.py"


def _load_prefetch_script() -> Any:
    spec = importlib.util.spec_from_file_location("_download_models_weights", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.fail(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every module-level weight registry in the library. A registry maps a variant
# name to the URL (or ordered fallback list) that variant loads from.
WEIGHT_REGISTRIES: dict[str, dict[str, Any]] = {
    "affine_shape": affine_shape.urls,
    "orientation": orientation.urls,
    "hardnet": hardnet.urls,
    "hynet": hynet.urls,
    "sosnet": sosnet.urls,
    "tfeat": tfeat.urls,
    "mkd": mkd.urls,
    "defmo": defmo.urls,
    "loftr": loftr.urls,
    "sold2": sold2.urls,
    "sold2_detector": sold2_detector.urls,
    "disk": disk.urls,
    "dedode": dedode.urls,
    "rt_detr": rt_detr.URLs,
    "keynet": {"keynet": keynet.KeyNet_URL},
    "yunet": {"yunet": yunet.url},
    "dexined": {"dexined": dexined.url},
    "xfeat": {"xfeat": xfeat.XFeat.weights_url},
}

# Checkpoints deliberately left out of the CI cache, with the reason. A variant
# belongs here when no test or doctest instantiates it; the cost of caching every
# variant of every model is paid on every job, so only what CI runs is cached.
NOT_PREFETCHED: dict[str, str] = {
    # Alternative training sets -- tests only ever build the liberty variants.
    "HardNetPP.pth": "HardNet++ weights; no test or doctest requests them",
    "hardnet8v2.pt": "HardNet8(pretrained=True); tests build HardNet8() untrained",
    "HyNet_ND.pth": "notredame variant; tests use the liberty default",
    "HyNet_YOS.pth": "yosemite variant; tests use the liberty default",
    "sosnet_32x32_hpatches_a.pth": "hpatches variant; tests use the liberty default",
    "tfeat-notredame.params": "notredame variant; tests use the liberty default",
    "tfeat-yosemite.params": "yosemite variant; tests use the liberty default",
    # Model variants no test selects.
    "loftr_indoor_ds_new.ckpt": "tests instantiate LoFTR('indoor') and LoFTR('outdoor') only",
    "sold2_wireframe.tar": "SOLD2_detector is only tested with pretrained=False",
    "xfeat.pt": "XFeat.from_pretrained is not exercised by the CPU test suite",
    "epipolar-save.pth": "tests call DISK.from_pretrained(checkpoint='depth')",
    # LightGlue heads: tests build the disk and doghardnet matchers, and the
    # suite also reaches the superpoint head. The rest are unused.
    "aliked_lightglue_v0-1_arxiv-pth": "no test selects the aliked LightGlue head",
    "sift_lightglue_v0-1_arxiv-pth": "no test selects the sift LightGlue head",
    "keynet_affnet_hardnet_lightglue.pth": "no test selects the keynet_affnet_hardnet LightGlue head",
    "dedodeb_lightglue.pth": "no test selects the dedodeb LightGlue head",
    "dedodeg_lightglue.pth": "no test selects the dedodeg LightGlue head",
    "raco_aliked_lightglue_v0-1_arxiv-pth": "no test selects the raco-aliked LightGlue head",
    "xfeat-lighterglue.pt": "no test selects the xfeat LightGlue head",
    "rtdetr_r34vd_dec4_6x_coco_from_paddle.pth": "only the r18vd variant is tested",
    "rtdetr_r50vd_6x_coco_from_paddle.pth": "only the r18vd variant is tested",
    "rtdetr_r50vd_m_6x_coco_from_paddle.pth": "only the r18vd variant is tested",
    "rtdetr_r101vd_6x_coco_from_paddle.pth": "only the r18vd variant is tested",
    # DeDoDe: tests cover the v2 detector and the B/G upright + B-SO2 descriptors.
    "dedode_detector_L.pth": "superseded by the L_v2 detector the tests use",
    "dedode_detector_C4.pth": "steerer variant no test selects",
    "dedode_detector_SO2.pth": "steerer variant no test selects",
    "B_C4_Perm_descriptor_setting_C.pth": "steerer variant no test selects",
    "G_C4_Perm_descriptor_setting_C.pth": "steerer variant no test selects",
    "G_SO2_Spread_descriptor_setting_C.pth": "steerer variant no test selects",
}


def _pinned_cache_name(url: str | list[str]) -> str:
    """Return the cache filename :func:`load_state_dict_from_url` would use.

    A list pins the basename of its **first** entry for every attempt, so the
    fallback source shares one cache slot with the primary.
    """
    primary = url if isinstance(url, str) else url[0]
    return Path(urlparse(primary).path).name


# LightGlue does not expose a registry: it derives both the URL and the cache
# name inside ``__init__``, and that name is *not* the URL basename -- the
# superpoint head loads ``superpoint_lightglue.pth`` but looks for it under
# ``superpoint_lightglue_v0-1_arxiv-pth``. A prefetch entry keyed by the
# basename would store the file where nothing reads it, so the names are
# captured from the class itself rather than written down twice.
LIGHTGLUE_FEATURES = tuple(sorted(LightGlue.features))


def _lightglue_cache_names(monkeypatch) -> dict[str, str]:
    """Return ``{feature: cache filename}`` without downloading anything."""
    captured: dict[str, str] = {}
    pending: list[str] = []

    def _capture(url: Any, **kwargs: Any) -> dict[str, Any]:
        captured[pending[-1]] = kwargs["file_name"]
        raise _Captured

    monkeypatch.setattr(lightglue_mod, "load_state_dict_from_url", _capture)
    for feature in LIGHTGLUE_FEATURES:
        pending.append(feature)
        try:
            LightGlue(feature)
        except _Captured:
            pass
    return captured


class _Captured(Exception):
    """Raised to stop LightGlue's ``__init__`` once the cache name is known."""


def _iter_checkpoints():
    """Yield ``(label, cache filename)`` for every checkpoint the library requests."""
    for registry, entries in WEIGHT_REGISTRIES.items():
        for variant, url in entries.items():
            if isinstance(url, dict):  # dedode nests detector/descriptor tables
                for sub, sub_url in url.items():
                    yield f"{registry}.{variant}.{sub}", _pinned_cache_name(sub_url)
            else:
                yield f"{registry}.{variant}", _pinned_cache_name(url)


class TestWeightsPrefetchCoverage:
    def test_script_is_importable(self) -> None:
        assert _SCRIPT.is_file(), f"{_SCRIPT} is missing"
        assert _load_prefetch_script().MODELS

    @pytest.mark.parametrize(("label", "cache_name"), list(_iter_checkpoints()), ids=str)
    def test_checkpoint_is_prefetched_or_exempt(self, label: str, cache_name: str) -> None:
        prefetched = _load_prefetch_script().MODELS
        assert cache_name in prefetched or cache_name in NOT_PREFETCHED, (
            f"{label} loads {cache_name!r}, which CI neither prefetches nor exempts. "
            f"Add it to MODELS in {_SCRIPT.relative_to(_REPO_ROOT)} (keyed by this exact "
            f"cache filename), or to NOT_PREFETCHED here with the reason it is not needed."
        )

    def test_exemptions_are_still_reachable(self, monkeypatch) -> None:
        """A stale exemption hides a checkpoint that no longer exists."""
        live = {name for _, name in _iter_checkpoints()}
        live |= set(_lightglue_cache_names(monkeypatch).values())
        stale = sorted(set(NOT_PREFETCHED) - live)
        assert not stale, f"NOT_PREFETCHED lists checkpoints the library no longer requests: {stale}"

    def test_lightglue_heads_are_prefetched_or_exempt(self, monkeypatch) -> None:
        prefetched = _load_prefetch_script().MODELS
        names = _lightglue_cache_names(monkeypatch)
        assert set(names) == set(LIGHTGLUE_FEATURES), "a LightGlue head stopped loading weights"
        for feature, cache_name in sorted(names.items()):
            assert cache_name in prefetched or cache_name in NOT_PREFETCHED, (
                f"LightGlue({feature!r}) looks for {cache_name!r}, which CI neither "
                f"prefetches nor exempts. Note this is not the URL basename."
            )

    def test_no_entry_is_both_prefetched_and_exempt(self) -> None:
        prefetched = _load_prefetch_script().MODELS
        overlap = sorted(set(prefetched) & set(NOT_PREFETCHED))
        assert not overlap, f"listed as both cached and exempt: {overlap}"
