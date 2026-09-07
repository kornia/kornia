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

Scope: this guards the checkpoints that go through
:func:`kornia.core.download.load_state_dict_from_url`, which is what the
``weights/`` cache holds. Two other download paths live in ``kornia/`` and are
invisible to the prefetch list -- its sibling
:func:`kornia.core.download.download_file_from_url`, which the vision-language
builders (``kornia/models/{kimi_vl,siglip2}/builder.py``) fetch a
``model.safetensors`` with, and ``CachedDownloader`` into
``.kornia_hub/`` (``kornia/models/small_sr.py``, ``kornia/contrib/super_resolution.py``,
``kornia/onnx/utils.py``, ``kornia/models/_hf_models/hf_onnx_community.py``,
``kornia/feature/lightglue_onnx/utils/download.py``). Nothing in the PR matrix
downloads through either today; a test that did would fetch live from every
matrix cell with this file still green.

``download_file_from_url`` does share the ``weights/`` cache, so its callers
could in principle be prefetched -- they are not, for a reason the checkpoints
above do not have: SigLIP2 takes the repository as an *argument*, so there is no
fixed set of variants to enumerate, and the two checkpoints in play are hundreds
of megabytes for a pair of integration tests. The call-site scan below covers
both functions, so a model that starts downloading through either is caught here
either way.

``docs/generate_examples.py`` also fetches one non-checkpoint tensor
(``knchurch_disk.pt``, the image pair the matching examples are drawn on)
straight through ``torch.hub.load_state_dict_from_url``. It lands in the same
``weights/`` cache but comes from no registry, so it is outside this guard as
well; it is small, single-source and fetched once per docs build.
"""

from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any, get_args, get_type_hints
from urllib.parse import urlparse

import pytest

from kornia.feature import affine_shape, defmo, hardnet, hynet, keynet, mkd, orientation, sosnet, tfeat, xfeat
from kornia.feature import lightglue as lightglue_mod
from kornia.feature.aliked import aliked
from kornia.feature.dedode import dedode
from kornia.feature.dedode import encoder as dedode_encoder
from kornia.feature.disk import disk
from kornia.feature.lightglue import LightGlue
from kornia.feature.loftr import loftr
from kornia.feature.sold2 import sold2, sold2_detector
from kornia.filters import dexined
from kornia.models import dexined as dexined_model
from kornia.models import tiny_vit, vit
from kornia.models.efficient_vit import model as efficient_vit
from kornia.models.rt_detr import model as rt_detr
from kornia.models.sam import model as sam
from kornia.models.yunet import model as yunet

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".github" / "download-models-weights.py"


@functools.lru_cache(maxsize=1)
def _load_prefetch_script() -> Any:
    """Import the prefetch script for its ``MODELS`` table.

    Cached: the parametrized coverage test asks for it once per checkpoint, and
    re-executing a script for every case is pure overhead.
    """
    spec = importlib.util.spec_from_file_location("_download_models_weights", _SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.fail(f"could not load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every weight source in the library. A registry maps a variant name to the URL
# (or ordered fallback list) that variant loads from.
#
# Most are module-level dicts. The rest build their URLs from a template or a
# helper, and are reconstructed below from the very constants the library
# formats -- never transcribed, or this file would drift the way the prefetch
# list it guards did. ``test_every_prefetched_entry_is_still_requested`` is what
# holds this table to "every": a checkpoint CI caches but no entry here accounts
# for is a hole in the enumeration.
# Each entry pairs the module the URLs live in with the registry itself, so the
# call-site scan below and the coverage checks read one list rather than two --
# a second hand-written copy of these modules is exactly the drift this file
# exists to catch.
WEIGHT_REGISTRIES: dict[str, tuple[ModuleType, dict[str, Any]]] = {
    "affine_shape": (affine_shape, affine_shape.urls),
    "orientation": (orientation, orientation.urls),
    "hardnet": (hardnet, hardnet.urls),
    "hynet": (hynet, hynet.urls),
    "sosnet": (sosnet, sosnet.urls),
    "tfeat": (tfeat, tfeat.urls),
    "mkd": (mkd, mkd.urls),
    "defmo": (defmo, defmo.urls),
    "loftr": (loftr, loftr.urls),
    "sold2": (sold2, sold2.urls),
    "sold2_detector": (sold2_detector, sold2_detector.urls),
    "disk": (disk, disk.urls),
    "dedode": (dedode, dedode.urls),
    "rt_detr": (rt_detr, rt_detr.URLs),
    "keynet": (keynet, {"keynet": keynet.KeyNet_URL}),
    "yunet": (yunet, {"yunet": yunet._url}),
    # Two independent registries, identical today, both loading the same cache
    # name -- which is what hides the models copy from the coverage check: the
    # filters copy accounts for the name either way. Holding both to MODELS also
    # holds the two copies equal to each other.
    "dexined": (dexined, {"dexined": dexined.url}),
    "dexined_model": (dexined_model, {"dexined": dexined_model._url}),
    "xfeat": (xfeat, {"xfeat": xfeat.XFeat.weights_url}),
    "dedode_encoder": (dedode_encoder, dedode_encoder.urls),
    "tiny_vit": (tiny_vit, tiny_vit.urls),
    "sam": (sam, {variant.name: url for variant, url in sam.urls.items()}),
    # ALIKED formats one template pair per backbone configuration.
    "aliked": (aliked, {name: [t.format(name) for t in aliked._CHECKPOINT_URLS] for name in aliked._ALIKED_CFGS}),
    # ViT and EfficientViT compute their URL from the variant.
    "vit": (vit, {variant: vit._get_weight_url(variant) for variant in vit._AVAILABLE_WEIGHTS}),
    "efficient_vit": (
        efficient_vit,
        {
            f"{model_type}-r{resolution}": efficient_vit._get_base_url(model_type, resolution)
            for model_type in get_args(get_type_hints(efficient_vit._get_base_url)["model_type"])
            for resolution in get_args(get_type_hints(efficient_vit._get_base_url)["resolution"])
        },
    ),
}

# The modules :data:`WEIGHT_REGISTRIES` reads from, as repository-relative paths.
# ``test_every_download_call_site_is_enumerated`` holds the table to its "every
# weight source" claim by checking this set against the files that actually call
# ``load_state_dict_from_url``: a new model with weights is then a failure here
# rather than a checkpoint nothing in this file can see.
_ENUMERATED_MODULE_PATHS = tuple(
    dict.fromkeys(Path(module.__file__).resolve() for module, _ in WEIGHT_REGISTRIES.values())
)

# The call-site scan reads ``kornia/`` from the checkout, so it can only speak
# for a kornia imported from that same checkout. An installed copy is a legitimate
# way to run the suite; it just makes this one check meaningless rather than
# failing, so it is skipped there instead of turning the whole file into a
# collection error on a path that is not relative to the repository.
_KORNIA_IS_THE_CHECKOUT = all(path.is_relative_to(_REPO_ROOT) for path in _ENUMERATED_MODULE_PATHS)
_ENUMERATED_MODULES = (
    {path.relative_to(_REPO_ROOT).as_posix() for path in _ENUMERATED_MODULE_PATHS} if _KORNIA_IS_THE_CHECKOUT else set()
)

# Modules that call ``load_state_dict_from_url`` without being a registry of
# their own, with the reason they need no entry above.
_DOWNLOAD_CALL_ALLOWLIST = {
    # ``kornia/core/__init__.py`` is deliberately absent: it re-exports the
    # names without ever calling one, so the scan below never matches it and an
    # entry for it would be exactly the dead exemption
    # ``test_no_allowlisted_module_has_stopped_downloading`` exists to catch.
    "kornia/core/download.py": "defines the functions",
    "kornia/feature/lightglue.py": "builds its URLs in __init__; captured by _lightglue_sources",
    "kornia/models/base.py": "loads whatever checkpoint the config it is handed carries",
    "kornia/models/kimi_vl/builder.py": "one safetensors checkpoint, hundreds of MB, integration tests only",
    "kornia/models/siglip2/builder.py": "safetensors checkpoint of a caller-named repo; no fixed variant list",
}

# Checkpoints deliberately left out of the CI cache, with the reason. A variant
# belongs here when nothing CI runs instantiates it; the cost of caching every
# variant of every model is paid on every job, so only what CI runs is cached.
#
# "What CI runs" is wider than the pytest matrix. The docs job restores the same
# cache and then builds the API reference, and ``docs/generate_examples.py``
# constructs KeyNet, DISK, ALIKED and XFeat pretrained with no download guard --
# so a reason phrased only in terms of tests and doctests can be true and still
# leave a checkpoint being fetched live on every documentation build.
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
    "sold2_wireframe.tar": "every SOLD2_detector in the suite passes pretrained=False",
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
    # DeDoDe: ``TestDeDoDe`` is class-level ``@pytest.mark.skip`` ("DeDoDe is
    # ummaintained"), so the scheduled slow run does not reach it either. The only
    # DeDoDe construction CI performs is the ``DeDoDe.from_pretrained`` doctest,
    # which selects the ``L-C4-v2`` detector and the ``B-upright`` descriptor --
    # those two are prefetched and every other variant is dead weight in the
    # cache. ``dedode_descriptor_G`` and the DINOv2 backbone it pulls are 2.2 GB
    # of it.
    "dedode_detector_L.pth": "superseded by the L_v2 detector the doctest uses",
    "dedode_detector_C4.pth": "steerer variant nothing in CI selects",
    "dedode_detector_SO2.pth": "steerer variant nothing in CI selects",
    "B_C4_Perm_descriptor_setting_C.pth": "steerer variant nothing in CI selects",
    "B_SO2_Spread_descriptor_setting_C.pth": "steerer variant nothing in CI selects",
    "G_C4_Perm_descriptor_setting_C.pth": "steerer variant nothing in CI selects",
    "G_SO2_Spread_descriptor_setting_C.pth": "steerer variant nothing in CI selects",
    "dedode_descriptor_G.pth": "only TestDeDoDe builds the G descriptor, and it is skipped",
    "dinov2_vitl14_pretrain.pth": "reachable only through the G descriptor, which nothing in CI builds",
    # Whole-model checkpoints whose only pretrained test carries
    # ``@pytest.mark.slow``, and which nothing in the docs build constructs. The
    # PR matrix leaves KORNIA_TEST_RUNSLOW unset and deselects them; the
    # scheduled run that does select them fetches these live, which is the
    # pre-existing state and not what the OriNet failures were.
    "tiny_vit_5m_22kto1k_distill.pth": "TinyViT.from_config('5m', pretrained=True) is slow-marked",
    "tiny_vit_11m_22kto1k_distill.pth": "TinyViT.from_config('11m', pretrained=True) is slow-marked",
    "tiny_vit_21m_22kto1k_distill.pth": "TinyViT.from_config('21m', pretrained=True) is slow-marked",
    "sam_vit_b_01ec64.pth": "Sam.from_config(SamConfig('vit_b', pretrained=True)) is slow-marked",
    "mobile_sam.pt": "Sam.from_config(SamConfig('mobile_sam', pretrained=True)) is slow-marked",
    "b1-r224.pt": "EfficientViT.from_config(EfficientViTConfig()) is slow-marked",
    # Variants of those models that no test selects at all.
    "aliked-t16.pth": "tests build ALIKED(model_name='aliked-t16') untrained",
    "aliked-n16rot.pth": "no test loads pretrained weights for this ALIKED variant",
    "aliked-n32.pth": "no test loads pretrained weights for this ALIKED variant",
    "tiny_vit_5m_22k_distill.pth": "in22k variant; pretrained=True selects in1k",
    "tiny_vit_11m_22k_distill.pth": "in22k variant; pretrained=True selects in1k",
    "tiny_vit_21m_22k_distill.pth": "in22k variant; pretrained=True selects in1k",
    "tiny_vit_21m_22kto1k_384_distill.pth": "img_size>=384 variant; no test asks for it",
    "tiny_vit_21m_22kto1k_512_distill.pth": "img_size>=512 variant; no test asks for it",
    "sam_vit_l_0b3195.pth": "only vit_b and mobile_sam have a pretrained test",
    "sam_vit_h_4b8939.pth": "only vit_b and mobile_sam have a pretrained test",
    "b2-r224.pt": "test_config only checks the URL string; nothing downloads it",
    "b3-r224.pt": "test_config only checks the URL string; nothing downloads it",
    "b1-r256.pt": "test_config only checks the URL string; nothing downloads it",
    "b2-r256.pt": "test_config only checks the URL string; nothing downloads it",
    "b3-r256.pt": "test_config only checks the URL string; nothing downloads it",
    "b1-r288.pt": "test_config only checks the URL string; nothing downloads it",
    "b2-r288.pt": "test_config only checks the URL string; nothing downloads it",
    "b3-r288.pt": "test_config only checks the URL string; nothing downloads it",
    # ViT: the from_config doctest builds vit_b/16, which is prefetched.
    "vit_l-16.pth": "no test or doctest selects this ViT variant",
    "vit_s-16.pth": "no test or doctest selects this ViT variant",
    "vit_ti-16.pth": "no test or doctest selects this ViT variant",
    "vit_b-32.pth": "no test or doctest selects this ViT variant",
    "vit_s-32.pth": "no test or doctest selects this ViT variant",
}


# ``download_file_from_url`` and its Hub wrapper ``download_hf_file`` fetch into
# the same cache without loading what they fetched, so a checkpoint pulled
# through either is as invisible to the prefetch list as one pulled through
# ``load_state_dict_from_url``.
_DOWNLOAD_CALL = re.compile(r"\b(?:load_state_dict_from_url|download_file_from_url|download_hf_file)\s*\(")


def _download_call_sites() -> set[str]:
    """Return the repository-relative modules in ``kornia/`` that call a download function."""
    return {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in sorted((_REPO_ROOT / "kornia").rglob("*.py"))
        if _DOWNLOAD_CALL.search(path.read_text(encoding="utf-8"))
    }


def _as_list(url: str | list[str]) -> list[str]:
    """Normalise a registry value to the ordered source list it stands for."""
    return [url] if isinstance(url, str) else list(url)


def _pinned_cache_name(url: str | list[str]) -> str:
    """Return the cache filename :func:`load_state_dict_from_url` would use.

    A list pins the basename of its **first** entry for every attempt, so the
    fallback source shares one cache slot with the primary.
    """
    return Path(urlparse(_as_list(url)[0]).path).name


# LightGlue does not expose a registry: it derives both the URL and the cache
# name inside ``__init__``, and that name is *not* the URL basename -- the
# superpoint head loads ``superpoint_lightglue.pth`` but looks for it under
# ``superpoint_lightglue_v0-1_arxiv-pth``. A prefetch entry keyed by the
# basename would store the file where nothing reads it, so the names are
# captured from the class itself rather than written down twice.
LIGHTGLUE_FEATURES = tuple(sorted(LightGlue.features))


def _lightglue_sources(monkeypatch) -> dict[str, tuple[str, list[str]]]:
    """Return ``{feature: (cache filename, source urls)}`` without downloading anything."""
    captured: dict[str, tuple[str, list[str]]] = {}
    pending: list[str] = []

    def _capture(url: Any, **kwargs: Any) -> dict[str, Any]:
        captured[pending[-1]] = (kwargs["file_name"], _as_list(url))
        raise _Captured

    monkeypatch.setattr(lightglue_mod, "load_state_dict_from_url", _capture)
    for feature in LIGHTGLUE_FEATURES:
        pending.append(feature)
        try:
            LightGlue(feature)
        except _Captured:
            pass
    return captured


def _lightglue_cache_names(monkeypatch) -> dict[str, str]:
    """Return ``{feature: cache filename}`` without downloading anything."""
    return {feature: name for feature, (name, _) in _lightglue_sources(monkeypatch).items()}


class _Captured(Exception):
    """Raised to stop LightGlue's ``__init__`` once the cache name is known."""


def _iter_checkpoints():
    """Yield ``(label, cache filename, source urls)`` for every checkpoint the library requests."""
    for registry, (_, entries) in WEIGHT_REGISTRIES.items():
        for variant, url in entries.items():
            if isinstance(url, dict):  # dedode nests detector/descriptor tables
                for sub, sub_url in url.items():
                    yield f"{registry}.{variant}.{sub}", _pinned_cache_name(sub_url), _as_list(sub_url)
            else:
                yield f"{registry}.{variant}", _pinned_cache_name(url), _as_list(url)


_CHECKPOINTS = list(_iter_checkpoints())
_CHECKPOINT_IDS = [label for label, _, _ in _CHECKPOINTS]


class TestWeightsPrefetchCoverage:
    def test_script_is_importable(self) -> None:
        assert _SCRIPT.is_file(), f"{_SCRIPT} is missing"
        assert _load_prefetch_script().MODELS

    @pytest.mark.parametrize(
        ("label", "cache_name"), [(label, name) for label, name, _ in _CHECKPOINTS], ids=_CHECKPOINT_IDS
    )
    def test_checkpoint_is_prefetched_or_exempt(self, label: str, cache_name: str) -> None:
        prefetched = _load_prefetch_script().MODELS
        assert cache_name in prefetched or cache_name in NOT_PREFETCHED, (
            f"{label} loads {cache_name!r}, which CI neither prefetches nor exempts. "
            f"Add it to MODELS in {_SCRIPT.relative_to(_REPO_ROOT)} (keyed by this exact "
            f"cache filename), or to NOT_PREFETCHED here with the reason it is not needed."
        )

    def test_prefetched_urls_match_the_library(self) -> None:
        """The cache is keyed by filename, so a repointed URL is invisible to the check above.

        ``MODELS`` is a second copy of the registries in ``kornia/``, and nothing
        else holds the two equal. Repoint a registry at a new revision or swap a
        dead mirror without touching the prefetch list and the key is unchanged:
        the coverage check stays green while CI caches the *old* bytes under the
        exact name every job looks for, so either the prefetch job hard-fails on a
        dead link or every matrix cell loads a stale checkpoint from a silent
        cache hit -- no download, no warning.
        """
        prefetched = _load_prefetch_script().MODELS
        drifted = [
            f"{label}: library {urls} != prefetch {_as_list(prefetched[cache_name])}"
            for label, cache_name, urls in _CHECKPOINTS
            if cache_name in prefetched and _as_list(prefetched[cache_name]) != urls
        ]
        assert not drifted, (
            "the prefetch list no longer mirrors the registries it copies:\n  "
            + "\n  ".join(drifted)
            + f"\nUpdate MODELS in {_SCRIPT.relative_to(_REPO_ROOT)}."
        )

    def test_exemptions_are_still_reachable(self, monkeypatch) -> None:
        """A stale exemption hides a checkpoint that no longer exists."""
        live = {name for _, name, _ in _iter_checkpoints()}
        live |= set(_lightglue_cache_names(monkeypatch).values())
        stale = sorted(set(NOT_PREFETCHED) - live)
        assert not stale, f"NOT_PREFETCHED lists checkpoints the library no longer requests: {stale}"

    def test_lightglue_heads_are_prefetched_or_exempt(self, monkeypatch) -> None:
        prefetched = _load_prefetch_script().MODELS
        sources = _lightglue_sources(monkeypatch)
        assert set(sources) == set(LIGHTGLUE_FEATURES), "a LightGlue head stopped loading weights"
        for feature, (cache_name, urls) in sorted(sources.items()):
            assert cache_name in prefetched or cache_name in NOT_PREFETCHED, (
                f"LightGlue({feature!r}) looks for {cache_name!r}, which CI neither "
                f"prefetches nor exempts. Note this is not the URL basename."
            )
            if cache_name in prefetched:
                assert _as_list(prefetched[cache_name]) == urls, (
                    f"LightGlue({feature!r}) loads {cache_name!r} from {urls}, but CI "
                    f"prefetches it from {_as_list(prefetched[cache_name])}. A head prefetched "
                    f"from fewer sources than it loads from has no fallback in the one job "
                    f"that runs before the matrix."
                )

    def test_every_prefetched_entry_is_still_requested(self, monkeypatch) -> None:
        """A cached checkpoint no registry accounts for is a hole in the enumeration.

        The other direction of the guard. Without it :data:`WEIGHT_REGISTRIES` --
        itself hand-maintained -- can miss a weight source entirely, and every
        checkpoint that source loads is then invisible: the prefetch list can go
        stale exactly the way OriNet's did and this file stays green.
        """
        live = {name for _, name, _ in _iter_checkpoints()}
        live |= set(_lightglue_cache_names(monkeypatch).values())
        orphaned = sorted(set(_load_prefetch_script().MODELS) - live)
        assert not orphaned, (
            f"CI prefetches {orphaned}, which no registry here accounts for. Add the "
            f"weight source to WEIGHT_REGISTRIES so the entry is guarded in both directions."
        )

    def test_no_allowlisted_module_has_stopped_downloading(self) -> None:
        """A dead allowlist entry reads as coverage for a module that no longer downloads.

        The scan below subtracts this allowlist, so an entry that matches nothing
        is invisible: rename or delete a builder and its exemption survives,
        ready to excuse a *new* module that lands under the old path. The
        stale-exemption check :data:`NOT_PREFETCHED` gets is this one.
        """
        if not _KORNIA_IS_THE_CHECKOUT:
            pytest.skip("kornia is imported from outside this checkout; the call-site scan cannot match it")
        stale = sorted(set(_DOWNLOAD_CALL_ALLOWLIST) - _download_call_sites())
        assert not stale, (
            f"_DOWNLOAD_CALL_ALLOWLIST exempts modules that no longer call a download "
            f"function: {stale}. Drop the entry, or fix the path it was renamed to."
        )

    def test_every_download_call_site_is_enumerated(self) -> None:
        """Nothing else holds :data:`WEIGHT_REGISTRIES` to "every weight source in the library".

        The two directions above both start from this table, so a source missing
        from it is invisible to all of them -- and a duplicated cache name hides
        the miss even from ``test_every_prefetched_entry_is_still_requested``,
        which is exactly how the second DexiNed registry in ``kornia/models``
        went unenumerated. Checking the call sites closes the class rather than
        the instance: the next model that loads weights fails here until its
        registry is listed.
        """
        if not _KORNIA_IS_THE_CHECKOUT:
            pytest.skip("kornia is imported from outside this checkout; the call-site scan cannot match it")
        unaccounted = sorted(_download_call_sites() - _ENUMERATED_MODULES - set(_DOWNLOAD_CALL_ALLOWLIST))
        assert not unaccounted, (
            f"these modules download weights but no entry in WEIGHT_REGISTRIES reads their "
            f"URLs, so their checkpoints are invisible to every check in this file: "
            f"{unaccounted}. Add the registry to WEIGHT_REGISTRIES, or the module to "
            f"_DOWNLOAD_CALL_ALLOWLIST with the reason it has no registry of its own."
        )

    def test_no_entry_is_both_prefetched_and_exempt(self) -> None:
        prefetched = _load_prefetch_script().MODELS
        overlap = sorted(set(prefetched) & set(NOT_PREFETCHED))
        assert not overlap, f"listed as both cached and exempt: {overlap}"
