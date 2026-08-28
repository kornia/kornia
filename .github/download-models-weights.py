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

"""Pre-populate the torch hub cache with the checkpoints CI needs.

CI restores the resulting ``weights/`` directory before the test and doctest
jobs, so a checkpoint listed here is never fetched from inside a test run. One
that is *missing* is fetched live by every matrix cell at once, which is how
unauthenticated jobs trip the rate limits of huggingface.co and github.com and
fail with what looks like a dead URL. ``tests/core/test_weights_prefetch.py``
guards against that: every checkpoint kornia can request must be listed here or
explicitly exempted there.

Keys are the **cache filename** kornia will look for, not a friendly name.
That is usually the basename of the primary URL, but not always -- LightGlue
pins names such as ``superpoint_lightglue_v0-1_arxiv-pth`` -- and a key that
disagrees stores the file where nothing looks for it, leaving the cache
silently useless. Values mirror the URL list in kornia, so the fallback source
and the retry/backoff of :func:`kornia.core.download.load_state_dict_from_url`
apply to the prefetch too.
"""

import argparse
import logging
import os

import torch

from kornia.core.download import load_state_dict_from_url

logger = logging.getLogger(__name__)

# Format: "<cache filename>": "<url>" | ["<primary url>", "<fallback url>"]
MODELS: dict[str, "str | list[str]"] = {
    # -- detectors, descriptors and orientation estimators -------------------
    # AffNet + OriNet: LAFAffNetShapeEstimator / LAFOrienter, and every composite
    # built on them (GFTTAffNetHardNet, KeyNetHardNet, KeyNetAffNetHardNet).
    "AffNet.pth": [
        "https://huggingface.co/kornia/affnet/resolve/main/AffNet.pth",
        "https://github.com/ducha-aiki/affnet/raw/master/pretrained/AffNet.pth",
    ],
    "OriNet.pth": [
        "https://huggingface.co/kornia/orinet/resolve/main/OriNet.pth",
        "https://github.com/ducha-aiki/affnet/raw/master/pretrained/OriNet.pth",
    ],
    # KeyNet detector: KeyNetDetector(True), used by KeyNetHardNet.
    "keynet_pytorch.pth": [
        "https://huggingface.co/kornia/keynet/resolve/main/keynet_pytorch.pth",
        "https://github.com/axelBarroso/Key.Net-Pytorch/raw/main/model/weights/keynet_pytorch.pth",
    ],
    # HardNet: the default descriptor of LAFDescriptor, so nearly every composite.
    "checkpoint_liberty_with_aug.pth": [
        "https://huggingface.co/kornia/hardnet/resolve/main/checkpoint_liberty_with_aug.pth",
        "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/"
        "checkpoint_liberty_with_aug.pth",
    ],
    # Patch descriptors with pretrained smoke/jit tests.
    "HyNet_LIB.pth": [
        "https://huggingface.co/kornia/hynet/resolve/main/HyNet_LIB.pth",
        "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_LIB.pth",
    ],
    "sosnet_32x32_liberty.pth": [
        "https://huggingface.co/kornia/sosnet/resolve/main/sosnet_32x32_liberty.pth",
        "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_liberty.pth",
    ],
    "tfeat-liberty.params": [
        "https://huggingface.co/kornia/tfeat/resolve/main/tfeat-liberty.params",
        "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-liberty.params",
    ],
    # MKD whitening models: TestMKDDescriptor parametrizes over all three kernels.
    "mkd-concat-64.pth": "https://github.com/manyids2/mkd_pytorch/raw/master/mkd_pytorch/mkd-concat-64.pth",
    "mkd-polar-64.pth": "https://github.com/manyids2/mkd_pytorch/raw/master/mkd_pytorch/mkd-polar-64.pth",
    "mkd-cart-64.pth": "https://github.com/manyids2/mkd_pytorch/raw/master/mkd_pytorch/mkd-cart-64.pth",
    # DISK - feature extraction (DISK.from_pretrained doctest and tests)
    "depth-save.pth": [
        "https://huggingface.co/kornia/disk/resolve/main/depth-save.pth",
        "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth",
    ],
    # SANDesc descriptor: the pretrained tests and the class doctest call SANDesc.from_pretrained.
    "sandesc_aliked.pth": [
        "https://huggingface.co/mattia-durso/SANDesc/resolve/main/pretrained/sandesc_aliked.pth?download=true",
        "https://cloud.tugraz.at/index.php/s/dBiF999GBMoRg8w/download/sandesc_aliked.pth",
    ],
    # -- matchers ------------------------------------------------------------
    # LoFTR: tests instantiate both the outdoor and indoor weights.
    "loftr_outdoor.ckpt": [
        "https://huggingface.co/kornia/loftr/resolve/main/loftr_outdoor.ckpt",
        "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt",
    ],
    "loftr_indoor.ckpt": [
        "https://huggingface.co/kornia/loftr/resolve/main/loftr_indoor.ckpt",
        "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt",
    ],
    # LightGlue pins its own cache names; these keys are not the URL basenames.
    "superpoint_lightglue_v0-1_arxiv-pth": [
        "https://huggingface.co/kornia/lightglue/resolve/main/superpoint_lightglue.pth",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
    ],
    "doghardnet_v0-1_arxiv-pth": [
        "https://huggingface.co/kornia/lightglue/resolve/main/doghardnet_lightglue.pth",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/doghardnet_lightglue.pth",
    ],
    "disk_lightglue_v0-1_arxiv-pth": [
        "https://huggingface.co/kornia/lightglue/resolve/main/disk_lightglue.pth",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth",
    ],
    # -- DeDoDe --------------------------------------------------------------
    # Only the pair the ``DeDoDe.from_pretrained`` doctest builds: ``TestDeDoDe``
    # is skipped outright, so every other variant -- and the 1.2 GB DINOv2
    # backbone the G descriptor pulls -- would be cached for nothing.
    "dedode_detector_L_v2.pth": [
        "https://huggingface.co/kornia/dedode/resolve/main/dedode_detector_L_v2.pth",
        "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
    ],
    "dedode_descriptor_B.pth": [
        "https://huggingface.co/kornia/dedode/resolve/main/dedode_descriptor_B.pth",
        "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
    ],
    # -- line, edge and object models ---------------------------------------
    "sold2_wireframe.pth": [
        "https://huggingface.co/kornia/sold2/resolve/main/sold2_wireframe.pth",
        "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth",
    ],
    "DexiNed_BIPED_10.pth": [
        "https://huggingface.co/kornia/dexined/resolve/main/DexiNed_BIPED_10.pth",
        "http://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth",
    ],
    "yunet_final.pth": [
        "https://huggingface.co/kornia/yunet/resolve/main/yunet_final.pth",
        "https://github.com/kornia/data/raw/main/yunet_final.pth",
    ],
    "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth": [
        "https://huggingface.co/kornia/rt_detr/resolve/main/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    ],
    "vit_b-16.pth": "https://huggingface.co/kornia/vit_b16_augreg_i21k_r224/resolve/main/vit_b-16.pth",
    # ALIKED and XFeat: no pytest job builds them pretrained, but
    # ``generate_examples.main`` does, once per docs build, and the docs job
    # restores this same cache.
    "aliked-n16.pth": [
        "https://huggingface.co/kornia/aliked/resolve/main/aliked-n16.pth",
        "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
    ],
    "xfeat.pt": "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt",
    # -- deblurring ----------------------------------------------------------
    # DeFMO(True): smoke and jit tests instantiate both halves.
    "encoder_best.pt": [
        "https://huggingface.co/kornia/defmo/resolve/main/encoder_best.pt",
        "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/encoder_best.pt",
    ],
    "rendering_best.pt": [
        "https://huggingface.co/kornia/defmo/resolve/main/rendering_best.pt",
        "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/rendering_best.pt",
    ],
}


if __name__ == "__main__":
    # Configured here rather than at import time: the drift guard imports this
    # file to read MODELS, and a test helper has no business installing a handler
    # on the root logger for the rest of the session.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser("WeightsDownloader")
    parser.add_argument("--target_directory", "-t", required=False, default="weights")

    args = parser.parse_args()

    # Set torch.hub directory - files will go to {target_directory}/checkpoints/
    torch.hub.set_dir(args.target_directory)
    # For HuggingFace model caching
    os.environ["HF_HOME"] = args.target_directory

    logger.info(f"Downloading models to: {torch.hub.get_dir()}/checkpoints/")

    # A failure is recorded rather than raised, so one run reports every dead
    # source instead of the first one and then stopping -- with two sources and
    # three attempts each behind every entry, finding them one push at a time is
    # expensive. The exit code still fails the job: ``actions/cache`` does not
    # save on a failed job, and the test matrix is gated on this one
    # (``needs: [pre-tests]``), so a missing checkpoint stops CI rather than
    # letting every matrix cell fetch it live.
    failed: list[str] = []
    for file_name, url in MODELS.items():
        logger.info(f"Downloading `{file_name}` from `{url if isinstance(url, str) else url[0]}`...")
        try:
            # Don't pass model_dir - use the default from torch.hub.set_dir()
            # This ensures files go to {hub_dir}/checkpoints/ matching test behavior.
            # file_name is pinned so the entry lands where kornia will look for it.
            load_state_dict_from_url(url, map_location=torch.device("cpu"), file_name=file_name)
        except Exception as e:  # noqa: BLE001 - report every failure, not just the first
            logger.error(f"Failed to download `{file_name}`: {type(e).__name__}: {e}")
            failed.append(file_name)

    if failed:
        logger.error(f"{len(failed)} of {len(MODELS)} checkpoints could not be downloaded: {failed}")
        raise SystemExit(1)

    logger.info("All models downloaded successfully!")
    raise SystemExit(0)
