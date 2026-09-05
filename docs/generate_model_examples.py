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

"""Render the input/output figures shown on the ``docs/source/models/*.rst`` pages.

Unlike ``generate_examples.py`` this script is *not* run by the Sphinx build: it downloads several
pretrained checkpoints (RT-DETR, SAM ViT-B, SOLD2, DeFMO, ...) and runs them on the CPU, which is too
slow and too download-heavy for a docs build. Run it once when a model or its API changes and commit
the resulting figures under ``docs/source/_static/img/models/``::

    python docs/generate_model_examples.py            # all models
    python docs/generate_model_examples.py loftr sold2  # a subset

Each ``figure_<name>`` function runs the model the same way the corresponding page's "Run it" snippet
does (same builder, weights, inputs and prompts) and then draws the result; keep the two in sync when
either changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # select the headless backend before pyplot is imported

import matplotlib.pyplot as plt
import requests
import torch

import kornia as K
from kornia.image import tensor_to_image
from kornia.io import get_sample_images

torch.manual_seed(0)

OUT = Path(__file__).absolute().parent / "source/_static/img/models"
DATA = "https://raw.githubusercontent.com/kornia/data/main/"
KNCHURCH = "https://github.com/kornia/data_test/raw/8b98f44abbe92b7a84631ed06613b08fee7dae14/knchurch_disk.pt"
IMAGENET_CLASSES = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

FIGURES: dict = {}


def figure(fn):
    FIGURES[fn.__name__.removeprefix("figure_")] = fn
    return fn


def sample(name: str, size: tuple[int, int] | None = None) -> torch.Tensor:
    """One ``(1, 3, H, W)`` float image in [0, 1] from the kornia data repository."""
    img = get_sample_images(paths=[DATA + name], resize=size, as_list=True)[0]
    return img[None]


def knchurch() -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.hub.load_state_dict_from_url(KNCHURCH, map_location="cpu")
    return d["img1"], d["img2"]


def imagenet_classes() -> list[str]:
    response = requests.get(IMAGENET_CLASSES, timeout=60)
    response.raise_for_status()
    return response.text.strip().splitlines()


def show(ax, img: torch.Tensor, title: str = "") -> None:
    ax.imshow(tensor_to_image(img[0] if img.dim() == 4 else img), cmap="gray" if img.shape[-3] == 1 else None)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.jpg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"wrote {OUT / name}.jpg")


def topk_bars(ax, logits: torch.Tensor, names: list[str], title: str) -> None:
    probs = logits.softmax(-1)[0]
    top = probs.topk(5)
    ax.barh(range(5)[::-1], top.values.tolist(), color="#4c72b0")
    ax.set_yticks(range(5)[::-1], [names[i] for i in top.indices.tolist()], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("softmax probability")
    ax.set_title(title, fontsize=10)


# --------------------------------------------------------------------------- object detection


@figure
def figure_rt_detr() -> None:
    from kornia.contrib.object_detection import RTDETRDetectorBuilder

    image = sample("delorean.png")
    detector = RTDETRDetectorBuilder.build("rtdetr_r18vd")  # runs at the 640 px the weights were trained for
    detections = detector(image)  # one (D, 6) tensor per image: class id, score, x, y, w, h

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    show(axs[0], image, "input")
    show(axs[1], image, f"RT-DETR (r18vd): {detections[0].shape[0]} detections above 0.3")
    for cls, score, x, y, w, h in detections[0].tolist():
        axs[1].add_patch(mpl.patches.Rectangle((x, y), w, h, fill=False, color="lime", lw=1.5))
        axs[1].text(x, y - 3, f"class {int(cls)} · {score:.2f}", color="lime", fontsize=8, fontweight="bold")
    save(fig, "rt_detr")


@figure
def figure_yunet() -> None:
    from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint

    image = sample("crowd.jpg", (587, 900))
    detector = FaceDetector()  # YuNet expects pixel values in [0, 255]
    faces = [FaceDetectorResult(f) for f in detector(image * 255.0)[0]]  # one result per detected face

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    show(axs[0], image, "input")
    show(axs[1], image, f"YuNet: {len(faces)} faces with score > 0.3 (boxes + 5 landmarks)")
    for f in faces:
        x1, y1, x2, y2 = f.xmin.item(), f.ymin.item(), f.xmax.item(), f.ymax.item()
        axs[1].add_patch(mpl.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=1.0))
        for kp in FaceKeypoint:
            x, y = f.get_keypoint(kp).tolist()
            axs[1].plot(x, y, ".", color="yellow", ms=2.5)
    fig.tight_layout()
    save(fig, "yunet")


# --------------------------------------------------------------------------- segmentation


def _sam_figure(model_type: str, name: str, title: str) -> None:
    from kornia.contrib.visual_prompter import VisualPrompter
    from kornia.geometry.keypoints import Keypoints
    from kornia.models.sam import SamConfig

    image = sample("simba.png")[0]  # (3, H, W)
    prompter = VisualPrompter(SamConfig(model_type, pretrained=True))
    prompter.set_image(image)  # encode once, query many times

    keypoints = Keypoints(torch.tensor([[[300.0, 90.0]]]))  # (K, N, 2): K prompts of N (x, y) points; on the eye
    labels = torch.tensor([[1]])  # 1 = foreground, 0 = background
    box = what(torch.tensor([[[180.0, 20.0, 380.0, 240.0]]]), mode="xyxy")  # around the head
    point_pred = prompter.predict(keypoints=keypoints, keypoints_labels=labels)  # 3 candidate masks
    box_pred = prompter.predict(boxes=box, multimask_output=False)  # 1 mask

    best = point_pred.scores.argmax()  # pick the candidate with the highest predicted IoU
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.0))
    show(axs[0], image, "input + prompts (star: point, box)")
    axs[0].plot(300, 90, "*", color="lime", ms=12, mec="black")
    axs[0].add_patch(mpl.patches.Rectangle((180, 20), 200, 220, fill=False, color="cyan", lw=1.5))
    for ax, pred, m, t in ((axs[1], point_pred, best, "point"), (axs[2], box_pred, 0, "box")):
        show(ax, image, f"{title}, {t} prompt (predicted IoU {pred.scores[0, m]:.2f})")
        mask = pred.binary_masks[0, m].float()
        ax.imshow(tensor_to_image(torch.stack([mask * 0, mask, mask * 0.6, mask * 0.55])), interpolation="nearest")
    fig.tight_layout()
    save(fig, name)


@figure
def figure_segment_anything() -> None:
    _sam_figure("vit_b", "segment_anything", "SAM ViT-B")


@figure
def figure_mobile_sam() -> None:
    _sam_figure("mobile_sam", "mobile_sam", "MobileSAM")


@figure
def figure_efficient_vit() -> None:
    from kornia.models.efficient_vit import EfficientViT, EfficientViTConfig

    image = sample("panda.jpg", (224, 224))
    model = EfficientViT.from_config(EfficientViTConfig.from_pretrained("b1", 224)).eval()
    with torch.no_grad():
        feats = model(image)  # dict of feature maps: "input", "stage0" ... "stage_final"

    stages = [k for k in feats if k.startswith("stage")]
    fig, axs = plt.subplots(1, 1 + len(stages), figsize=(2.3 * (1 + len(stages)), 2.6))
    show(axs[0], image, "input 224x224")
    for ax, k in zip(axs[1:], stages):
        f = feats[k][0]
        ax.imshow(f.abs().mean(0), cmap="magma")
        ax.set_title(f"{k}: {tuple(f.shape)}", fontsize=8)
        ax.set_axis_off()
    fig.suptitle("EfficientViT-B1 pyramid, mean |activation| per stage", fontsize=10)
    save(fig, "efficient_vit")


# --------------------------------------------------------------------------- classification backbones


@figure
def figure_tiny_vit() -> None:
    from kornia.models.tiny_vit import TinyViT

    image = sample("panda.jpg", (224, 224))
    model = TinyViT.from_config("5m", pretrained=True).eval()  # ImageNet-1k weights
    with torch.no_grad():
        logits = model(image)  # (1, 1000)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), width_ratios=[1, 1.6])
    show(axs[0], image, "input 224x224")
    topk_bars(axs[1], logits, imagenet_classes(), "TinyViT-5M top-5 ImageNet classes")
    fig.tight_layout()
    save(fig, "tiny_vit")


@figure
def figure_vit() -> None:
    from kornia.models.vit import VisionTransformer

    image = sample("panda.jpg", (224, 224))
    vit = VisionTransformer.from_config("vit_b/16", pretrained=True).eval()  # AugReg ImageNet-21k weights
    with torch.no_grad():
        tokens = vit(image)  # (1, 197, 768): class token + 14x14 patch tokens

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), width_ratios=[1, 1, 1.3])
    show(axs[0], image, "input 224x224 → 14x14 patches of 16 px")
    for p in range(0, 224, 16):
        axs[0].axhline(p - 0.5, color="white", lw=0.4, alpha=0.7)
        axs[0].axvline(p - 0.5, color="white", lw=0.4, alpha=0.7)
    axs[1].imshow(tokens[0, 1:].norm(dim=-1).view(14, 14), cmap="viridis")
    axs[1].set_title("‖patch token‖ on the 14x14 grid", fontsize=10)
    axs[1].set_axis_off()
    axs[2].imshow(tokens[0, :, :96].T, aspect="auto", cmap="coolwarm")
    axs[2].set_title("output (1, 197, 768): first 96 dims of each token", fontsize=10)
    axs[2].set_xlabel("token (0 = class token)")
    axs[2].set_ylabel("embedding dim")
    fig.suptitle("VisionTransformer vit_b/16 with the pretrained AugReg ImageNet-21k weights", fontsize=10)
    fig.tight_layout()
    save(fig, "vit")


@figure
def figure_vit_mobile() -> None:
    from kornia.models.vit_mobile import MobileViT

    image = sample("panda.jpg", (256, 256))
    mvit = MobileViT(mode="xxs").eval()  # random init: no pretrained weights
    with torch.no_grad():
        feats = mvit(image)  # (1, 320, 8, 8) feature map

    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
    show(axs[0], image, "input 256x256")
    axs[1].imshow(feats[0].abs().mean(0), cmap="magma")
    axs[1].set_title(f"output {tuple(feats.shape)}: mean |activation|", fontsize=10)
    axs[1].set_axis_off()
    fig.suptitle("MobileViT-XXS with random initialisation (stride-32 feature map)", fontsize=10)
    fig.tight_layout()
    save(fig, "vit_mobile")


# --------------------------------------------------------------------------- local features and matching


def _draw_matches(ax, img1, img2, pts1, pts2, title: str, max_lines: int = 150) -> None:
    w = img1.shape[-1]
    show(ax, torch.cat([img1, img2], dim=-1), title)
    idx = torch.randperm(len(pts1))[:max_lines]
    for (x1, y1), (x2, y2) in zip(pts1[idx].tolist(), pts2[idx].tolist()):
        ax.plot([x1, x2 + w], [y1, y2], color="lime", lw=0.4, alpha=0.8)
    ax.plot(pts1[:, 0], pts1[:, 1], ".", color="yellow", ms=1.2)
    ax.plot(pts2[:, 0] + w, pts2[:, 1], ".", color="yellow", ms=1.2)


@figure
def figure_loftr() -> None:
    from kornia.feature import LoFTR

    img1, img2 = knchurch()  # two (1, 3, H, W) views of the same scene
    matcher = LoFTR(pretrained="outdoor").eval()
    with torch.no_grad():
        out = matcher({"image0": K.color.rgb_to_grayscale(img1), "image1": K.color.rgb_to_grayscale(img2)})
    pts1, pts2, conf = out["keypoints0"], out["keypoints1"], out["confidence"]

    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_matches(
        ax, img1, img2, pts1, pts2, f"LoFTR (outdoor): {len(pts1)} matches, mean confidence {conf.mean():.2f}"
    )
    save(fig, "loftr")


@figure
def figure_sold2() -> None:
    from kornia.feature import SOLD2

    img1, img2 = knchurch()
    gray = K.color.rgb_to_grayscale(torch.cat([img1, img2]))
    sold2 = SOLD2(pretrained=True).eval()
    with torch.no_grad():
        out = sold2(gray)  # line_segments: list of (N, 2, 2) in (y, x); dense_desc: (B, 128, H/4, W/4)
        lines1, lines2 = out["line_segments"]
        matches = sold2.match(
            lines1, lines2, out["dense_desc"][0:1], out["dense_desc"][1:2]
        )  # (N1,) index into lines2, -1 = none

    valid = matches != -1
    m1, m2 = lines1[valid], lines2[matches[valid]]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), width_ratios=[1, 1, 2])
    for ax, img, lines, t in (
        (axs[0], img1, lines1, "SOLD2 lines, image 1"),
        (axs[1], img2, lines2, "SOLD2 lines, image 2"),
    ):
        show(ax, img, f"{t} ({len(lines)})")
        for (y1, x1), (y2, x2) in lines.tolist():
            ax.plot([x1, x2], [y1, y2], color="lime", lw=0.8)
    w = img1.shape[-1]
    show(axs[2], torch.cat([img1, img2], dim=-1), f"{int(valid.sum())} matched lines")
    colors = plt.cm.hsv(torch.linspace(0, 1, len(m1)).numpy())
    for c, ((ay, ax_), (by, bx)), ((cy, cx), (dy, dx)) in zip(colors, m1.tolist(), m2.tolist()):
        axs[2].plot([ax_, bx], [ay, by], color=c, lw=1.2)
        axs[2].plot([cx + w, dx + w], [cy, dy], color=c, lw=1.2)
    save(fig, "sold2")


@figure
def figure_hardnet() -> None:
    from kornia.feature import HardNet, KeyNetDetector, extract_patches_from_pyramid, match_snn

    img1, img2 = knchurch()
    gray1, gray2 = K.color.rgb_to_grayscale(img1), K.color.rgb_to_grayscale(img2)
    detector = KeyNetDetector(pretrained=True, num_features=1500).eval()
    hardnet = HardNet(pretrained=True).eval()
    with torch.no_grad():
        lafs1, _ = detector(gray1)
        lafs2, _ = detector(gray2)
        patches1 = extract_patches_from_pyramid(gray1, lafs1)[0]  # (N, 1, 32, 32)
        patches2 = extract_patches_from_pyramid(gray2, lafs2)[0]
        desc1, desc2 = hardnet(patches1), hardnet(patches2)  # (N, 128), L2-normalised
        _, idx = match_snn(desc1, desc2, th=0.9)

    pts1 = K.feature.get_laf_center(lafs1)[0, idx[:, 0]]
    pts2 = K.feature.get_laf_center(lafs2)[0, idx[:, 1]]
    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tensor_to_image(torch.cat(list(patches1[:8]), dim=-1)), cmap="gray")
    ax.set_title("8 of the 1500 KeyNet patches (32x32)", fontsize=10)
    ax.set_axis_off()
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(desc1[:8], aspect="auto", cmap="coolwarm")
    ax.set_title("their HardNet descriptors (8 x 128)", fontsize=10)
    ax.set_yticks([])
    ax = fig.add_subplot(gs[:, 1])
    _draw_matches(ax, img1, img2, pts1, pts2, f"KeyNet + HardNet + SNN matching: {len(idx)} matches")
    fig.tight_layout()
    save(fig, "hardnet")


@figure
def figure_affnet() -> None:
    from kornia.feature import KeyNetDetector, LAFAffNetShapeEstimator, laf_to_boundary_points

    img1, _ = knchurch()
    gray = K.color.rgb_to_grayscale(img1)
    detector = KeyNetDetector(pretrained=True, num_features=32).eval()
    affnet = LAFAffNetShapeEstimator(pretrained=True).eval()
    with torch.no_grad():
        lafs, _ = detector(gray)  # (1, N, 2, 3) circular local affine frames
        lafs_affine = affnet(lafs, gray)  # (1, N, 2, 3) affine-covariant frames

    h, w = gray.shape[-2:]
    fig, axs = plt.subplots(1, 2, figsize=(7, 4.4))
    panels = ((axs[0], lafs, "KeyNet LAFs (isotropic)"), (axs[1], lafs_affine, "after AffNet (affine shape)"))
    for ax, frames, t in panels:
        show(ax, img1, t)
        for pts in laf_to_boundary_points(frames)[0].tolist():
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="lime", lw=1.0)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    save(fig, "affnet")


# --------------------------------------------------------------------------- edges and enhance


@figure
def figure_dexined() -> None:
    from kornia.contrib.edge_detection import EdgeDetectorBuilder

    image = sample("girona.png")
    detector = EdgeDetectorBuilder.build("dexined", pretrained=True, image_size=352)  # resize + normalise + sigmoid
    with torch.no_grad():
        edges = detector(image)[0]  # list of (1, 1, H, W) edge probabilities in [0, 1], one per input image

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    show(axs[0], image, "input")
    show(axs[1], 1 - edges, "DexiNed edge probability (dark = edge)")
    save(fig, "dexined")


@figure
def figure_defmo() -> None:
    from kornia.feature import DeFMO

    # Synthesise a striped ball flying across a static background: the blurred frame is the
    # mean of its 24 positions, exactly the image-formation model DeFMO inverts.
    background = sample("girona.png", (240, 320))
    h, w = background.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    orange = torch.tensor([1.0, 0.6, 0.1])[None, :, None, None]
    navy = torch.tensor([0.05, 0.05, 0.25])[None, :, None, None]
    frames = []
    for t in torch.linspace(0, 1, 24):
        cx, cy = 80 + 160 * t, 150 - 60 * t
        alpha = (((xx - cx) ** 2 + (yy - cy) ** 2) < 42**2).float()[None, None]
        stripe = (((xx - cx) + (yy - cy)).abs() < 10).float()[None, None]
        frames.append(alpha * (orange * (1 - stripe) + navy * stripe) + (1 - alpha) * background)
    blurred = torch.stack(frames).mean(0)  # (1, 3, H, W)

    defmo = DeFMO(pretrained=True).eval()
    with torch.no_grad():
        subframes = defmo(torch.cat([blurred, background], dim=1))  # (1, 24, 4, H, W) RGBA sub-frames

    fig, axs = plt.subplots(2, 4, figsize=(12, 4.8))
    show(axs[0, 0], blurred, "input: motion-blurred frame")
    show(axs[0, 1], background, "input: background estimate")
    show(axs[0, 2], frames[0], "ground truth, first sub-frame")
    show(axs[0, 3], frames[-1], "ground truth, last sub-frame")
    for ax, i in zip(axs[1], (0, 8, 16, 23)):
        rgba = subframes[0, i]
        composed = rgba[3:] * rgba[:3] + (1 - rgba[3:]) * background[0]  # alpha-blend onto the background
        show(ax, composed, f"DeFMO sub-frame {i + 1}/24")
    fig.tight_layout()
    save(fig, "defmo")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    unknown = [n for n in wanted if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {sorted(FIGURES)}")
    for n in wanted:
        print(f"== {n}")
        FIGURES[n]()
