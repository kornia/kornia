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

"""ONNX export survey — area `models` (kornia.models, kornia.contrib, kornia.tracking, kornia.sensors, kornia.image)."""

from __future__ import annotations

import sys
import traceback

import torch
from harness import case, run_cases

import kornia as K

torch.manual_seed(0)
torch.set_num_threads(4)

# ----------------------------------------------------------------------------- helpers


def ev(m):
    """Return module in eval mode (lambda targets are not put in eval() by the harness)."""
    return m.eval()


def build(ctor):
    """Build lazily-failing objects: if the constructor raises, return a callable that re-raises (-> eager-fail)."""
    try:
        return ctor(), None
    except Exception as e:
        tb = traceback.format_exc()
        return None, (e, tb)


def raiser(exc):
    def _f(*a, **k):
        raise exc[0]

    return _f


def reuse(obj, err):
    """Constructor returning an already-built object, re-raising its build error if any."""

    def _c():
        if err is not None:
            raise err[0]
        return obj

    return _c


def mcase(name, group, ctor, call, inputs, kwargs=None, **kw):
    """Case whose target is `call(obj)` where obj = ctor(); constructor errors surface as eager-fail."""
    obj, err = build(ctor)
    if err is not None:
        kw["note"] = (kw.get("note", "") + f" | CONSTRUCTOR FAILED: {type(err[0]).__name__}: {err[0]}").strip(" |")
        return case(name, group, raiser(err), inputs, kwargs, **kw)
    return case(name, group, call(obj), inputs, kwargs, **kw)


IMG = torch.rand(1, 3, 64, 96)
IMG2 = torch.rand(2, 3, 48, 64)
IMG32 = torch.rand(1, 3, 224, 224)

CASES: list[dict] = []
A = CASES.append

# ============================================================================= kornia.models.rt_detr
from kornia.models.rt_detr import RTDETR, DETRPostProcessor, RTDETRConfig, RTDETRModelType  # noqa: E402

rtdetr_pre, _e = build(lambda: ev(RTDETR.from_pretrained("rtdetr_r18vd")))
A(
    mcase(
        "models.rt_detr.RTDETR[r18vd,pretrained]",
        "models.rt_detr",
        reuse(rtdetr_pre, _e),
        lambda m: m,
        [torch.rand(1, 3, 256, 320)],
        note="weights rtdetr_r18vd (HF kornia/rt_detr); outputs (logits (1,300,80), boxes (1,300,4)); 256x320 input",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.rt_detr.RTDETR[r18vd,random,160x192]",
        "models.rt_detr",
        lambda: ev(RTDETR.from_name("rtdetr_r18vd", num_classes=80)),
        lambda m: m,
        [torch.rand(1, 3, 160, 192)],
        note="random init via RTDETR.from_name; 160x192 input. NOTE: 64x96 fails eagerly ('selected "
        "index k out of range'): "
        "the decoder topk selects 300 queries so the input must yield >=300 encoder tokens (>= ~128x160)",
        tags=("model",),
    )
)
A(
    mcase(
        "models.rt_detr.RTDETR.from_config[r18vd,num_classes=3]",
        "models.rt_detr",
        lambda: ev(RTDETR.from_config(RTDETRConfig(RTDETRModelType.resnet18d, 3))),
        lambda m: m,
        [torch.rand(1, 3, 160, 192)],
        note="RTDETRConfig(RTDETRModelType.resnet18d, num_classes=3); random init (str model_type "
        "'rtdetr_r18vd' is rejected: AttributeError)",
        tags=("model",),
    )
)
A(case("models.rt_detr.RTDETRConfig", "models.rt_detr", None, [], skip="config dataclass, not a callable graph"))
A(case("models.rt_detr.RTDETRModelType", "models.rt_detr", None, [], skip="enum"))
_logits = torch.randn(1, 300, 80)
_boxes = torch.rand(1, 300, 4) * 0.5 + 0.25
_sizes = torch.tensor([[64.0, 96.0]])
A(
    case(
        "models.rt_detr.DETRPostProcessor[confidence_filtering=False]",
        "models.rt_detr",
        DETRPostProcessor(num_classes=80, num_top_queries=300, confidence_filtering=False),
        [_logits, _boxes, _sizes],
        note="num_top_queries=300 baked; returns (1,300,6) tensor; uses topk + gather + mod",
    )
)
A(
    case(
        "models.rt_detr.DETRPostProcessor[threshold=0.3,filter_as_zero]",
        "models.rt_detr",
        DETRPostProcessor(confidence_threshold=0.3, num_classes=80, num_top_queries=300, filter_as_zero=True),
        [_logits, _boxes, _sizes],
        note="threshold baked as buffer; filtered rows zeroed -> fixed shape",
    )
)
A(
    case(
        "models.rt_detr.DETRPostProcessor[threshold=0.3,list output]",
        "models.rt_detr",
        DETRPostProcessor(confidence_threshold=0.3, num_classes=80, num_top_queries=300),
        [_logits, _boxes, _sizes],
        note="default: BoxFiltering returns Python list of (Di,6) tensors, data-dependent shapes",
    )
)

# ============================================================================= kornia.models.sam
from kornia.models.sam import Sam, SamConfig  # noqa: E402
from kornia.models.structures import SegmentationResults  # noqa: E402

A(
    case(
        "models.sam.Sam.forward",
        "models.sam",
        None,
        [],
        skip="forward(images, batched_prompts: list[dict], multimask_output: bool) -> list[SegmentationResults]; "
        "list/dict/bool inputs are not exportable — sub-modules exported as separate cases",
    )
)
A(case("models.sam.SamConfig", "models.sam", None, [], skip="config dataclass"))
A(case("models.sam.SamModelType", "models.sam", None, [], skip="enum"))
A(case("models.structures.Prompts", "models.sam", None, [], skip="plain dataclass holding tensors, no computation"))
A(
    case(
        "models.structures.SegmentationResults",
        "models.sam",
        SegmentationResults,
        [torch.randn(1, 3, 64, 96), torch.rand(1, 3)],
        note="dataclass container; `binary_masks` property = logits > 0",
    )
)
A(
    case(
        "models.structures.SegmentationResults.binary_masks",
        "models.sam",
        lambda lg, sc: SegmentationResults(lg, sc).binary_masks,
        [torch.randn(1, 3, 64, 96), torch.rand(1, 3)],
        note="mask_threshold=0.0 baked",
    )
)

sam_mobile, _e_sm = build(lambda: ev(Sam.from_config(SamConfig("mobile_sam", pretrained=True))))
sam_b, _e_sb = build(lambda: ev(Sam.from_config(SamConfig("vit_b", pretrained=False))))
A(
    case(
        "models.sam.Sam.image_encoder[vit_b,pretrained]",
        "models.sam",
        None,
        [],
        skip="weights 375 MB but ViT-B on a mandatory 1024x1024 input: random-init vit_b encoder exported instead "
        "(same graph); mobile_sam covers the pretrained path",
    )
)
A(case("models.sam.Sam.image_encoder[vit_l]", "models.sam", None, [], skip="weights > 500 MB (1.25 GB)"))
A(case("models.sam.Sam.image_encoder[vit_h]", "models.sam", None, [], skip="weights > 500 MB (2.56 GB)"))


def _sam_sub(model, err, attr, name, inputs, kwargs=None, **kw):
    if err is not None:
        kw["note"] = (kw.get("note", "") + f" | CONSTRUCTOR FAILED: {err[0]}").strip(" |")
        return case(name, "models.sam", raiser(err), inputs, kwargs, **kw)
    return case(name, "models.sam", getattr(model, attr), inputs, kwargs, **kw)


_img_emb = torch.randn(1, 256, 64, 64)
_sparse = torch.randn(1, 3, 256)
_dense = torch.randn(1, 256, 64, 64)
for tagname, mdl, err in (("mobile_sam", sam_mobile, _e_sm), ("vit_b,random", sam_b, _e_sb)):
    pe = None if err else mdl.prompt_encoder
    A(
        case(
            f"models.sam.Sam.prompt_encoder[points,{tagname}]",
            "models.sam",
            (raiser(err) if err else (lambda c, l, _pe=pe: _pe(points=(c, l), boxes=None, masks=None))),
            [torch.tensor([[[300.0, 400.0], [512.0, 512.0]]]), torch.tensor([[1, 0]])],
            note="points=(coords (1,2,2), labels (1,2) int64); pads with a -1 label point; "
            "boolean-mask indexed assignment "
            "`point_embedding[labels == -1] = 0.0`",
            tags=("model",) + (("pretrained",) if "mobile" in tagname else ()),
        )
    )
    A(
        case(
            f"models.sam.Sam.prompt_encoder[boxes,{tagname}]",
            "models.sam",
            (raiser(err) if err else (lambda b, _pe=pe: _pe(points=None, boxes=b, masks=None))),
            [torch.tensor([[100.0, 120.0, 600.0, 700.0]])],
            note="boxes (1,4) in 1024x1024 pixel coords",
            tags=("model",),
        )
    )
    A(
        case(
            f"models.sam.Sam.prompt_encoder[masks,{tagname}]",
            "models.sam",
            (raiser(err) if err else (lambda m, _pe=pe: _pe(points=None, boxes=None, masks=m))),
            [torch.randn(1, 1, 256, 256)],
            note="mask prompt (1,1,256,256) -> dense (1,256,64,64) via conv downscale",
            tags=("model",),
        )
    )
    A(
        case(
            f"models.sam.Sam.prompt_encoder.get_dense_pe[{tagname}]",
            "models.sam",
            (raiser(err) if err else (lambda d, _pe=pe: _pe.get_dense_pe())),
            [torch.zeros(1)],
            note="no tensor input: graph is a constant positional encoding (1,256,64,64)",
            tags=("model",),
        )
    )
    A(
        _sam_sub(
            mdl,
            err,
            "mask_decoder",
            f"models.sam.Sam.mask_decoder[multimask,{tagname}]",
            [_img_emb, _img_emb.clone(), _sparse, _dense],
            {"multimask_output": True},
            note="multimask_output=True baked; inputs image_embeddings/image_pe (1,256,64,64), sparse (1,3,256), "
            "dense (1,256,64,64); TwoWayTransformer",
            tags=("model",),
        )
    )
    A(
        _sam_sub(
            mdl,
            err,
            "mask_decoder",
            f"models.sam.Sam.mask_decoder[single,{tagname}]",
            [_img_emb, _img_emb.clone(), _sparse, _dense],
            {"multimask_output": False},
            note="multimask_output=False baked",
            tags=("model",),
        )
    )

# ============================================================================= kornia.models.sam3
from kornia.models.sam3 import ImageEncoderHiera  # noqa: E402

A(
    mcase(
        "models.sam3.ImageEncoderHiera[tiny,random]",
        "models.sam3",
        lambda: ev(ImageEncoderHiera(img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=2)),
        lambda m: m,
        [torch.rand(1, 3, 64, 64)],
        note="random init, img_size=64 (square, pos_embed fixed to img_size)",
        tags=("model",),
    )
)
A(
    case(
        "models.sam3.ImageEncoderHiera[default 1024]",
        "models.sam3",
        None,
        [],
        skip="default config = ViT-B at 1024x1024 random init; no weights loader exists in kornia; "
        "tiny variant exported",
    )
)

# ============================================================================= kornia.models.yunet
from kornia.models.yunet import YuNet  # noqa: E402
from kornia.models.yunet.processors import PriorBox, decode  # noqa: E402

A(
    mcase(
        "models.yunet.YuNet[test,pretrained]",
        "models.yunet",
        lambda: ev(YuNet("test", pretrained=True)),
        lambda m: m,
        [IMG],
        note="weights yunet_final.pth; dict outputs loc/conf/iou; H,W must be multiples of 32",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.yunet.YuNet[test,batch2]",
        "models.yunet",
        lambda: ev(YuNet("test", pretrained=True)),
        lambda m: m,
        [torch.rand(2, 3, 64, 128)],
        tags=("model", "pretrained", "batch>1"),
    )
)
A(
    case(
        "models.yunet.processors.PriorBox",
        "models.yunet",
        lambda d: PriorBox([[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]], [8, 16, 32, 64], False, (64, 96))(),
        [torch.zeros(1)],
        note="no tensor input: fully constant graph (priors for 64x96)",
    )
)
_pri = PriorBox([[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]], [8, 16, 32, 64], False, (64, 96))()
A(
    case(
        "models.yunet.processors.decode",
        "models.yunet",
        decode,
        [torch.randn(_pri.shape[0], 14) * 0.1, _pri],
        {"variances": [0.1, 0.2]},
        note="variances baked",
    )
)

# ============================================================================= kornia.models.dexined
from kornia.models.dexined import DexiNed  # noqa: E402

dexi, _e_dx = build(lambda: ev(DexiNed(pretrained=True)))
A(
    mcase(
        "models.dexined.DexiNed[pretrained]",
        "models.dexined",
        reuse(dexi, _e_dx),
        lambda m: m,
        [IMG],
        note="weights DexiNed_BIPED_10.pth; output (1,1,H,W)",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.dexined.DexiNed.get_features",
        "models.dexined",
        reuse(dexi, _e_dx),
        lambda m: m,
        [IMG],
        method="get_features",
        note="returns list of 6 side outputs",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.dexined.DexiNed[batch2,48x64]",
        "models.dexined",
        reuse(dexi, _e_dx),
        lambda m: m,
        [IMG2],
        tags=("model", "pretrained", "batch>1"),
    )
)

# ============================================================================= depth / segmentation / base
A(
    case(
        "models.depth_estimation.DepthEstimation",
        "models.depth_estimation",
        None,
        [],
        skip="HFONNXComunnityModel: loads a pre-built ONNX graph from HF onnx-community and runs it with onnxruntime; "
        "not a torch module",
    )
)
A(
    case(
        "models.depth_estimation.DepthAnythingONNXBuilder",
        "models.depth_estimation",
        None,
        [],
        skip="builder returns the onnx-community ONNX model wrapped in ONNXSequential; nothing to export",
    )
)
from kornia.models.segmentation import SegmentationModelsBuilder, SemanticSegmentation  # noqa: E402

A(
    case(
        "models.segmentation.SegmentationModelsBuilder.build",
        "models.segmentation",
        lambda d: SegmentationModelsBuilder.build("Unet", "resnet34", encoder_weights=None),
        [IMG],
        note="segmentation_models_pytorch not installed -> ImportError expected; even with it installed the builder "
        "instantiates abstract SemanticSegmentation (no from_config, no __init__) -> TypeError (kornia bug)",
    )
)
A(
    case(
        "models.segmentation.SemanticSegmentation",
        "models.segmentation",
        lambda d: SemanticSegmentation(
            model=torch.nn.Identity(), pre_processor=torch.nn.Identity(), post_processor=torch.nn.Identity(), name="x"
        ),
        [IMG],
        note="KORNIA BUG: abstract class (ModelBase.from_config not implemented) -> cannot be instantiated",
    )
)
A(case("models.base.ModelBase", "models.base", None, [], skip="abstract base class (from_config abstract)"))
A(
    case(
        "models.base.ModelBase.save/load/resize",
        "models.base",
        None,
        [],
        skip="file-IO / PIL helpers on ModelBase, not tensor graphs",
    )
)
A(
    case(
        "models._hf_models.HFONNXComunnityModel",
        "models.base",
        None,
        [],
        skip="ORT wrapper around downloaded onnx-community graphs; not a torch module",
    )
)

# ============================================================================= efficient_vit
from kornia.models.efficient_vit import EfficientViT, EfficientViTConfig  # noqa: E402
from kornia.models.efficient_vit.backbone import (  # noqa: E402
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_l0,
)

A(
    mcase(
        "models.efficient_vit.EfficientViT[b1-r224,pretrained]",
        "models.efficient_vit",
        lambda: ev(EfficientViT.from_config(EfficientViTConfig.from_pretrained("b1", 224))),
        lambda m: m,
        [IMG],
        note="weights HF kornia/efficientvit_imagenet_b1_r224 (b1-r224.pt); dict of stage features; 64x96 input",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.efficient_vit.backbone.efficientvit_backbone_b0",
        "models.efficient_vit",
        lambda: ev(efficientvit_backbone_b0()),
        lambda m: m,
        [IMG],
        note="random init; dict output",
        tags=("model",),
    )
)
A(
    mcase(
        "models.efficient_vit.backbone.efficientvit_backbone_b1",
        "models.efficient_vit",
        lambda: ev(efficientvit_backbone_b1()),
        lambda m: m,
        [IMG],
        note="random init",
        tags=("model",),
    )
)
A(
    mcase(
        "models.efficient_vit.backbone.efficientvit_backbone_l0",
        "models.efficient_vit",
        lambda: ev(efficientvit_backbone_l0()),
        lambda m: m,
        [IMG],
        note="random init (large family)",
        tags=("model",),
    )
)
for _n in ("b2", "b3", "l1", "l2", "l3"):
    A(
        case(
            f"models.efficient_vit.backbone.efficientvit_backbone_{_n}",
            "models.efficient_vit",
            None,
            [],
            skip=f"same architecture family as b0/b1/l0 with more channels/blocks; {_n} not exported separately",
        )
    )
A(
    case(
        "models.efficient_vit.EfficientViTBackbone",
        "models.efficient_vit",
        None,
        [],
        skip="class instantiated by efficientvit_backbone_b* factories (covered)",
    )
)
A(
    case(
        "models.efficient_vit.EfficientViTLargeBackbone",
        "models.efficient_vit",
        None,
        [],
        skip="class instantiated by efficientvit_backbone_l* factories (covered)",
    )
)

# ============================================================================= tiny_vit / vit / vit_mobile
from kornia.models.tiny_vit import TinyViT  # noqa: E402
from kornia.models.vit import VisionTransformer  # noqa: E402
from kornia.models.vit_mobile import MobileViT  # noqa: E402

A(
    mcase(
        "models.tiny_vit.TinyViT[5m,in1k]",
        "models.tiny_vit",
        lambda: ev(TinyViT.from_config("5m", pretrained="in1k")),
        lambda m: m,
        [IMG32],
        note="weights tiny_vit_5m_224 in1k (HF kornia/tiny_vit); img_size=224 baked (window attention)",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.tiny_vit.TinyViT[5m,mobile_sam,random,img_size=64]",
        "models.tiny_vit",
        lambda: ev(TinyViT.from_config("5m", img_size=64, mobile_sam=True)),
        lambda m: m,
        [torch.rand(1, 3, 64, 64)],
        note="mobile_sam=True head: unflatten + neck -> (1,256,4,4)",
        tags=("model",),
    )
)
A(
    case(
        "models.tiny_vit.TinyViT[non-square input]",
        "models.tiny_vit",
        None,
        [],
        skip="eager RuntimeError (view '[1,8,8,64]' invalid) — TinyViT hard-codes H=W=img_size in its window attention",
    )
)
A(case("contrib.TinyViT", "contrib", None, [], skip="alias of kornia.models.tiny_vit.TinyViT (covered)"))
A(
    mcase(
        "models.vit.VisionTransformer[small,random]",
        "models.vit",
        lambda: ev(VisionTransformer(image_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=2)),
        lambda m: m,
        [torch.rand(1, 3, 32, 32)],
        note="random init; image_size=32 patch=8; output (1,17,64)",
        tags=("model",),
    )
)
A(
    case(
        "models.vit.VisionTransformer[non-square input]",
        "models.vit",
        None,
        [],
        skip="eager RuntimeError (pos-embed 17 vs 25 tokens) — VisionTransformer requires "
        "H=W=image_size (fixed pos_embed)",
    )
)
A(
    case(
        "models.vit.VisionTransformer.from_config[vit_b/16]",
        "models.vit",
        None,
        [],
        skip="86M-param random init (no pretrained weights shipped); small config covers the graph",
    )
)
A(
    mcase(
        "models.vit_mobile.MobileViT[xxs,random,64x128]",
        "models.vit_mobile",
        lambda: ev(MobileViT(mode="xxs")),
        lambda m: m,
        [torch.rand(1, 3, 64, 128)],
        note="random init; H,W must be multiples of 64 (64x96 fails eagerly: view '[96,2,1,2]' "
        "invalid — stride-32 grid "
        "must be divisible by patch_size 2)",
        tags=("model",),
    )
)
A(
    mcase(
        "models.vit_mobile.MobileViT[xxs,random,256]",
        "models.vit_mobile",
        lambda: ev(MobileViT(mode="xxs")),
        lambda m: m,
        [torch.rand(1, 3, 256, 256)],
        note="documented input size 256x256",
        tags=("model",),
    )
)

# ============================================================================= small_sr
from kornia.models.small_sr import SmallSRNet, SmallSRNetWrapper  # noqa: E402

A(
    mcase(
        "models.small_sr.SmallSRNet[x3,pretrained]",
        "models.small_sr",
        lambda: ev(SmallSRNet(3, pretrained=True)),
        lambda m: m,
        [torch.rand(1, 1, 16, 24)],
        note="weights small_sr.pth; Y channel (1,1,H,W) -> (1,1,3H,3W) PixelShuffle",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "models.small_sr.SmallSRNetWrapper[x3,pretrained]",
        "models.small_sr",
        lambda: ev(SmallSRNetWrapper(3, pretrained=True)),
        lambda m: m,
        [torch.rand(1, 3, 16, 24)],
        note="RGB->YCbCr, SR on Y, bicubic upsample CbCr, ->RGB",
        tags=("model", "pretrained"),
    )
)
A(
    case(
        "contrib.SuperResolution",
        "contrib",
        lambda x: K.contrib.SuperResolution(
            model=torch.nn.Identity(), pre_processor=torch.nn.Identity(), post_processor=torch.nn.Identity(), name="x"
        ),
        [IMG],
        note="KORNIA BUG: abstract (ModelBase.from_config not implemented, no __init__) -> TypeError",
    )
)
A(
    case(
        "contrib.SmallSRBuilder.build",
        "contrib",
        lambda x: K.contrib.SmallSRBuilder.build("small_sr", pretrained=False),
        [IMG],
        note="KORNIA BUG: builder instantiates abstract SuperResolution -> TypeError at super_resolution.py:262",
    )
)
A(
    case(
        "contrib.RRDBNetBuilder.build",
        "contrib",
        None,
        [],
        skip="builds a 17M-parameter Real-ESRGAN generator and would hit the same abstract SuperResolution TypeError",
    )
)

# ============================================================================= processors
from kornia.models.processors import OutputRangePostProcessor, ResizePostProcessor, ResizePreProcessor  # noqa: E402
from kornia.models.processors.naflex import NaFlex  # noqa: E402

A(
    case(
        "models.processors.ResizePreProcessor",
        "models.processors",
        ResizePreProcessor(32, 40),
        [IMG],
        note="size (32,40) baked; returns (resized, original_sizes); per-image Python loop",
    )
)
A(
    case(
        "models.processors.ResizePreProcessor[batch2]",
        "models.processors",
        ResizePreProcessor(32, 40),
        [IMG2],
        tags=("batch>1",),
    )
)
A(
    case(
        "models.processors.ResizePostProcessor",
        "models.processors",
        ResizePostProcessor(),
        [torch.rand(1, 1, 32, 40), torch.tensor([[64, 96]])],
        note="original_sizes via .cpu().numpy().tolist(); returns inputs unchanged when torch.onnx.is_in_onnx_export()",
    )
)
A(
    case(
        "models.processors.OutputRangePostProcessor",
        "models.processors",
        OutputRangePostProcessor(0.0, 1.0),
        [torch.randn(1, 3, 16, 24)],
        note="clamp(0,1) baked",
    )
)
_pe_conv = torch.nn.Conv2d(3, 8, 8, 8)
A(
    case(
        "models.processors.NaFlex[interpolated pos]",
        "models.processors",
        NaFlex(_pe_conv, torch.randn(16, 8)),
        [torch.rand(1, 3, 32, 48)],
        note="pos-embed 4x4 grid interpolated to 4x6 with F.interpolate (bilinear); patch fn is a Conv2d",
    )
)
A(
    case(
        "models.processors.NaFlex[matching pos]",
        "models.processors",
        NaFlex(_pe_conv, torch.randn(16, 8)),
        [torch.rand(1, 3, 32, 32)],
        note="num_patches == pos rows -> plain add",
    )
)

# ============================================================================= VLM re-implementations (pure PyTorch)
from kornia.models.kimi_vl import KimiVLConfig, KimiVLModel, KimiVLProjectorConfig, MoonViT, MoonViTConfig  # noqa: E402
from kornia.models.paligemma import PaliGemma, PaliGemmaConfig  # noqa: E402
from kornia.models.qwen25 import Qwen2VLVisionTransformer  # noqa: E402
from kornia.models.siglip2 import SigLip2Config, SigLip2ImagePreprocessor, SigLip2Model  # noqa: E402
from kornia.models.siglip2.config import SigLip2TextConfig, SigLip2VisionConfig  # noqa: E402
from kornia.models.smolvlm2 import SmolVLM2  # noqa: E402

_vcfg = SigLip2VisionConfig(
    image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
)
_tcfg = SigLip2TextConfig(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=16,
)
_sig, _e_sig = build(lambda: ev(SigLip2Model(SigLip2Config(_vcfg, _tcfg, projection_dim=32))))
A(
    case(
        "models.siglip2.SigLip2Model[image only,tiny,random]",
        "models.vlm",
        raiser(_e_sig) if _e_sig else (lambda px: _sig(pixel_values=px)),
        [torch.rand(1, 3, 32, 32)],
        note="pure-PyTorch SigLIP2 (no transformers); tiny random config; SigLip2Result dataclass, None fields dropped",
        tags=("model",),
    )
)
A(
    case(
        "models.siglip2.SigLip2Model[image+text,tiny,random]",
        "models.vlm",
        raiser(_e_sig) if _e_sig else (lambda px, ids: _sig(pixel_values=px, input_ids=ids)),
        [torch.rand(2, 3, 32, 32), torch.randint(0, 100, (2, 8))],
        note="image+text -> embeds + logits_per_image/text",
        tags=("model", "batch>1"),
    )
)
A(
    case(
        "models.siglip2.SigLip2Model.get_image_features[non-square]",
        "models.vlm",
        None,
        [],
        skip="eager RuntimeError (pos-embed 4 vs 6 patches) — vision tower requires H=W=image_size "
        "(no pos-embed interpolation)",
    )
)
A(
    case(
        "models.siglip2.SigLip2ImagePreprocessor",
        "models.vlm",
        SigLip2ImagePreprocessor((32, 40)),
        [torch.rand(1, 3, 48, 64) * 255],
        note="Rescale(1/255) + Resize bicubic antialias + Normalize; size baked",
    )
)
A(
    case(
        "models.siglip2.SigLip2Builder.from_pretrained_hf",
        "models.vlm",
        None,
        [],
        skip="downloads google/siglip2-base-patch16-224 safetensors (~1.5 GB) from HF; weights > 500 MB",
    )
)
A(
    case(
        "models.siglip2.SigLip2Builder.from_name",
        "models.vlm",
        None,
        [],
        skip="random-init base config (~375M params); tiny random config covers the same graph",
    )
)
A(case("models.siglip2.SigLip2Config/SigLip2Result", "models.vlm", None, [], skip="config / result dataclasses"))

_mcfg = MoonViTConfig(
    image_size=28, patch_size=14, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
)
A(
    mcase(
        "models.kimi_vl.MoonViT[tiny,random]",
        "models.vlm",
        lambda: ev(MoonViT(_mcfg)),
        lambda m: m,
        [torch.rand(1, 3, 28, 42)],
        note="pure-PyTorch MoonViT; 2x3 patches vs 2x2 pos-embed -> interpolation path; 2D RoPE",
        tags=("model",),
    )
)
A(
    mcase(
        "models.kimi_vl.KimiVLModel[tiny,random]",
        "models.vlm",
        lambda: ev(KimiVLModel(KimiVLConfig(_mcfg, KimiVLProjectorConfig(32, 64, 48)))),
        lambda m: m,
        [torch.rand(1, 3, 28, 28)],
        note="MoonViT + pixel-unshuffle projector; h,w patch grid from input shape",
        tags=("model",),
    )
)
A(
    case(
        "models.kimi_vl.KimiVLBuilder",
        "models.vlm",
        None,
        [],
        skip="weights > 500 MB (Kimi-VL-A3B-Instruct vision tower + projector safetensors, multi-GB download)",
    )
)
A(
    case(
        "models.kimi_vl.KimiVLProjector",
        "models.vlm",
        None,
        [],
        skip="covered inside KimiVLModel case (needs h,w ints)",
    )
)
A(
    mcase(
        "models.qwen25.Qwen2VLVisionTransformer[tiny,random]",
        "models.vlm",
        lambda: ev(Qwen2VLVisionTransformer(embed_dim=32, depth=1, num_heads=2)),
        lambda m: m,
        [torch.rand(1, 3, 28, 42)],
        note="pure-PyTorch skeleton; conv14 patch embed + rotary attention blocks",
        tags=("model",),
    )
)
_pcfg = PaliGemmaConfig(
    vision_config=_vcfg,
    vocab_size=100,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    max_position_embeddings=64,
)
A(
    mcase(
        "models.paligemma.PaliGemma[tiny,random]",
        "models.vlm",
        lambda: ev(PaliGemma(_pcfg)),
        lambda m: m,
        [torch.randint(0, 100, (1, 4)), torch.rand(1, 3, 32, 32)],
        note="pure-PyTorch SigLIP2 tower + Gemma decoder (RMSNorm, RoPE, GQA); inputs (input_ids, pixel_values)",
        tags=("model",),
    )
)
A(
    mcase(
        "models.smolvlm2.SmolVLM2",
        "models.vlm",
        lambda: ev(SmolVLM2(16, 16)),
        lambda m: m,
        [torch.rand(1, 5, 16), torch.rand(1, 5, 16)],
        note="placeholder scaffold: sum of two Linear projections; image/text sequences must have equal length",
        tags=("model",),
    )
)

# ============================================================================= kornia.contrib — patches
_x = torch.rand(1, 2, 32, 48)
A(
    case(
        "contrib.extract_tensor_patches",
        "contrib.patches",
        K.contrib.extract_tensor_patches,
        [_x],
        {"window_size": 8, "stride": 8},
        note="window 8, stride 8 baked -> (1,24,2,8,8)",
    )
)
A(
    case(
        "contrib.extract_tensor_patches[rect window,stride,padding]",
        "contrib.patches",
        K.contrib.extract_tensor_patches,
        [_x],
        {"window_size": (8, 12), "stride": (4, 6), "padding": (2, 3)},
        note="non-square window/stride, padding (2,3)",
    )
)
A(
    case(
        "contrib.extract_tensor_patches[allow_auto_padding]",
        "contrib.patches",
        K.contrib.extract_tensor_patches,
        [_x],
        {"window_size": 10, "stride": 7, "allow_auto_padding": True},
        note="auto padding computed from static shape",
    )
)
A(
    case(
        "contrib.extract_tensor_patches[batch2]",
        "contrib.patches",
        K.contrib.extract_tensor_patches,
        [IMG2],
        {"window_size": 16, "stride": 16},
        tags=("batch>1",),
    )
)
A(case("contrib.ExtractTensorPatches", "contrib.patches", K.contrib.ExtractTensorPatches(8, stride=4, padding=2), [_x]))
_patches = K.contrib.extract_tensor_patches(_x, window_size=8, stride=8)
A(
    case(
        "contrib.combine_tensor_patches",
        "contrib.patches",
        K.contrib.combine_tensor_patches,
        [_patches],
        {"original_size": (32, 48), "window_size": 8, "stride": 8},
        note="original_size/window/stride baked; fold + overlap norm",
    )
)
_patches_o = K.contrib.extract_tensor_patches(_x, window_size=(8, 12), stride=(4, 6), padding=(2, 3))
A(
    case(
        "contrib.combine_tensor_patches[overlap,unpadding]",
        "contrib.patches",
        K.contrib.combine_tensor_patches,
        [_patches_o],
        {"original_size": (32, 48), "window_size": (8, 12), "stride": (4, 6), "unpadding": (2, 3)},
        note="overlapping windows: divides by ones-fold count (eps=1e-8)",
    )
)
_patches_a = K.contrib.extract_tensor_patches(_x, window_size=10, stride=7, allow_auto_padding=True)
A(
    case(
        "contrib.combine_tensor_patches[allow_auto_unpadding]",
        "contrib.patches",
        K.contrib.combine_tensor_patches,
        [_patches_a],
        {"original_size": (32, 48), "window_size": 10, "stride": 7, "allow_auto_unpadding": True},
    )
)
A(
    case(
        "contrib.CombineTensorPatches",
        "contrib.patches",
        K.contrib.CombineTensorPatches((32, 48), 8, stride=8),
        [_patches],
    )
)
A(
    case(
        "contrib.compute_padding",
        "contrib.patches",
        lambda d: K.contrib.compute_padding((32, 48), 10, 7),
        [torch.zeros(1)],
        note="returns tuple of Python ints -> expected no-tensor-output",
    )
)

# ============================================================================= kornia.contrib — tensor ops
_bin = (torch.rand(1, 1, 16, 24) > 0.6).float()
A(
    case(
        "contrib.connected_components",
        "contrib",
        K.contrib.connected_components,
        [_bin],
        {"num_iterations": 50},
        note="num_iterations=50 baked (Python for-loop unrolled into 50 max_pool2d stages); input 0/1 float mask",
    )
)
A(
    case(
        "contrib.connected_components[batch2,default 100 it]",
        "contrib",
        K.contrib.connected_components,
        [(torch.rand(2, 1, 12, 20) > 0.6).float()],
        note="100 unrolled iterations",
        tags=("batch>1",),
    )
)
A(
    case(
        "contrib.distance_transform",
        "contrib",
        K.contrib.distance_transform,
        [_bin],
        note="kernel_size=3, h=0.35 baked; Python loop over max(H,W)//2 iterations unrolled",
    )
)
A(
    case(
        "contrib.distance_transform[kernel_size=5]", "contrib", K.contrib.distance_transform, [_bin], {"kernel_size": 5}
    )
)
A(
    case(
        "contrib.distance_transform[3d]",
        "contrib",
        K.contrib.distance_transform,
        [(torch.rand(1, 1, 6, 10, 12) > 0.7).float()],
        note="3D input path",
        tags=("3d",),
    )
)
A(case("contrib.DistanceTransform", "contrib", K.contrib.DistanceTransform(kernel_size=3, h=0.35), [_bin]))
A(
    case(
        "contrib.diamond_square",
        "contrib",
        lambda d: K.contrib.diamond_square((1, 1, 16, 24), roughness=0.5, random_scale=1.0),
        [torch.zeros(1)],
        check=False,
        note="no tensor input; torch.rand inside graph (RandomUniformLike); output_size baked",
    )
)
A(
    case(
        "contrib.diamond_square[normalize_range]",
        "contrib",
        lambda d: K.contrib.diamond_square((2, 3, 16, 24), roughness=0.7, normalize_range=(0.0, 1.0)),
        [torch.zeros(1)],
        check=False,
        note="normalize_range applied",
        tags=("batch>1",),
    )
)
A(
    case(
        "contrib.histogram_matching",
        "contrib",
        K.contrib.histogram_matching,
        [torch.rand(1, 1, 16, 24), torch.rand(1, 1, 20, 30)],
        note="unique(return_inverse, return_counts) + cumsum + interp",
    )
)
A(
    case(
        "contrib.interp",
        "contrib",
        K.contrib.interp,
        [torch.rand(20) * 10, torch.linspace(0, 10, 6), torch.rand(6)],
        note="1-D linear interp via searchsorted/bucketize",
    )
)
_km_centers = torch.tensor([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
_km = K.contrib.KMeans(3, _km_centers, tolerance=None, max_iterations=5, seed=0)
A(
    case(
        "contrib.KMeans.predict",
        "contrib",
        _km.predict,
        [torch.randn(20, 2) * 0.5 + torch.tensor([2.5, 2.5])],
        note="cluster centers fixed (3,2); argmin of pairwise distances",
    )
)


def _kmeans_fit(x):
    km = K.contrib.KMeans(3, _km_centers.clone(), tolerance=None, max_iterations=5, seed=0)
    km.fit(x)
    return km.cluster_centers


A(
    case(
        "contrib.KMeans.fit",
        "contrib",
        _kmeans_fit,
        [torch.randn(20, 2) * 0.5 + torch.tensor([2.5, 2.5])],
        note="max_iterations=5, tolerance=None (fixed iteration count) but fit() uses torch.nonzero + data-dependent "
        "`if selected.shape[0] == 0` per cluster",
    )
)
A(case("contrib.Lambda", "contrib", K.contrib.Lambda(K.color.rgb_to_grayscale), [IMG], note="wraps rgb_to_grayscale"))

# ============================================================================= kornia.contrib — model wrappers
from kornia.contrib.face_detection import FaceDetector, FaceDetectorResult  # noqa: E402

fd, _e_fd = build(lambda: ev(FaceDetector()))
A(
    mcase(
        "contrib.FaceDetector",
        "contrib.wrappers",
        reuse(fd, _e_fd),
        lambda m: m,
        [torch.rand(1, 3, 64, 96) * 255],
        note="YuNet + PriorBox + decode + confidence mask + NMS; returns list of (N,15); data-dependent shapes",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "contrib.FaceDetector.model",
        "contrib.wrappers",
        reuse(fd, _e_fd),
        lambda m: m.model,
        [torch.rand(1, 3, 64, 96) * 255],
        note="the underlying YuNet only (as FaceDetector.model)",
        tags=("model", "pretrained"),
    )
)
A(
    case(
        "contrib.FaceDetectorResult",
        "contrib.wrappers",
        lambda d: torch.stack(
            [
                FaceDetectorResult(d).xmin,
                FaceDetectorResult(d).ymin,
                FaceDetectorResult(d).xmax,
                FaceDetectorResult(d).ymax,
                FaceDetectorResult(d).score,
                FaceDetectorResult(d).top_left[..., 0],
                FaceDetectorResult(d).bottom_right[..., 1],
            ],
            -1,
        ),
        [torch.rand(15) * 60],
        note="property accessors on a (15,) detection vector; __init__ requires len(data) >= 15 "
        "(message says 14) so a (N,15) "
        "batch with N<15 is rejected",
    )
)
A(case("contrib.FaceKeypoint", "contrib.wrappers", None, [], skip="enum"))

edge, _e_ed = build(lambda: ev(K.contrib.EdgeDetectorBuilder.build("dexined", pretrained=True, image_size=64)))
A(
    mcase(
        "contrib.EdgeDetectorBuilder.build[dexined,image_size=64]",
        "contrib.wrappers",
        reuse(edge, _e_ed),
        lambda m: m,
        [torch.rand(1, 3, 48, 80) * 255],
        note="Normalize -> DexiNed -> Sigmoid with ResizePreProcessor(64,64) and ResizePostProcessor (numpy tolist); "
        "image_size=64 baked",
        tags=("model", "pretrained"),
    )
)
A(
    mcase(
        "contrib.EdgeDetector.model",
        "contrib.wrappers",
        reuse(edge, _e_ed),
        lambda m: m.model,
        [torch.rand(1, 3, 64, 64)],
        note="core nn.Sequential(Normalize, DexiNed, Sigmoid) without pre/post processors",
        tags=("model", "pretrained"),
    )
)

od, _e_od = build(
    lambda: ev(
        K.contrib.RTDETRDetectorBuilder.build("rtdetr_r18vd", pretrained=True, image_size=256, confidence_threshold=0.0)
    )
)
A(
    mcase(
        "contrib.RTDETRDetectorBuilder.build[r18vd,image_size=256,threshold=0]",
        "contrib.wrappers",
        reuse(od, _e_od),
        lambda m: m,
        [torch.rand(1, 3, 200, 320)],
        note="ObjectDetector = ResizePreProcessor(256,256) + RTDETR + DETRPostProcessor; threshold 0 "
        "-> all 300 boxes (1,300,6). "
        "KORNIA BUG: builder passes `confidence_filtering or not is_in_onnx_export()` so confidence_filtering=False "
        "is ignored at build time (list output); forward is @torch.inference_mode()",
        tags=("model", "pretrained"),
    )
)
od_f, _e_odf = build(
    lambda: ev(
        K.contrib.RTDETRDetectorBuilder.build("rtdetr_r18vd", pretrained=True, image_size=256, confidence_threshold=0.3)
    )
)
A(
    mcase(
        "contrib.RTDETRDetectorBuilder.build[r18vd,threshold=0.3,list output]",
        "contrib.wrappers",
        reuse(od_f, _e_odf),
        lambda m: m,
        [torch.rand(1, 3, 200, 320)],
        note="default confidence_filtering=True unless is_in_onnx_export(); list of (Di,6) tensors",
        tags=("model", "pretrained"),
    )
)
A(case("contrib.ObjectDetector", "contrib.wrappers", None, [], skip="constructed by RTDETRDetectorBuilder (covered)"))
A(
    case(
        "contrib.ObjectDetectorResult/BoundingBox/BoundingBoxDataFormat",
        "contrib.wrappers",
        None,
        [],
        skip="dataclasses / enum",
    )
)
A(
    case(
        "contrib.object_detection.results_from_detections",
        "contrib.wrappers",
        None,
        [],
        skip="returns a Python list of ObjectDetectorResult dataclasses",
    )
)
A(
    case(
        "contrib.object_detection.ResizePreProcessor",
        "contrib.wrappers",
        None,
        [],
        skip="re-export of kornia.models.processors.ResizePreProcessor (covered)",
    )
)
_det = torch.cat([torch.randint(0, 80, (1, 300, 1)).float(), torch.rand(1, 300, 1), torch.rand(1, 300, 4) * 100], -1)
A(
    case(
        "contrib.object_detection.BoxFiltering[filter_as_zero]",
        "contrib.wrappers",
        K.contrib.object_detection.BoxFiltering(filter_as_zero=True),
        [_det, torch.tensor(0.5)],
        note="threshold passed as 0-d tensor input; `confidence_threshold or "
        "self.confidence_threshold` -> bool() on tensor",
    )
)
A(
    case(
        "contrib.object_detection.BoxFiltering[filter_as_zero,classes_to_keep]",
        "contrib.wrappers",
        K.contrib.object_detection.BoxFiltering(torch.tensor(0.5), torch.tensor([1, 2, 3]), filter_as_zero=True),
        [_det],
        note="threshold + classes baked as buffers",
    )
)
A(
    case(
        "contrib.object_detection.BoxFiltering[list output]",
        "contrib.wrappers",
        K.contrib.object_detection.BoxFiltering(torch.tensor(0.5)),
        [_det],
        note="returns list of variable-size tensors",
    )
)
A(
    case(
        "contrib.ImageStitcher",
        "contrib.wrappers",
        None,
        [],
        skip="LoFTR matcher + RANSAC (Python loop over data-dependent inlier counts) + `.item()` in postprocess; "
        "variable-arity forward(*imgs) — blocker recorded, not exportable",
    )
)
A(
    case(
        "contrib.VisualPrompter",
        "contrib.wrappers",
        None,
        [],
        skip="stateful plain-Python API (set_image/predict) over SAM with Keypoints/Boxes prompts; "
        "sub-modules covered by "
        "models.sam cases; default config is vit_h (>500 MB)",
    )
)
A(case("contrib.BoxMotTracker", "contrib.wrappers", None, [], skip="`boxmot` not installed; stateful tracker"))

# ============================================================================= kornia.tracking
A(
    case(
        "tracking.HomographyTracker",
        "tracking",
        None,
        [],
        skip="stateful tracker (GFTTAffNetHardNet + LoFTR + RANSAC) returning (H, bool); RANSAC and LoFTR coarse "
        "matching are data-dependent — blocker recorded",
    )
)

# ============================================================================= kornia.sensors.camera
from kornia.geometry.vector import Vector2, Vector3  # noqa: E402
from kornia.image import ImageSize  # noqa: E402
from kornia.sensors.camera import (  # noqa: E402
    CameraModel,
    CameraModelType,
    PinholeModel,
)

_pin = PinholeModel(ImageSize(480, 640), torch.tensor([[328.0, 328.0, 320.0, 240.0]]))
_pts3 = torch.tensor([[1.0, 2.0, 5.0], [-0.5, 0.3, 2.0]])
A(
    case(
        "sensors.camera.PinholeModel.project",
        "sensors.camera",
        lambda p: _pin.project(Vector3(p)).data,
        [_pts3],
        note="params (fx,fy,cx,cy) baked; Vector3 in -> Vector2 out (.data)",
    )
)
A(
    case(
        "sensors.camera.PinholeModel.unproject",
        "sensors.camera",
        lambda p, d: _pin.unproject(Vector2(p), d).data,
        [torch.tensor([[320.0, 240.0], [100.0, 50.0]]), torch.tensor([2.0, 3.0])],
        note="depth as live tensor",
    )
)
A(
    case(
        "sensors.camera.PinholeModel.matrix",
        "sensors.camera",
        lambda d: _pin.matrix(),
        [torch.zeros(1)],
        note="no tensor input: constant 3x3 K",
    )
)
A(
    case(
        "sensors.camera.PinholeModel.scale",
        "sensors.camera",
        lambda s: _pin.scale(s).params,
        [torch.tensor(0.5)],
        note="scale factor live tensor; returns new model params",
    )
)
A(
    case(
        "sensors.camera.CameraModel[pinhole].project",
        "sensors.camera",
        lambda p: (
            CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.tensor([[328.0, 328.0, 320.0, 240.0]]))
            .project(Vector3(p))
            .data
        ),
        [_pts3],
        note="generic wrapper delegating via __getattr__",
    )
)
A(
    case(
        "sensors.camera.BrownConradyModel.project",
        "sensors.camera",
        None,
        [],
        skip="raises NotImplementedError in kornia (BrownConradyTransform.distort is a stub)",
    )
)
A(
    case(
        "sensors.camera.KannalaBrandtK3.project",
        "sensors.camera",
        None,
        [],
        skip="raises NotImplementedError in kornia (KannalaBrandtK3Transform.distort is a stub)",
    )
)
A(
    case(
        "sensors.camera.Orthographic.project",
        "sensors.camera",
        None,
        [],
        skip="raises NotImplementedError in kornia (Orthographic.project is a stub)",
    )
)
A(case("sensors.camera.CameraModelBase/CameraModelType", "sensors.camera", None, [], skip="base class / enum"))

# ============================================================================= kornia.image
from kornia.image import ChannelsOrder, Image, ImageLayout, PixelFormat  # noqa: E402
from kornia.image.base import ColorSpace  # noqa: E402


def _mk_image(t: torch.Tensor, cs: ColorSpace = ColorSpace.RGB) -> Image:
    layout = ImageLayout(ImageSize(t.shape[-2], t.shape[-1]), t.shape[-3], ChannelsOrder.CHANNELS_FIRST)
    return Image(t, PixelFormat(cs, 32), layout)


A(
    case(
        "image.Image.to_gray",
        "image",
        lambda t: _mk_image(t).to_gray().data,
        [torch.rand(3, 16, 24)],
        note="Image container (C,H,W) float32; RGB->gray",
    )
)
A(case("image.Image.to_bgr", "image", lambda t: _mk_image(t).to_bgr().data, [torch.rand(3, 16, 24)]))
A(
    case(
        "image.Image.to_rgb[from bgr]",
        "image",
        lambda t: _mk_image(t, ColorSpace.BGR).to_rgb().data,
        [torch.rand(3, 16, 24)],
    )
)
A(case("image.Image.float", "image", lambda t: _mk_image(t).float().data, [torch.rand(3, 16, 24)], note="dtype no-op"))
A(case("image.Image.from_numpy/to_numpy/write/from_file", "image", None, [], skip="numpy / file IO"))
A(case("image.ImageSize/ImageLayout/PixelFormat/ChannelsOrder", "image", None, [], skip="metadata dataclasses / enums"))
A(
    case(
        "image.image_to_tensor/tensor_to_image/ImageToTensor/image_list_to_tensor",
        "image",
        None,
        [],
        skip="numpy <-> tensor conversion utilities (numpy input)",
    )
)
A(case("image.image_to_string/print_image", "image", None, [], skip="return/print strings"))
A(case("image.perform_keep_shape_image/video", "image", None, [], skip="decorators"))
A(
    case(
        "image.draw_line",
        "image",
        K.image.draw_line,
        [torch.zeros(3, 16, 24), torch.tensor([2, 3]), torch.tensor([20, 12]), torch.tensor([1.0, 0.5, 0.2])],
        note="Bresenham-like Python loop over max(|dx|,|dy|) computed from tensor values",
    )
)
A(
    case(
        "image.draw_point2d",
        "image",
        K.image.draw_point2d,
        [torch.zeros(3, 16, 24), torch.tensor([[2, 3], [10, 12], [20, 5]]), torch.tensor([1.0, 0.5, 0.2])],
        note="`zip(*points)` iterates tensor rows in Python",
    )
)
A(
    case(
        "image.draw_rectangle",
        "image",
        K.image.draw_rectangle,
        [
            torch.zeros(1, 3, 16, 24),
            torch.tensor([[[2.0, 3.0, 12.0, 10.0], [5.0, 1.0, 20.0, 14.0]]]),
            torch.tensor([1.0, 0.5, 0.2]),
        ],
        note="Python loop over batch and boxes with tensor-derived slicing",
    )
)
A(
    case(
        "image.draw_rectangle[fill]",
        "image",
        K.image.draw_rectangle,
        [torch.zeros(1, 3, 16, 24), torch.tensor([[[2.0, 3.0, 12.0, 10.0]]]), torch.tensor([1.0, 0.5, 0.2])],
        {"fill": True},
    )
)
A(
    case(
        "image.draw_convex_polygon",
        "image",
        K.image.draw_convex_polygon,
        [
            torch.zeros(1, 3, 16, 24),
            torch.tensor([[[2.0, 2.0], [20.0, 4.0], [18.0, 14.0], [4.0, 12.0]]]),
            torch.tensor([[1.0, 0.5, 0.2]]),
        ],
        note="polygon rasterisation via edge functions (vectorised)",
    )
)
A(
    case(
        "image.make_grid",
        "image",
        K.image.make_grid,
        [torch.rand(6, 3, 8, 12)],
        {"n_row": 3, "padding": 2},
        note="n_row, padding baked",
    )
)

# ============================================================================= heavy SAM encoders last
A(
    case(
        "models.sam.Sam.image_encoder[mobile_sam,pretrained]",
        "models.sam",
        raiser(_e_sm) if _e_sm else sam_mobile.image_encoder,
        [torch.rand(1, 3, 1024, 1024)],
        note="weights mobile_sam.pt (HF kornia/mobile_sam); TinyViT-5m img_size=1024 mobile_sam head -> (1,256,64,64); "
        "1024x1024 input mandatory",
        tags=("model", "pretrained"),
    )
)
A(
    case(
        "models.sam.Sam.image_encoder[vit_b,random]",
        "models.sam",
        raiser(_e_sb) if _e_sb else sam_b.image_encoder,
        [torch.rand(1, 3, 1024, 1024)],
        note="ImageEncoderViT ViT-B, window attention 14, random init; 1024x1024 mandatory (abs pos_embed 64x64)",
        tags=("model",),
    )
)

if __name__ == "__main__":
    run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)
