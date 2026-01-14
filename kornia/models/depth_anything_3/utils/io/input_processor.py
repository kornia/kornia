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

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input processor for Depth Anything 3 (parallelized).

This version removes the square center-crop step for "*crop" methods (same as your note).
In addition, it parallelizes per-image preprocessing using the provided `parallel_execution`.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from kornia.models.depth_anything_3.utils.logger import logger
from kornia.models.depth_anything_3.utils.parallel_utils import parallel_execution


class InputProcessor:
    """Prepares a batch of images for model inference.
    This processor converts a list of image file paths into a single, model-ready
    tensor. The processing pipeline is executed in parallel across multiple workers
    for efficiency.

    Pipeline:
      1) Load image and convert to RGB
      2) Boundary resize (upper/lower bound, preserving aspect ratio)
      3) Enforce divisibility by PATCH_SIZE:
         - "*resize" methods: each dimension is rounded to nearest multiple
           (may up/downscale a few px)
         - "*crop"   methods: each dimension is floored to nearest multiple via center crop
      4) Convert to tensor and apply ImageNet normalization
      5) Stack into (1, N, 3, H, W)

    Parallelization:
      - Each image is processed independently in a worker.
      - Order of outputs matches the input order.
    """

    NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    PATCH_SIZE = 14

    def __init__(self):
        pass

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        *,
        num_workers: int = 8,
        print_progress: bool = False,
        sequential: bool | None = None,
        desc: str | None = "Preprocess",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Returns:
        (tensor, extrinsics_list, intrinsics_list)
        tensor shape: (1, N, 3, H, W)
        """
        sequential = self._resolve_sequential(sequential, num_workers)
        exts_list, ixts_list = self._validate_and_pack_meta(image, extrinsics, intrinsics)

        results = self._run_parallel(
            image=image,
            exts_list=exts_list,
            ixts_list=ixts_list,
            process_res=process_res,
            process_res_method=process_res_method,
            num_workers=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
        )

        proc_imgs, out_sizes, out_ixts, out_exts = self._unpack_results(results)
        proc_imgs, out_sizes, out_ixts = self._unify_batch_shapes(proc_imgs, out_sizes, out_ixts)

        batch_tensor = self._stack_batch(proc_imgs)
        out_exts = (
            torch.from_numpy(np.asarray(out_exts)).float() if out_exts is not None and out_exts[0] is not None else None
        )
        out_ixts = (
            torch.from_numpy(np.asarray(out_ixts)).float() if out_ixts is not None and out_ixts[0] is not None else None
        )
        return (batch_tensor, out_exts, out_ixts)

    # -----------------------------
    # __call__ helpers
    # -----------------------------
    def _resolve_sequential(self, sequential: bool | None, num_workers: int) -> bool:
        return (num_workers <= 1) if sequential is None else sequential

    def _validate_and_pack_meta(
        self,
        images: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> tuple[list[np.ndarray | None] | None, list[np.ndarray | None] | None]:
        if extrinsics is not None and len(extrinsics) != len(images):
            raise ValueError("Length of extrinsics must match images when provided.")
        if intrinsics is not None and len(intrinsics) != len(images):
            raise ValueError("Length of intrinsics must match images when provided.")
        exts_list = [e for e in extrinsics] if extrinsics is not None else None
        ixts_list = [k for k in intrinsics] if intrinsics is not None else None
        return exts_list, ixts_list

    def _run_parallel(
        self,
        *,
        image: list[np.ndarray | Image.Image | str],
        exts_list: list[np.ndarray | None] | None,
        ixts_list: list[np.ndarray | None] | None,
        process_res: int,
        process_res_method: str,
        num_workers: int,
        print_progress: bool,
        sequential: bool,
        desc: str | None,
    ):
        results = parallel_execution(
            image,
            exts_list,
            ixts_list,
            action=self._process_one,  # (img, extrinsic, intrinsic, ...)
            num_processes=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        if not results:
            raise RuntimeError("No preprocessing results returned. Check inputs and parallel_execution.")
        return results

    def _unpack_results(self, results):
        """results: List[Tuple[torch.Tensor, Tuple[H, W], Optional[np.ndarray], Optional[np.ndarray]]]
        -> processed_images, out_sizes, out_intrinsics, out_extrinsics
        """
        try:
            processed_images, out_sizes, out_intrinsics, out_extrinsics = zip(*results)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected results structure from parallel_execution: {type(results)} / sample: {results[0]}"
            ) from e

        return list(processed_images), list(out_sizes), list(out_intrinsics), list(out_extrinsics)

    def _unify_batch_shapes(
        self,
        processed_images: list[torch.Tensor],
        out_sizes: list[tuple[int, int]],
        out_intrinsics: list[np.ndarray | None],
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]], list[np.ndarray | None]]:
        """Center-crop all tensors to the smallest H, W; adjust intrinsics' cx, cy accordingly."""
        if len(set(out_sizes)) <= 1:
            return processed_images, out_sizes, out_intrinsics

        min_h = min(h for h, _ in out_sizes)
        min_w = min(w for _, w in out_sizes)
        logger.warn(
            f"Images in batch have different sizes {out_sizes}; center-cropping all to smallest ({min_h},{min_w})"
        )

        center_crop = T.CenterCrop((min_h, min_w))
        new_imgs, new_sizes, new_ixts = [], [], []
        for img_t, (H, W), K in zip(processed_images, out_sizes, out_intrinsics):
            crop_top = max(0, (H - min_h) // 2)
            crop_left = max(0, (W - min_w) // 2)
            new_imgs.append(center_crop(img_t))
            new_sizes.append((min_h, min_w))
            if K is None:
                new_ixts.append(None)
            else:
                K_adj = K.copy()
                K_adj[0, 2] -= crop_left
                K_adj[1, 2] -= crop_top
                new_ixts.append(K_adj)
        return new_imgs, new_sizes, new_ixts

    def _stack_batch(self, processed_images: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(processed_images)

    # -----------------------------
    # Per-item worker
    # -----------------------------
    def _process_one(
        self,
        img: np.ndarray | Image.Image | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
    ) -> tuple[torch.Tensor, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        # Load & remember original size
        pil_img = self._load_image(img)
        orig_w, orig_h = pil_img.size

        # Boundary resize
        pil_img = self._resize_image(pil_img, process_res, process_res_method)
        w, h = pil_img.size
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w, h)

        # Enforce divisibility by PATCH_SIZE
        if process_res_method.endswith("resize"):
            pil_img = self._make_divisible_by_resize(pil_img, self.PATCH_SIZE)
            new_w, new_h = pil_img.size
            intrinsic = self._resize_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        elif process_res_method.endswith("crop"):
            pil_img = self._make_divisible_by_crop(pil_img, self.PATCH_SIZE)
            new_w, new_h = pil_img.size
            intrinsic = self._crop_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        # Convert to tensor & normalize
        img_tensor = self._normalize_image(pil_img)
        _, H, W = img_tensor.shape
        assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

        # Return: (img_tensor, (H, W), intrinsic, extrinsic)
        return img_tensor, (H, W), intrinsic, extrinsic

    # -----------------------------
    # Intrinsics transforms
    # -----------------------------
    def _resize_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        # scale fx, cx by w ratio; fy, cy by h ratio
        K[:1] *= w / float(orig_w)
        K[1:2] *= h / float(orig_h)
        return K

    def _crop_ixt(
        self,
        intrinsic: np.ndarray | None,
        orig_w: int,
        orig_h: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        if intrinsic is None:
            return None
        K = intrinsic.copy()
        crop_h = (orig_h - h) // 2
        crop_w = (orig_w - w) // 2
        K[0, 2] -= crop_w
        K[1, 2] -= crop_h
        return K

    # -----------------------------
    # I/O & normalization
    # -----------------------------
    def _load_image(self, img: np.ndarray | Image.Image | str) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            # Assume HxWxC uint8/RGB
            return Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    def _normalize_image(self, img: Image.Image) -> torch.Tensor:
        img_tensor = T.ToTensor()(img)
        return self.NORMALIZE(img_tensor)

    # -----------------------------
    # Boundary resizing
    # -----------------------------
    def _resize_image(self, img: Image.Image, target_size: int, method: str) -> Image.Image:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side(img, target_size)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side(img, target_size)
        else:
            raise ValueError(f"Unsupported resize method: {method}")

    def _resize_longest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest == target_size:
            return img
        scale = target_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    def _resize_shortest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    # -----------------------------
    # Make divisible by PATCH_SIZE
    # -----------------------------
    def _make_divisible_by_crop(self, img: Image.Image, patch: int) -> Image.Image:
        """Floor each dimension to the nearest multiple of PATCH_SIZE via center crop.
        Example: 504x377 -> 504x364
        """
        w, h = img.size
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def _make_divisible_by_resize(self, img: Image.Image, patch: int) -> Image.Image:
        """Round each dimension to nearest multiple of PATCH_SIZE via small resize."""
        w, h = img.size

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch))
        new_h = max(1, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        upscale = (new_w > w) or (new_h > h)
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)


# Backward compatibility alias
InputAdapter = InputProcessor


# ===========================
# Minimal test runner (parallel execution)
# ===========================
if __name__ == "__main__":
    """
    Minimal test suite:
      - Creates pairs of images so batch shapes match.
      - Tests all four process_res_methods.
      - Prints fx fy cx cy IN->OUT per image.
      - Includes cases with K/E provided and with None.
    """

    def fmt_k_line(K: np.ndarray | None) -> str:
        if K is None:
            return "None"
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        return f"fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}"

    def show_result(
        tag: str,
        tensor: torch.Tensor,
        Ks_in: Sequence[np.ndarray | None] | None = None,
        Ks_out: Sequence[np.ndarray | None] | None = None,
    ):
        B, N, C, H, W = tensor.shape
        print(f"[{tag}] shape={tuple(tensor.shape)}  HxW=({H},{W})  div14=({H % 14 == 0},{W % 14 == 0})")
        assert H % 14 == 0 and W % 14 == 0, f"{tag}: output size not divisible by 14!"
        if Ks_in is not None or Ks_out is not None:
            Ks_in = Ks_in or [None] * N
            Ks_out = Ks_out or [None] * N
            for i in range(N):
                print(f"  K[{i}]: {fmt_k_line(Ks_in[i])}  ->  {fmt_k_line(Ks_out[i])}")

    proc = InputProcessor()
    process_res = 504
    methods = ["upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"]

    # Example sizes (two orientations)
    small_sizes = [(680, 1208), (1208, 680)]
    large_sizes = [(1208, 680), (680, 1208)]

    def make_K(w, h, fx=1200.0, fy=1100.0):
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K

    def run_suite(suite_name: str, sizes: list[tuple[int, int]]):
        print(f"\n===== {suite_name} =====")
        for w, h in sizes:
            img = Image.new("RGB", (w, h), color=(123, 222, 100))
            batch_imgs = [img, img]

            # intrinsics / extrinsics examples
            Ks_in = [make_K(w, h), make_K(w, h)]
            Es_in = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]

            for m in methods:
                tensor, Es_out, Ks_out = proc(
                    image=batch_imgs,
                    process_res=process_res,
                    process_res_method=m,
                    num_workers=8,
                    print_progress=False,
                    intrinsics=Ks_in,  # test with non-None
                    extrinsics=Es_in,
                )
                show_result(f"{suite_name} size=({w},{h}) | {m}", tensor, Ks_in, Ks_out)

            # Also test None path
            tensor2, Es_out2, Ks_out2 = proc(
                image=batch_imgs,
                process_res=process_res,
                process_res_method="upper_bound_resize",
                num_workers=8,
                intrinsics=None,
                extrinsics=None,
            )
            show_result(
                f"{suite_name} size=({w},{h}) | upper_bound_resize | no K/E",
                tensor2,
                None,
                Ks_out2,
            )

    run_suite("SMALL", small_sizes)
    run_suite("LARGE", large_sizes)

    # Extra sanity for 504x376
    print("\n===== EXTRA sanity for 504x376 =====")
    img_example = Image.new("RGB", (504, 376), color=(10, 20, 30))
    Ks_in_extra = [make_K(504, 376, fx=900.0, fy=900.0), make_K(504, 376, fx=900.0, fy=900.0)]

    out_r, _, Ks_out_r = proc(
        image=[img_example, img_example],
        process_res=504,
        process_res_method="upper_bound_resize",
        num_workers=8,
        intrinsics=Ks_in_extra,
    )
    out_c, _, Ks_out_c = proc(
        image=[img_example, img_example],
        process_res=504,
        process_res_method="upper_bound_crop",
        num_workers=8,
        intrinsics=Ks_in_extra,
    )
    _, _, _, Hr, Wr = out_r.shape
    _, _, _, Hc, Wc = out_c.shape
    print(f"upper_bound_resize -> ({Hr},{Wr})  (rounded to nearest multiple of 14)")
    show_result("Ks after upper_bound_resize", out_r, Ks_in_extra, Ks_out_r)
    print(f"upper_bound_crop   -> ({Hc},{Wc})  (floored to multiple of 14)")
    show_result("Ks after upper_bound_crop", out_c, Ks_in_extra, Ks_out_c)
