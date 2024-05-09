"""Module containing RANSAC modules."""

import math
from functools import partial
from typing import Callable, Optional, Tuple

import torch

from kornia.core import Device, Module, Tensor, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry import (
    find_fundamental,
    find_homography_dlt,
    find_homography_dlt_iterated,
    find_homography_lines_dlt,
    find_homography_lines_dlt_iterated,
    symmetrical_epipolar_distance,
)
from kornia.geometry.homography import (
    line_segment_transfer_error_one_way,
    oneway_transfer_error,
    sample_is_valid_for_homography,
)

__all__ = ["RANSAC"]


class RANSAC(Module):
    """Module for robust geometry estimation with RANSAC. https://en.wikipedia.org/wiki/Random_sample_consensus.

    Args:
        model_type: type of model to estimate: "homography", "fundamental", "fundamental_7pt",
            "homography_from_linesegments".
        inliers_threshold: threshold for the correspondence to be an inlier.
        batch_size: number of generated samples at once.
        max_iterations: maximum batches to generate. Actual number of models to try is ``batch_size * max_iterations``.
        confidence: desired confidence of the result, used for the early stopping.
        max_local_iterations: number of local optimization (polishing) iterations.
    """

    def __init__(
        self,
        model_type: str = "homography",
        inl_th: float = 2.0,
        batch_size: int = 2048,
        max_iter: int = 10,
        confidence: float = 0.99,
        max_lo_iters: int = 5,
    ) -> None:
        super().__init__()
        self.supported_models = ["homography", "fundamental", "fundamental_7pt", "homography_from_linesegments"]
        self.inl_th = inl_th
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        self.model_type = model_type

        self.error_fn: Callable[..., Tensor]
        self.minimal_solver: Callable[..., Tensor]
        self.polisher_solver: Callable[..., Tensor]

        if model_type == "homography":
            self.error_fn = oneway_transfer_error
            self.minimal_solver = find_homography_dlt
            self.polisher_solver = find_homography_dlt_iterated
            self.minimal_sample_size = 4
        elif model_type == "homography_from_linesegments":
            self.error_fn = line_segment_transfer_error_one_way
            self.minimal_solver = find_homography_lines_dlt
            self.polisher_solver = find_homography_lines_dlt_iterated
            self.minimal_sample_size = 4
        elif model_type == "fundamental":
            self.error_fn = symmetrical_epipolar_distance
            self.minimal_solver = find_fundamental
            self.minimal_sample_size = 8
            self.polisher_solver = find_fundamental
        elif model_type == "fundamental_7pt":
            self.error_fn = symmetrical_epipolar_distance
            self.minimal_solver = partial(find_fundamental, method="7POINT")
            self.minimal_sample_size = 7
            self.polisher_solver = find_fundamental
        else:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {self.supported_models}")

    def sample(self, sample_size: int, pop_size: int, batch_size: int, device: Device = torch.device("cpu")) -> Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches to get benefit of the parallel
        processing, esp.

        on GPU.
        """
        rand = torch.rand(batch_size, pop_size, device=device)
        _, out = rand.topk(k=sample_size, dim=1)
        return out

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        eps = 1e-9
        if num_tc <= sample_size:
            return 1.0
        if n_inl == num_tc:
            return 1.0
        return math.log(1.0 - conf) / min(-eps, math.log(max(eps, 1.0 - math.pow(n_inl / num_tc, sample_size))))

    def estimate_model_from_minsample(self, kp1: Tensor, kp2: Tensor) -> Tensor:
        batch_size, sample_size = kp1.shape[:2]
        H = self.minimal_solver(kp1, kp2, torch.ones(batch_size, sample_size, dtype=kp1.dtype, device=kp1.device))
        return H

    def verify(self, kp1: Tensor, kp2: Tensor, models: Tensor, inl_th: float) -> Tuple[Tensor, Tensor, float]:
        if len(kp1.shape) == 2:
            kp1 = kp1[None]
        if len(kp2.shape) == 2:
            kp2 = kp2[None]
        batch_size = models.shape[0]
        if self.model_type == "homography_from_linesegments":
            errors = self.error_fn(kp1.expand(batch_size, -1, 2, 2), kp2.expand(batch_size, -1, 2, 2), models)
        else:
            errors = self.error_fn(kp1.expand(batch_size, -1, 2), kp2.expand(batch_size, -1, 2), models)
        inl = errors <= inl_th
        models_score = inl.to(kp1).sum(dim=1)
        best_model_idx = models_score.argmax()
        best_model_score = models_score[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inl[best_model_idx]
        return model_best, inliers_best, best_model_score

    def remove_bad_samples(self, kp1: Tensor, kp2: Tensor) -> Tuple[Tensor, Tensor]:
        """"""
        # ToDo: add (model-specific) verification of the samples,
        # E.g. constraints on not to be a degenerate sample
        if self.model_type == "homography":
            mask = sample_is_valid_for_homography(kp1, kp2)
            return kp1[mask], kp2[mask]
        return kp1, kp2

    def remove_bad_models(self, models: Tensor) -> Tensor:
        # ToDo: add more and better degenerate model rejection
        # For now it is simple and hardcoded
        main_diagonal = torch.diagonal(models, dim1=1, dim2=2)
        mask = main_diagonal.abs().min(dim=1)[0] > 1e-4
        return models[mask]

    def polish_model(self, kp1: Tensor, kp2: Tensor, inliers: Tensor) -> Tensor:
        # TODO: Replace this with MAGSAC++ polisher
        kp1_inl = kp1[inliers][None]
        kp2_inl = kp2[inliers][None]
        num_inl = kp1_inl.size(1)
        model = self.polisher_solver(
            kp1_inl, kp2_inl, torch.ones(1, num_inl, dtype=kp1_inl.dtype, device=kp1_inl.device)
        )
        return model

    def validate_inputs(self, kp1: Tensor, kp2: Tensor, weights: Optional[Tensor] = None) -> None:
        if self.model_type in ["homography", "fundamental"]:
            KORNIA_CHECK_SHAPE(kp1, ["N", "2"])
            KORNIA_CHECK_SHAPE(kp2, ["N", "2"])
            if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
                raise ValueError(
                    "kp1 and kp2 should be                                  equal shape at least"
                    f" [{self.minimal_sample_size}, 2],                                  got {kp1.shape}, {kp2.shape}"
                )
        if self.model_type == "homography_from_linesegments":
            KORNIA_CHECK_SHAPE(kp1, ["N", "2", "2"])
            KORNIA_CHECK_SHAPE(kp2, ["N", "2", "2"])
            if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
                raise ValueError(
                    "kp1 and kp2 should be                                  equal shape at least"
                    f" [{self.minimal_sample_size}, 2, 2],                                  got {kp1.shape},"
                    f" {kp2.shape}"
                )

    def forward(self, kp1: Tensor, kp2: Tensor, weights: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Main forward method to execute the RANSAC algorithm.

        Args:
            kp1: source image keypoints :math:`(N, 2)`.
            kp2: distance image keypoints :math:`(N, 2)`.
            weights: optional correspondences weights. Not used now.

        Returns:
            - Estimated model, shape of :math:`(1, 3, 3)`.
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
        """
        self.validate_inputs(kp1, kp2, weights)
        best_score_total: float = float(self.minimal_sample_size)
        num_tc: int = len(kp1)
        best_model_total = zeros(3, 3, dtype=kp1.dtype, device=kp1.device)
        inliers_best_total: Tensor = zeros(num_tc, 1, device=kp1.device, dtype=torch.bool)
        for i in range(self.max_iter):
            # Sample minimal samples in batch to estimate models
            idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size, kp1.device)
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]

            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            if len(kp1_sampled) == 0:
                continue
            # Estimate models
            models = self.estimate_model_from_minsample(kp1_sampled, kp2_sampled)
            models = self.remove_bad_models(models)
            if (models is None) or (len(models) == 0):
                continue
            # Score the models and select the best one
            model, inliers, model_score = self.verify(kp1, kp2, models, self.inl_th)
            # Store far-the-best model and (optionally) do a local optimization
            if model_score > best_score_total:
                # Local optimization
                for lo_step in range(self.max_lo_iters):
                    model_lo = self.polish_model(kp1, kp2, inliers)
                    if (model_lo is None) or (len(model_lo) == 0):
                        continue
                    _, inliers_lo, score_lo = self.verify(kp1, kp2, model_lo, self.inl_th)
                    # print (f"Orig score = {best_model_score}, LO score = {score_lo} TC={num_tc}")
                    if score_lo > model_score:
                        model = model_lo.clone()[0]
                        inliers = inliers_lo.clone()
                        model_score = score_lo
                    else:
                        break
                # Now storing the best model
                best_model_total = model.clone()
                inliers_best_total = inliers.clone()
                best_score_total = model_score

                # Should we already stop?
                new_max_iter = int(
                    self.max_samples_by_conf(int(best_score_total), num_tc, self.minimal_sample_size, self.confidence)
                )
                # print (f"New max_iter = {new_max_iter}")
                # Stop estimation, if the model is very good
                if (i + 1) * self.batch_size >= new_max_iter:
                    break
        # local optimization with all inliers for better precision
        return best_model_total, inliers_best_total
