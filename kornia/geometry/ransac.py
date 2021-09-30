"""Module containing RANSAC modules"""
from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.homography import symmetric_transfer_error
from kornia.geometry import symmetrical_epipolar_distance
from kornia.geometry import (find_fundamental,
                             find_homography_dlt,
                             find_homography_dlt_iterated)

__all__ = ["RANSAC"]


class RANSAC(nn.Module):
    supported_models = ['homography', 'fundamental']
    def __init__(self,
                 model_type: str = 'homography',
                 batch_size: int = 2048,
                 max_iter: int = 10,
                 inl_th: float = 2.0,
                 confidence: float = 0.99):
        super().__init__()
        if model_type not in self.supported_models:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {supported_models}")
        self.inl_th = inl_th
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        if model_type == 'homography':
            self.error_fn = symmetric_transfer_error
            self.minimal_solver = find_homography_dlt
            self.polisher_solver = find_homography_dlt_iterated
            self.minimal_sample_size = 4
        elif model_type == 'fundamental':
            self.error_fn = symmetrical_epipolar_distance
            self.minimal_solver = find_fundamental
            self.minimal_sample_size = 8
            # ToDo: implement 7pt solver instead of 8pt minimal_solver
            # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L498
            self.polisher_solver = find_fundamental
        else:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {supported_models}")
        return

    def sample(self, sample_size: int,
               pop_size: int,
               batch_size: int, device=torch.device('cpu')) -> torch.Tensor:
        '''Minimal sampler, but unlike traditional RANSAC we sample in batches
         to get benefit of the parallel processing, esp. on GPU'''
        out: torch.Tensor = torch.empty(batch_size, sample_size)
        # for loop, until https://github.com/pytorch/pytorch/issues/42502 accepted
        for i in range(batch_size):
            out[i] = torch.randperm(pop_size,dtype=torch.int32, device=device)[:sample_size]
        return out

    @staticmethod
    def max_samples_by_conf(n_inl:int, num_tc:int, sample_size:int, conf: float):
        '''Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus'''
        if n_inl  == num_tc:
            return 1
        return math.log(1.- conf) / math.log(1. - math.pow(n_inl/num_tc, sample_size))

    def estimate_model_from_minsample(self, kp1, kp2):
        batch_size, sample_size = kp1.shape[:2]
        H = self.minimal_solver(kp1,
                                kp2,
                                torch.ones(batch_size,
                                           sample_size,
                                           dtype=kp1.dtype,
                                           device=kp1.device))
        return H

    def verify(self, kp1, kp2, models, inl_th):
        if len(kp1.shape) == 2:
            kp1 = kp1[None]
        if len(kp2.shape) == 2:
            kp2 = kp2[None]
        batch_size = models.shape[0]
        errors = self.error_fn(kp1.expand(batch_size, -1, 2),
                               kp2.expand(batch_size, -1, 2),
                               models)
        inl = (errors <= inl_th)
        models_score = inl.float().sum(dim=1)
        best_model_idx = models_score.argmax()
        best_model_score = models_score[best_model_idx]
        model_best = models[best_model_idx].clone()
        inliers_best = inl[best_model_idx]
        return model_best, inliers_best, best_model_score

    def remove_bad_samples(self, kp1, kp2):
        ''''''
        # ToDo: add (model-specific) verification of the samples,
        # E.g. contraints on not to be a degenerate sample
        return kp1, kp2

    def remove_bad_models(self, models):
        # ToDo: add more and better degenerate model rejection
        # For now it is simple and hardcoded
        main_diagonal = torch.diagonal(models,
                                       dim1=1,
                                       dim2=2)
        mask = main_diagonal.abs().min(dim=1)[0] > 1e-6
        return models[mask]

    def polish_model(self, kp1, kp2, inliers):
        # ToDo: Replace this with MAGSAC++ polisher
        kp1_inl = kp1[inliers][None]
        kp2_inl = kp2[inliers][None]
        num_inl = kp1_inl.size(1)
        model = self.polisher_solver(kp1_inl,
                                     kp2_inl,
                                     torch.ones(1,
                                                num_inl,
                                                dtype=kp1_inl.dtype,
                                                device=kp1_inl.device))
        return model
    def forward(self, kp1, kp2):
        assert len(kp1.shape) == 2
        assert len(kp2.shape) == 2

        best_score_total = 1
        num_tc = len(kp1)
        best_model_total = None
        inliers_best_total = torch.zeros(num_tc, 1, dtype=bool)
        for i in range(self.max_iter):
            # Sample minimal samples in batch to estimate models
            idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size).long()
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]

            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            # Estimate models
            models = self.estimate_model_from_minsample(kp1_sampled, kp2_sampled)
            models = self.remove_bad_models(models)
            if models is None:
                continue
            # Score the models and select the best one
            model_best, inliers_best, best_model_score = self.verify(kp1, kp2, models, self.inl_th)
            # Store far-the-best model and (optionally) do a local optimization
            if best_model_score > best_score_total:
                best_model_total = model_best.clone()
                inliers_best_total = inliers_best.clone()
                best_score_total = best_model_score
                # Local optimization
                model_lo = self.polish_model(kp1, kp2, inliers_best)
                _, inliers_lo, score_lo = self.verify(kp1, kp2, model_lo, self.inl_th)
                # print (f"Orig score = {best_model_score}, LO score = {score_lo} TC={num_tc}")
                if score_lo > best_score_total:
                    best_model_total = model_lo.clone()[0]
                    inliers_best_total = inliers_lo.clone()
                    best_score_total = score_lo
                new_max_iter = int(self.max_samples_by_conf(best_score_total,
                                                            num_tc,
                                                            self.minimal_sample_size,
                                                            self.confidence))
                # print (f"New max_iter = {new_max_iter}")
                # Stop estimation, if the model is very good
                if (i+1) * self.batch_size >= new_max_iter:
                    break
        # local optimization with all inliers for better precision
        return best_model_total, inliers_best_total
