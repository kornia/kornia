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

import json
from copy import deepcopy

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation.container.params import ParamItem
from kornia.constants import DataKey, Resample
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from testing.base import BaseTester


class TestAugmentationAudit(BaseTester):
    def test_flip_and_actual_keypoints(self, device, dtype):
        image = torch.rand(2, 1, 8, 10, device=device, dtype=dtype)
        points = torch.tensor([[[1.0, 2.0], [11.0, 4.0]]], device=device, dtype=dtype).expand(2, -1, -1)
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1), data_keys=["input", "keypoints"])
        outputs, report = aug.audit(image, points)
        self.assert_close(
            outputs[1], torch.tensor([[[8.0, 2.0], [-2.0, 4.0]]], device=device, dtype=dtype).expand(2, -1, -1)
        )
        assert report.geometry_status == "available"
        assert report.steps[0].name == "RandomHorizontalFlip_0"
        assert report.spatial[0].metric == "euclidean"
        self.assert_close(report.spatial[0].out_of_frame, torch.ones(2, device=device, dtype=torch.long))
        self.assert_close(report.spatial[0].roundtrip_max, torch.zeros_like(report.spatial[0].roundtrip_max))
        assert json.loads(report.to_json())["geometry_status"] == "available"

    def test_rng_and_replay(self, device, dtype):
        image = torch.rand(2, 1, 10, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomAffine(30), K.RandomBrightness(0.2))
        torch.manual_seed(123)
        ordinary = aug(image)
        state = torch.random.get_rng_state()
        torch.manual_seed(123)
        audited, report = aug.audit(image)
        self.assert_close(audited, ordinary)
        assert torch.equal(torch.random.get_rng_state(), state)
        self.assert_close(aug(image, params=report.params), audited)

    def test_nested_order(self, device, dtype):
        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        points = image.new_tensor([[[2, 3]]])
        aug = K.AugmentationSequential(
            K.ImageSequential(K.RandomHorizontalFlip(p=1), K.RandomVerticalFlip(p=1)),
            K.RandomBrightness(0.2),
            data_keys=["input", "keypoints"],
        )
        _, report = aug.audit(image, points)
        assert [step.name for step in report.steps] == [
            "ImageSequential_0.RandomHorizontalFlip_0",
            "ImageSequential_0.RandomVerticalFlip_1",
            "RandomBrightness_1",
        ]
        self.assert_close(report.matrix, image.new_tensor([[[-1, 0, 9], [0, -1, 7], [0, 0, 1]]]))
        assert report.spatial[0].roundtrip_max.item() < 1e-4

    def test_repeated_operations_capture_each_matrix(self, device, dtype):
        image = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        points = image.new_tensor([[[2, 3]]])
        module = K.RandomAffine(0, translate=(0.3, 0.3), p=1)
        aug = K.AugmentationSequential(module, data_keys=["input", "keypoints"])
        first = module.forward_parameters(image.shape)
        second = deepcopy(first)
        first["translations"] = image.new_tensor([[1, 0]])
        second["translations"] = image.new_tensor([[0, 2]])
        params = [ParamItem("RandomAffine_0", first), ParamItem("RandomAffine_0", second)]
        outputs, report = aug.audit(image, points, params=params)
        assert [step.occurrence for step in report.steps] == [0, 1]
        self.assert_close(report.steps[0].matrix[..., :2, 2], image.new_tensor([[1, 0]]))
        self.assert_close(report.steps[1].matrix[..., :2, 2], image.new_tensor([[0, 2]]))
        self.assert_close(outputs[1], image.new_tensor([[[3, 5]]]))
        assert report.spatial[0].roundtrip_max.item() < 1e-4

    def test_mixed_application(self, device, dtype):
        image = torch.rand(2, 1, 8, 10, device=device, dtype=dtype)
        points = image.new_tensor([[[1, 2]], [[1, 2]]])
        flip = K.RandomHorizontalFlip(p=0.5)
        params = flip.forward_parameters(image.shape)
        params["batch_prob"] = image.new_tensor([1, 0])
        aug = K.AugmentationSequential(flip, data_keys=["input", "keypoints"])
        outputs, report = aug.audit(image, points, params=[ParamItem("RandomHorizontalFlip_0", params)])
        self.assert_close(outputs[1], image.new_tensor([[[8, 2]], [[1, 2]]]))
        self.assert_close(report.matrix[1], torch.eye(3, device=device, dtype=report.matrix.dtype))
        assert report.spatial[0].roundtrip_max.max().item() < 1e-4

    @pytest.mark.parametrize("cropping_mode", ["slice", "resample"])
    def test_crop_padding_and_content_loss(self, device, dtype, cropping_mode):
        image = torch.rand(2, 1, 6, 8, device=device, dtype=dtype)
        points = image.new_tensor([[[0, 0], [3, 3], [7, 5]]]).expand(2, -1, -1)
        aug = K.AugmentationSequential(
            K.RandomCrop((4, 5), padding=2, cropping_mode=cropping_mode),
            data_keys=["input", "keypoints"],
        )
        _, report = aug.audit(image, points)
        assert report.geometry_status == "available"
        assert report.spatial[0].roundtrip_max.max().item() < 1e-4
        assert any("discard image content" in message for message in report.warnings)
        assert not report.image_reconstruction_evaluated

    @pytest.mark.parametrize("mode", ["bbox", "bbox_xyxy", "bbox_xywh"])
    def test_box_corner_order(self, device, dtype, mode):
        image = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        vertices = image.new_tensor([[[[1, 2], [5, 2], [5, 6], [1, 6]]]])
        box = Boxes(vertices)
        tensor_mode = {"bbox": "vertices_plus", "bbox_xyxy": "xyxy_plus", "bbox_xywh": "xywh"}[mode]
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1), data_keys=["input", mode])
        _, report = aug.audit(image, box.to_tensor(mode=tensor_mode))
        assert report.spatial[0].metric == "corner_hausdorff"
        assert report.spatial[0].roundtrip_max.item() < 1e-4

    def test_box_tensor_envelope_loss(self, device, dtype):
        image = torch.rand(1, 1, 20, 20, device=device, dtype=dtype)
        boxes = Boxes(image.new_tensor([[[[6, 7], [12, 7], [12, 13], [6, 13]]]]))
        aug = K.AugmentationSequential(K.RandomRotation((45, 45), p=1), data_keys=["input", "bbox"])
        _, tensor_report = aug.audit(image, boxes.to_tensor(mode="vertices_plus"))
        _, object_report = aug.audit(image, boxes)
        assert tensor_report.spatial[0].roundtrip_max.item() > 1
        assert object_report.spatial[0].roundtrip_max.item() < 1e-4
        assert any("round-trip error" in warning for warning in tensor_report.warnings)

    def test_detects_actual_spatial_error(self, device, dtype):
        class BrokenFlip(K.RandomHorizontalFlip):
            def apply_transform_keypoint(self, input, params, flags, transform=None):
                return input.clone()

        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(BrokenFlip(p=1), data_keys=["input", "keypoints"])
        _, report = aug.audit(image, image.new_tensor([[[1, 2]]]))
        self.assert_close(report.spatial[0].roundtrip_max, image.new_tensor([7]))
        assert "round-trip error" in report.summary()

    def test_nonrigid_silent_identity_is_not_supported(self, device, dtype):
        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomElasticTransform(p=1), data_keys=["input", "keypoints"])
        points = image.new_tensor([[[1, 2]]])
        outputs, report = aug.audit(image, points)
        self.assert_close(outputs[1], points)
        assert report.geometry_status == "unsupported"
        assert report.matrix is None and report.inverse_matrix is None
        assert report.spatial[0].roundtrip_valid_count.item() == 0
        assert torch.isnan(report.spatial[0].roundtrip_max).all()
        assert "non-rigid" in report.summary()

    def test_singular_and_nonfinite_matrices(self, device, dtype):
        class SingularFlip(K.RandomHorizontalFlip):
            def compute_transformation(self, input, params, flags):
                matrix = self.identity_matrix(input)
                matrix[1, 1, 1] = 0
                matrix[2, 0, 0] = float("nan")
                return matrix

        image = torch.rand(3, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(SingularFlip(p=1))
        _, report = aug.audit(image)
        assert report.geometry_status == "singular"
        assert report.invertible.tolist() == [True, False, False]
        assert report.inverse_matrix[1:].isnan().all()
        assert json.loads(report.to_json())["matrix"]["values"][2][0][0] is None

    @pytest.mark.parametrize("key, shape", [("keypoints", (2, 0, 2)), ("bbox", (2, 0, 4, 2))])
    def test_empty_spatial_inputs(self, device, dtype, key, shape):
        image = torch.rand(2, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1), data_keys=["input", key])
        _, report = aug.audit(image, torch.empty(shape, device=device, dtype=dtype))
        assert report.spatial[0].count.tolist() == [0, 0]
        assert report.spatial[0].roundtrip_valid_count.tolist() == [0, 0]
        assert report.spatial[0].roundtrip_max.isnan().all()
        assert json.loads(report.to_json())["spatial"][0]["roundtrip_max"]["values"] == [None, None]

    def test_nonfinite_and_boundaries(self, device, dtype):
        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        points = image.new_tensor([[[0, 0], [9, 7], [10, 7], [float("nan"), 0]]])
        aug = K.AugmentationSequential(torch.nn.Identity(), data_keys=["input", "keypoints"])
        _, report = aug.audit(image, points)
        item = report.spatial[0]
        assert item.count.item() == 4 and item.out_of_frame.item() == 1 and item.nonfinite.item() == 1
        assert item.out_of_frame_fraction.item() == 0.25 and item.roundtrip_valid_count.item() == 3
        json.loads(report.to_json())

    def test_snapshots_and_gradients(self, device, dtype):
        image = torch.rand(2, 1, 8, 10, device=device, dtype=dtype, requires_grad=True)
        points = image.new_tensor([[[1, 2]], [[1, 2]]]).requires_grad_()
        aug = K.AugmentationSequential(K.RandomAffine(20, p=1), data_keys=["input", "keypoints"])
        outputs, report = aug.audit(image, points)
        audited_grad = torch.autograd.grad(outputs[0].square().sum() + outputs[1].square().sum(), (image, points))
        serialized = report.to_json()
        ordinary = aug(image, points, params=report.params)
        normal_grad = torch.autograd.grad(ordinary[0].square().sum() + ordinary[1].square().sum(), (image, points))
        for actual, expected in zip(audited_grad, normal_grad):
            self.assert_close(actual, expected)
        aug(image, points)
        assert report.to_json() == serialized
        assert report.matrix.grad_fn is None
        assert report.steps[0].matrix.grad_fn is None
        assert report.params[0].data["translations"].grad_fn is None

    def test_mask_metadata_and_data_key_override(self, device, dtype):
        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1))
        outputs, report = aug.audit(image, image.clone(), data_keys=["input", "mask"])
        self.assert_close(outputs[0], outputs[1])
        assert report.inputs[1]["data_key"] == "MASK"
        assert report.configured_extra_args[DataKey.MASK]["resample"] == Resample.NEAREST
        assert aug.data_keys == [DataKey.INPUT]

    @pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
    def test_invalid_threshold_no_side_effect(self, threshold):
        aug = K.AugmentationSequential(K.RandomHorizontalFlip())
        state = torch.random.get_rng_state()
        with pytest.raises(ValueError, match="roundtrip_tolerance"):
            aug.audit(torch.ones(1, 1, 8, 8), roundtrip_tolerance=threshold)
        assert torch.equal(state, torch.random.get_rng_state())
        assert aug._params is None

    def test_unsupported_inputs_before_execution(self):
        module = K.RandomHorizontalFlip()
        aug = K.AugmentationSequential(module)
        state = torch.random.get_rng_state()
        for image in [torch.ones(1, 8, 8), torch.ones(0, 1, 8, 8), {"input": torch.ones(1, 1, 8, 8)}]:
            with pytest.raises(ValueError, match="BCHW"):
                aug.audit(image)
        assert torch.equal(state, torch.random.get_rng_state())
        assert not module._forward_hooks

    def test_hooks_removed_after_failure(self):
        class FailingModule(torch.nn.Module):
            def forward(self, input):
                raise RuntimeError("deliberate failure")

        module = FailingModule()
        aug = K.AugmentationSequential(module)
        with pytest.raises(RuntimeError, match="deliberate failure"):
            aug.audit(torch.ones(1, 1, 8, 8))
        assert not module._forward_hooks

    def test_reject_shared_aliases_and_video(self):
        module = K.RandomHorizontalFlip()
        shared = K.AugmentationSequential(module, module)
        video = K.AugmentationSequential(K.VideoSequential(K.RandomHorizontalFlip()))
        for aug in (shared, video):
            with pytest.raises(ValueError):
                aug.audit(torch.ones(1, 1, 8, 8))
        assert not module._forward_hooks

    def test_empty_pipeline(self, device, dtype):
        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        output, report = K.AugmentationSequential().audit(image)
        self.assert_close(output, image)
        assert report.steps == [] and report.params == []
        assert report.geometry_status == "available"
        self.assert_close(report.matrix, torch.eye(3, device=device, dtype=report.matrix.dtype).unsqueeze(0))

    def test_random_repeated_sampling(self, device, dtype):
        image = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(K.RandomAffine(20, p=1), random_apply=3)
        output, report = aug.audit(image)
        assert [step.occurrence for step in report.steps] == [0, 1, 2]
        assert len(report.params) == 3
        self.assert_close(aug(image, params=report.params), output)

    def test_input_box_metadata_and_warning_threshold(self, device, dtype):
        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        boxes = image.new_tensor([[[1, 2, 12, 6]]])
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(p=1), data_keys=["input", "bbox_xyxy"])
        _, report = aug.audit(image, boxes, out_of_frame_tolerance=1)
        assert report.inputs[1]["shape"] == (1, 1, 4)
        assert report.spatial[0].out_of_frame.item() == 1
        assert not any("out-of-frame" in warning for warning in report.warnings)

    def test_inverse_projective_horizon_is_invalid(self, device, dtype):
        class HorizonFlip(K.RandomHorizontalFlip):
            def compute_transformation(self, input, params, flags):
                matrix = self.identity_matrix(input)
                matrix[:, 2, 0] = 1
                return matrix

            def apply_transform_keypoint(self, input, params, flags, transform=None):
                return input.clone()

        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(HorizonFlip(p=1), data_keys=["input", "keypoints"])
        _, report = aug.audit(image, image.new_tensor([[[1, 2]]]))
        assert report.geometry_status == "available"
        assert report.spatial[0].nonfinite.item() == 0
        assert report.spatial[0].roundtrip_valid_count.item() == 0
        assert "some round-trip errors are nonfinite" in report.summary()

    def test_unknown_module_is_unavailable(self, device, dtype):
        aug = K.AugmentationSequential(torch.nn.ReLU())
        output, report = aug.audit(torch.ones(1, 1, 8, 10, device=device, dtype=dtype))
        assert output.shape == (1, 1, 8, 10)
        assert report.geometry_status == "unsupported"
        assert report.steps[0].unsupported_reason is not None

    def test_reject_ragged_container_objects_before_execution(self, device, dtype):
        image = torch.ones(2, 1, 8, 10, device=device, dtype=dtype)
        vertices = image.new_tensor([[[1, 2], [5, 2], [5, 6], [1, 6]], [[2, 3], [6, 3], [6, 7], [2, 7]]])
        spatial = Boxes([vertices, vertices[:1]])
        module = K.RandomHorizontalFlip()
        aug = K.AugmentationSequential(module, data_keys=["input", "bbox"])
        state = torch.random.get_rng_state()
        with pytest.raises(ValueError, match="ragged"):
            aug.audit(image, spatial)
        assert torch.equal(state, torch.random.get_rng_state())
        assert aug._params is None
        assert not module._forward_hooks

    def test_even_median_and_invalid_errors(self, device, dtype):
        class BrokenFlip(K.RandomHorizontalFlip):
            def apply_transform_keypoint(self, input, params, flags, transform=None):
                return input.clone()

        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        aug = K.AugmentationSequential(BrokenFlip(p=1), data_keys=["input", "keypoints"])
        _, report = aug.audit(image, image.new_tensor([[[3, 2], [4, 2], [float("nan"), 2]]]))
        item = report.spatial[0]
        assert item.roundtrip_valid_count.item() == 2
        assert item.roundtrip_mean.item() == 2
        assert item.roundtrip_median.item() == 2
        assert item.roundtrip_max.item() == 3

    def test_none_data_keys_requires_explicit_override(self):
        aug = K.AugmentationSequential(K.RandomHorizontalFlip(), data_keys=None)
        image = torch.ones(1, 1, 8, 8)
        with pytest.raises(ValueError, match="data_keys"):
            aug.audit(image)
        _, report = aug.audit(image, data_keys=["input"])
        assert report.geometry_status == "available"

    def test_changed_label_cardinality_cannot_broadcast(self, device, dtype):
        class DroppingFlip(K.RandomHorizontalFlip):
            def apply_transform_keypoint(self, input, params, flags, transform=None):
                return Keypoints(input.data[:, :1])

        image = torch.rand(1, 1, 8, 10, device=device, dtype=dtype)
        module = DroppingFlip(p=1)
        aug = K.AugmentationSequential(module, data_keys=["input", "keypoints"])
        with pytest.raises(ValueError, match=r"spatial.*shape"):
            aug.audit(image, image.new_tensor([[[3, 2], [4, 2]]]))
        assert not module._forward_hooks
