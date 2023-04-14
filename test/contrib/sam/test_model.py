import pytest
import torch

from kornia.contrib import Sam
from kornia.core import pad
from kornia.testing import BaseTester


def _pad_rb(x, size):
    """Pads right bottom."""
    pad_h = size - x.shape[-2]
    pad_w = size - x.shape[-1]
    return pad(x, (0, pad_w, 0, pad_h))


class TestSam(BaseTester):
    def test_smoke(self):
        ...

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('N', [2, 5])
    @pytest.mark.parametrize('multimask_output', [True, False])
    def test_cardinality(self, dtype, device, batch_size, N, multimask_output):
        inpt = torch.rand(1, 3, 77, 128, device=device, dtype=dtype)
        model = Sam.from_pretrained('vit_b', device=device)
        inpt = _pad_rb(inpt, model.image_encoder.img_size)
        keypoints = torch.randint(0, min(inpt.shape[-2:]), (batch_size, N, 2), device=device).to(dtype=dtype)
        labels = torch.randint(0, 1, (batch_size, N), device=device).to(dtype=dtype)

        out = model(inpt, [{'points': (keypoints, labels)}], multimask_output)

        C = 3 if multimask_output else 1
        assert len(out) == inpt.size(0)
        assert out[0].logits.shape == (batch_size, C, 256, 256)

    def test_exception(self):
        model = Sam.build('vit_b')
        with pytest.raises(TypeError) as errinfo:
            inpt = torch.rand(3, 1, 2)
            model(inpt, [], False)
        assert 'shape must be [[\'B\', \'3\', \'H\', \'W\']]. Got torch.Size([3, 1, 2])' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            inpt = torch.rand(2, 3, 1, 2)
            model(inpt, [{}], False)
        assert 'The number of images (`B`) should match with the length of prompts!' in str(errinfo)

    @pytest.mark.parametrize('model_type', ['vit_b', 'vit_l', 'vit_h'])
    def test_build(self, device, model_type):
        model = Sam.build(model_type)
        model = model.to(device=device)

        assert isinstance(model, Sam)
        assert next(model.parameters()).device == device

    @pytest.mark.parametrize(
        'model_type,checkpoint', [('vit_b', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')]
    )
    def test_from_pretrained(self, device, model_type, checkpoint):
        model = Sam.from_pretrained(model_type, checkpoint, device)

        assert isinstance(model, Sam)
        assert next(model.parameters()).device == device

    @pytest.mark.skip(reason='Unsupport at moment -- the code is not tested for trainning and had `torch.no_grad`')
    def test_gradcheck(self, device):
        ...

    @pytest.mark.skip(reason='Unecessary test')
    def test_module(self):
        ...

    @pytest.mark.skip(reason='Needs to be reviewed')
    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(1, 3, 128, 75, device=device, dtype=dtype)

        op = Sam.build('vit_b')
        op = op.to(device=device, dtype=dtype)
        op_optimized = torch_optimizer(op)

        img = _pad_rb(img, op.image_encoder.img_size)

        expected = op(img, [{}], False)
        actual = op_optimized(img, [{}], False)

        self.assert_close(expected[0].logits, actual[0].logits)
        self.assert_close(expected[0].scores, actual[0].scores)

    @pytest.mark.parametrize(
        'model_type,checkpoint', [('vit_b', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')]
    )
    def test_prediction(self, device, dtype, model_type, checkpoint):
        B, C, H, W = 1, 3, 256, 256
        model = Sam.from_pretrained(model_type, checkpoint, device)
        dummy_image = torch.rand(B, C, H, W, device=device, dtype=dtype)
        dummy_image[:, :, 50:150, 50:150] = 1.0
        expected = dummy_image[:, 0:1, ...] == 1.0

        dummy_image = _pad_rb(dummy_image, model.image_encoder.img_size)

        keypoints = torch.tensor([[[100, 100]]], device=device, dtype=dtype)
        labels = torch.tensor([[1]], device=device, dtype=dtype)
        pts = (keypoints, labels)
        boxes = torch.tensor([[[40, 40, 120, 120]]], device=device, dtype=dtype)

        prediction = model(dummy_image, [{'points': pts, 'boxes': boxes}], False)
        padded_size = tuple(dummy_image.shape[-2:])

        prediction[0].original_res_logits(padded_size, (H, W), padded_size)

        self.assert_close(expected, prediction[0].binary_masks)
