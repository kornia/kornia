import pytest
import torch

from kornia.contrib.sam.prompter import SamPrompter
from kornia.testing import BaseTester


class TestSamPrompter(BaseTester):
    def test_smoke(self, device, dtype):
        inpt = torch.rand(3, 77, 128, device=device, dtype=dtype)
        prompter = SamPrompter('vit_b', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', device)

        prompter.set_image(inpt)
        assert prompter.is_image_set

        prompter.reset_image()
        assert not prompter.is_image_set

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('N', [2, 5])
    @pytest.mark.parametrize('multimask_output', [True, False])
    def test_cardinality(self, dtype, device, batch_size, N, multimask_output):
        inpt = torch.rand(3, 77, 128, device=device, dtype=dtype)
        prompter = SamPrompter('vit_b', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', device)

        keypoints = torch.randint(0, min(inpt.shape[-2:]), (batch_size, N, 2), device=device).to(dtype=dtype)
        labels = torch.randint(0, 1, (batch_size, N), device=device).to(dtype=dtype)

        prompter.set_image(inpt)

        out = prompter.predict((keypoints, labels), multimask_output=multimask_output)

        C = 3 if multimask_output else 1
        assert out.logits.shape == (C, 256, 256)

    def test_exception(self):
        prompter = SamPrompter('vit_b', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')

        inpt = torch.rand(1, 3, 1, 2)

        # Wrong shape for the image
        with pytest.raises(TypeError) as errinfo:
            prompter.set_image(inpt, [], False)
        assert 'shape must be [[\'3\', \'H\', \'W\']]. Got torch.Size([1, 3, 1, 2])' in str(errinfo)

        # predict without set an image
        with pytest.raises(Exception) as errinfo:
            prompter.predict()
        assert 'An image must be set with `self.set_image(...)`' in str(errinfo)

        # Valid masks
        with pytest.raises(TypeError) as errinfo:
            prompter._valid_masks(inpt)
        assert 'shape must be [[\'K\', \'1\', \'256\', \'256\']]. Got torch.Size([1, 3, 1, 2])' in str(errinfo)

        # Valid boxes
        with pytest.raises(TypeError) as errinfo:
            prompter._valid_boxes(inpt)
        assert 'shape must be [[\'K\', \'4\']]. Got torch.Size([1, 3, 1, 2])' in str(errinfo)

        # Valid keypoints
        with pytest.raises(TypeError) as errinfo:
            prompter._valid_keypoints(inpt, None)
        assert 'shape must be [[\'K\', \'N\', \'2\']]. Got torch.Size([1, 3, 1, 2])' in str(errinfo)

        with pytest.raises(TypeError) as errinfo:
            prompter._valid_keypoints(torch.rand(1, 1, 2), inpt)
        assert 'shape must be [[\'K\', \'N\']]. Got torch.Size([1, 3, 1, 2])' in str(errinfo)

        with pytest.raises(Exception) as errinfo:
            prompter._valid_keypoints(torch.rand(1, 1, 2), torch.rand(2, 1))
        assert 'The keypoints and labels should have the same batch size' in str(errinfo)

    @pytest.mark.skip(reason='Unecessary test')
    def test_gradcheck(self, device):
        ...

    @pytest.mark.skip(reason='Unecessary test')
    def test_module(self):
        ...

    @pytest.mark.skip(reason='Needs to be reviewed')
    def test_dynamo(self, device, dtype, torch_optimizer):
        img = torch.rand(3, 128, 75, device=device, dtype=dtype)

        prompter = SamPrompter('vit_b', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', device)
        prompter.set_image(img)

        op = prompter.predict
        op_optimized = torch_optimizer(op)

        expected = op(img)
        actual = op_optimized(img)

        self.assert_close(expected.logits, actual.logits)
        self.assert_close(expected.scores, actual.scores)
