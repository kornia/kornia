import pytest
import torch

from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, SIFTFeature
from kornia.geometry import resize, transform_points
from kornia.testing import assert_close
from kornia.tracking import HomographyTracker


@pytest.fixture
def data():
    url = 'https://github.com/kornia/data_test/blob/main/loftr_outdoor_and_homography_data.pt?raw=true'
    return torch.hub.load_state_dict_from_url(url)


class TestHomographyTracker:
    def test_smoke(self, device):
        tracker = HomographyTracker().to(device)
        assert tracker is not None

    def test_nomatch(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(100), DescriptorMatcher('smnn', 0.95)).to(device, dtype)
        tracker = HomographyTracker(matcher, matcher, minimum_inliers_num=100)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        tracker.set_target(data["image0"])
        torch.random.manual_seed(0)
        _, success = tracker(torch.zeros_like(data["image0"]))
        assert not success

    def test_real(self, device, dtype, data):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000), DescriptorMatcher('snn', 0.8)).to(device, dtype)
        tracker = HomographyTracker(matcher, matcher).to(device, dtype)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        h0, w0 = data["image0"].shape[2:]
        data["image0"] = resize(
            data["image0"], (int(h0 // 2), int(w0 // 2)), interpolation='bilinear', align_corners=False
        )
        data["image1"] = resize(
            data["image1"], (int(h0 // 2), int(w0 // 2)), interpolation='bilinear', align_corners=False
        )
        with torch.no_grad():
            tracker.set_target(data["image0"])
            torch.manual_seed(3)  # issue kornia#2027
            homography, success = tracker(data["image1"])
        assert success
        pts_src = data['pts0'].to(device, dtype) / 2.0
        pts_dst = data['pts1'].to(device, dtype) / 2.0
        # Reprojection error of 5px is OK
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)
        # next frame
        with torch.no_grad():
            torch.manual_seed(3)  # issue kornia#2027
            homography, success = tracker(data["image1"])
        assert success
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=5e-2, atol=5)
