import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.feature import *
from kornia.geometry import resize, transform_points
from kornia.testing import assert_close
from kornia.tracking import *


class TestHomographyTracker:
    @pytest.mark.skipif(torch.__version__.startswith('1.6'),
                        reason='1.6.0 not supporting the pretrained weights as they are packed.')
    def test_smoke(self, device):
        tracker = HomographyTracker().to(device)

    def test_nomatch(self, device, dtype):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(SIFTFeature(100),
                                      DescriptorMatcher('smnn', 0.95)).to(device,
                                                                          dtype)
        tracker = HomographyTracker(matcher, matcher, minimum_inliers_num=100)
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        tracker.set_target(data["image0"])
        homography, success = tracker(torch.zeros_like(data["image0"]))
        assert not success

    @pytest.mark.skipif(torch.__version__.startswith('1.6'),
                        reason='1.6.0 not supporting the pretrained weights as they are packed.')
    def test_real(self, device, dtype):
        # This is not unit test, but that is quite good integration test
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000),
                                      DescriptorMatcher('snn', 0.8)).to(device)
        tracker = HomographyTracker(matcher).to(device, dtype)
        data = torch.load("data/test/loftr_outdoor_and_homography_data.pt")
        h0, w0 = data["image0"].shape[2:]
        data["image0"] = resize(data["image0"], (int(h0 // 2), int(w0 // 2)))
        data["image1"] = resize(data["image1"], (int(h0 // 2), int(w0 // 2)))
        tracker.set_target(data["image0"])
        homography, success = tracker(data["image1"])
        assert success
        pts_src = data['pts0'].to(device, dtype) / 2.0
        pts_dst = data['pts1'].to(device, dtype) / 2.0
        # Reprojection error of 5px is OK
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)
        # next frame
        homography, success = tracker(data["image1"])
        assert success
        assert_close(
            transform_points(homography[None], pts_src[None]),
            pts_dst[None],
            rtol=5e-2,
            atol=5)
