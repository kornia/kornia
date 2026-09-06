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

import pytest

BOXMOT_REASON = '`boxmot` is not installed. Install it with: pip install "kornia[tracking]"'


class TestBoxMotTracker:
    """Smoke tests for the ``kornia[tracking]`` extra.

    The tests that touch ``boxmot`` itself skip unless it is installed; they exist so the wrapper
    cannot rot silently once somebody installs the extra.
    """

    def test_wrapper_is_importable(self):
        # The wrapper itself imports `boxmot` lazily, so this runs without the extra installed.
        from kornia.contrib import BoxMotTracker
        from kornia.contrib.boxmot_tracker import BoxMotTracker as BoxMotTrackerDirect

        assert BoxMotTracker is BoxMotTrackerDirect

    def test_lazy_loader_resolves_tracker_class(self):
        pytest.importorskip("boxmot", reason=BOXMOT_REASON)

        from kornia.core.external import boxmot

        tracker_cls = boxmot.DeepOCSORT
        assert isinstance(tracker_cls, type)
        assert tracker_cls.__name__ == "DeepOCSORT"
