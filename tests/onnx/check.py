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
# limitations under the License.
import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))
# ruff: noqa: E402
import test_resize_onnx

test_resize_onnx.test_resize_dynamo_with_binding()
print("Test 1 passed")
test_resize_onnx.test_resize_upscale_dynamo()
print("Test 2 passed")
test_resize_onnx.test_resize_downscale_dynamo()
print("Test 3 passed")
test_resize_onnx.test_resize_nearest_dynamo()
print("Test 4 passed")
print("ALL TESTS PASSED")
